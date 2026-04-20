# --- Archivo: query_processor.py ---

import json
import math
import ollama
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


# =====================================================================
# CONFIGURACIÓN DEL PREPROCESAMIENTO ESPACIAL DETERMINISTA
# =====================================================================
# Estos valores son el único sitio donde hay que tocar si se quiere
# recalibrar la semántica espacial (umbrales de distancia, alturas, etc.).
# No se necesita volver a tocar los prompts.

# Coordenadas locales de la cámara (THREE.js convention):
#   -Z = delante   +Z = detrás
#   -X = izquierda +X = derecha
#   -Y = abajo     +Y = arriba

# Umbrales de distancia horizontal en metros (sqrt(x^2 + z^2))
DIST_ARM_REACH = 2.0   # dentro del alcance del brazo
DIST_VERY_CLOSE = 5.0  # muy cerca
DIST_CLOSE = 10.0      # a unos pasos
DIST_MEDIUM = 20.0     # distancia media
# > 20 m se clasifica como "far"

# Umbrales verticales en metros
Y_ABOVE = 1.5   # por encima de la cabeza
Y_BELOW = -0.8  # en el suelo

# Mapa dirección -> frase natural (la que se copia tal cual en la respuesta)
DIRECTION_PHRASE = {
    "front":        "directly in front of you",
    "front-right":  "ahead and to your right",
    "right":        "directly to your right",
    "back-right":   "behind you to your right",
    "behind":       "directly behind you",
    "back-left":    "behind you to your left",
    "left":         "directly to your left",
    "front-left":   "ahead and to your left",
}

DISTANCE_PHRASE = {
    "within_arm_reach": "within arm's reach",
    "very_close":       "very close",
    "close":            "a few steps away",
    "medium":           "at medium distance",
    "far":              "far away",
}

VERTICAL_SUFFIX = {
    "above_eye_level": " (above your eye level)",
    "eye_level":       "",
    "below_eye_level": " (on the floor level)",
}

# Brújula equivalente (front = Norte)
COMPASS_MAP = {
    "front":       "N",
    "front-right": "NE",
    "right":       "E",
    "back-right":  "SE",
    "behind":      "S",
    "back-left":   "SW",
    "left":        "W",
    "front-left":  "NW",
}


class QueryProcessor:
    """
    Encapsula toda la lógica de IA para procesar las consultas del usuario en la escena de VR.

    Incluye un paso de PREPROCESAMIENTO ESPACIAL DETERMINISTA (IA simbólica) que
    se ejecuta ANTES de invocar al MLLM Qwen 2.5 VL. A partir de las coordenadas
    relativas (x, y, z) de cada objeto, se calculan campos semánticos de alto
    nivel (dirección, distancia, posición vertical, frase natural ya redactada)
    y se le entregan ya resueltos al modelo. De este modo Qwen no tiene que
    razonar sobre coordenadas numéricas, tarea en la que los MLLM suelen fallar.
    """

    def __init__(self, modelo_vision="qwen2.5vl:latest"):
        """
        Carga todos los modelos necesarios una sola vez al iniciar el servidor.
        """
        self.modelo_vision = modelo_vision

        # 1. Cargar el Clasificador de Intenciones (tu modelo fine-tuneado)
        print("🔄 Cargando modelo de clasificación de intenciones (fine-tuned)...")
        self.intent_classifier = pipeline(
            "text-classification",
            model="./modelo_vr_guardado",
            tokenizer="./modelo_vr_guardado",
            device=-1
        )
        print("✅ Modelo de clasificación cargado.")

        # 2. Cargar el Traductor Helsinki-NLP
        print("🔄 Cargando modelo de traducción Helsinki-NLP/opus-mt-en-es...")
        translator_checkpoint = "Helsinki-NLP/opus-mt-en-es"
        translator_tokenizer = AutoTokenizer.from_pretrained(translator_checkpoint)
        translator_model = AutoModelForSeq2SeqLM.from_pretrained(translator_checkpoint)
        self.translator = pipeline(
            "translation_en_to_es",
            model=translator_model,
            tokenizer=translator_tokenizer,
            device=0
        )
        print("✅ Modelo de traducción cargado.")

    # -----------------------------------------------------------------
    # PREPROCESAMIENTO ESPACIAL DETERMINISTA (IA SIMBÓLICA)
    # -----------------------------------------------------------------

    @staticmethod
    def _compute_direction(x: float, z: float) -> str:
        """
        Clasifica la posición (x, z) en uno de los 8 sectores angulares.
        Se usa atan2(x, -z) porque -Z es 'delante' en coordenadas de cámara local.
        El ángulo queda en [-180, 180], con 0° = delante, 90° = derecha,
        ±180° = detrás, -90° = izquierda.
        """
        angle = math.degrees(math.atan2(x, -z))

        if -22.5 <= angle < 22.5:
            return "front"
        elif 22.5 <= angle < 67.5:
            return "front-right"
        elif 67.5 <= angle < 112.5:
            return "right"
        elif 112.5 <= angle < 157.5:
            return "back-right"
        elif angle >= 157.5 or angle < -157.5:
            return "behind"
        elif -157.5 <= angle < -112.5:
            return "back-left"
        elif -112.5 <= angle < -67.5:
            return "left"
        else:  # -67.5 <= angle < -22.5
            return "front-left"

    @staticmethod
    def _compute_distance_bucket(horizontal_dist: float) -> str:
        """Clasifica la distancia horizontal (en el plano XZ) en categorías discretas."""
        if horizontal_dist < DIST_ARM_REACH:
            return "within_arm_reach"
        elif horizontal_dist < DIST_VERY_CLOSE:
            return "very_close"
        elif horizontal_dist < DIST_CLOSE:
            return "close"
        elif horizontal_dist < DIST_MEDIUM:
            return "medium"
        else:
            return "far"

    @staticmethod
    def _compute_vertical_position(y: float) -> str:
        """Clasifica la altura relativa del objeto respecto a la cabeza del usuario."""
        if y > Y_ABOVE:
            return "above_eye_level"
        elif y < Y_BELOW:
            return "below_eye_level"
        else:
            return "eye_level"

    @staticmethod
    def _build_position_description(direction: str, dist_bucket: str, vert_pos: str) -> str:
        """Construye una frase natural en inglés lista para que el MLLM la copie tal cual."""
        return (
            f"{DISTANCE_PHRASE[dist_bucket]}, "
            f"{DIRECTION_PHRASE[direction]}"
            f"{VERTICAL_SUFFIX[vert_pos]}"
        )

    def _enrich_spatial_context(self, objetos_visibles: list) -> list:
        """
        Toma la lista cruda de objetos (con relative_position en x, y, z) y
        devuelve una lista enriquecida con campos semánticos ya calculados.
        La salida va ORDENADA por distancia horizontal ascendente (lo más
        cercano primero), lo cual ayuda al MLLM a priorizar de forma natural.

        Estructura de cada objeto enriquecido:
            {
                "label":                "Treasure Chest",
                "description":          "An old wooden chest...",
                "position_description": "very close, ahead and to your right",
                "direction":            "front-right",
                "distance_bucket":      "very_close",
                "angular_sector":       "NE",
                "vertical_position":    "eye_level",
                "is_in_front":          true,
                "horizontal_distance_m": 3.21,
                "contained_objects":    [ ... ]    // opcional
            }
        """
        enriched = []
        for obj in objetos_visibles:
            # Objeto sin posición -> lo pasamos tal cual (edge case)
            if "relative_position" not in obj:
                enriched.append(dict(obj))
                continue

            pos = obj["relative_position"]
            x = float(pos.get("x", 0.0))
            y = float(pos.get("y", 0.0))
            z = float(pos.get("z", 0.0))

            horizontal_dist = math.sqrt(x * x + z * z)
            direction = self._compute_direction(x, z)
            dist_bucket = self._compute_distance_bucket(horizontal_dist)
            vert_pos = self._compute_vertical_position(y)
            position_description = self._build_position_description(
                direction, dist_bucket, vert_pos
            )

            enriched_obj = {
                "label": obj.get("label", "unknown"),
                "description": obj.get("description", ""),
                "position_description": position_description,
                "direction": direction,
                "distance_bucket": dist_bucket,
                "angular_sector": COMPASS_MAP[direction],
                "vertical_position": vert_pos,
                "is_in_front": direction in ("front", "front-left", "front-right"),
                "horizontal_distance_m": round(horizontal_dist, 2),
            }

            # Conservamos los sub-objetos (heredan la posición del padre)
            sub = obj.get("contained_objects")
            if sub:
                enriched_obj["contained_objects"] = [
                    {
                        "label": s.get("label"),
                        "description": s.get("description", ""),
                    }
                    for s in sub
                ]

            enriched.append(enriched_obj)

        # Ordenar por distancia horizontal (más cerca primero)
        enriched.sort(key=lambda o: o.get("horizontal_distance_m", float("inf")))

        return enriched

    # -----------------------------------------------------------------
    # CLASIFICACIÓN DE INTENCIÓN
    # -----------------------------------------------------------------

    def _classify_intent(self, texto_usuario: str) -> str:
        """
        Clasifica el texto del usuario usando el modelo fine-tuneado.
        """
        resultado = self.intent_classifier(texto_usuario)
        intencion = resultado[0]['label']
        confianza = resultado[0]['score']
        print(f"🔀 INTENCIÓN DETECTADA: {intencion} (Confianza: {confianza:.2f})")
        return intencion

    # -----------------------------------------------------------------
    # BLOQUE DE CONTEXTO ESPACIAL COMPARTIDO POR TODOS LOS PROMPTS
    # -----------------------------------------------------------------

    @staticmethod
    def _spatial_fields_docblock() -> str:
        """
        Bloque de documentación común sobre los campos semánticos que incluimos
        en cada objeto. Reemplaza la antigua 'enseñanza' de cómo interpretar
        coordenadas X/Y/Z que se repetía en cada prompt.
        """
        return (
            "Each object in the JSON has been pre-processed and already contains "
            "deterministic spatial fields. You MUST use them directly and NOT try "
            "to compute spatial relations from numbers:\n"
            "  - \"position_description\": a ready-made English phrase "
            "(e.g. \"very close, ahead and to your right\"). "
            "Whenever you need to tell the user where an object is, copy or "
            "lightly adapt THIS phrase — do NOT invent your own.\n"
            "  - \"direction\": one of front | front-right | right | back-right | "
            "behind | back-left | left | front-left\n"
            "  - \"distance_bucket\": one of within_arm_reach | very_close | close | "
            "medium | far\n"
            "  - \"vertical_position\": one of above_eye_level | eye_level | "
            "below_eye_level\n"
            "  - \"is_in_front\": true if the object is in any of the three frontal "
            "sectors (front, front-left, front-right)\n"
            "  - \"horizontal_distance_m\": numeric distance, FOR INTERNAL REASONING "
            "ONLY — never mention meters or numbers in your answer.\n"
            "Objects are already SORTED from nearest to farthest, so the first "
            "items are the most relevant to the user.\n"
            "Some objects include \"contained_objects\": sub-elements inside them "
            "that share the parent's position."
        )

    # -----------------------------------------------------------------
    # GENERACIÓN DE PROMPTS (uno por intención)
    # -----------------------------------------------------------------

    def _generate_prompt(self, intencion: str, texto_usuario: str, objetos_enriquecidos: list) -> str:
        """
        Selecciona y construye el prompt adecuado según la intención clasificada.
        Recibe YA la lista enriquecida (no cruda) — toda la lógica espacial
        está resuelta antes de este paso.
        """
        contexto_json = json.dumps(objetos_enriquecidos, indent=2, ensure_ascii=False)
        spatial_docs = self._spatial_fields_docblock()

        if intencion == "descripcion_escena":
            print("📝 Usando prompt para: Descripción General de Escena")
            return f"""You are an accessibility assistant focused on describing VR scenes for blind users.

Your task is to describe what you see in the IMAGE so that a blind person immersed in the scene can clearly understand it.

SCENE OBJECTS (ground truth — fully reliable, already sorted nearest first):
{contexto_json}

{spatial_docs}

CRITICAL RULES:

1. USE DETECTED OBJECTS AS YOUR BASE
   - Objects appearing in the JSON are unequivocally reliable, they are really in the scene
   - MENTION ONLY objects that appear in the JSON
   - DO NOT invent or add objects not in the list
   - Use the exact labels from the JSON

2. USE THE IMAGE ONLY FOR VISUAL DETAILS. Add details not present in the JSON, if they help a blind person picture the scene:
   - Colors: "dark brown", "bright green", "sky blue"
   - Textures: "aged wood", "rusty metal"
   - Lighting: "well lit", "soft shadows", "bright light"
   - General environment: sky, terrain, atmosphere

3. NEVER mention:
   - Terms like: "video game", "virtual scenario", "game scene"
   - The user ALREADY KNOWS they are immersed in VR
   - Raw coordinates, numbers, meters, or JSON field names

4. SPATIAL LANGUAGE: Use the "position_description" field of each object directly. You may lightly rephrase it to fit the flow of the sentence, but keep the same meaning. Never deduce positions yourself.

5. DESCRIPTION STRUCTURE:
   - First sentence: General context (atmosphere, setting)
   - Second sentence: Main closest objects (first items in the sorted JSON)
   - Third sentence: Secondary or more distant objects
   - Fourth sentence (optional): Environment details
   - Maximum 4 sentences, natural and fluid

6. PRIORITY: Follow the order of the JSON (it is sorted by distance). Closer objects go first, farther ones later or may be omitted.

7. Tone: Descriptive, direct and useful.

CRITICAL: Output your description in ENGLISH. The translation to Spanish will be done automatically."""

        elif intencion == "localizacion_objeto":
            print("📝 Usando prompt para: Localización de Objeto")
            return f"""You are an accessibility assistant helping a blind user locate specific objects inside a VR scene.

USER QUERY: "{texto_usuario}"
YOUR TASK: The user wants to find one or more objects. Identify what they are looking for, locate it in the JSON, and give a precise spatial answer.

SCENE OBJECTS (ground truth — fully reliable, sorted nearest first):
{contexto_json}

{spatial_docs}

CRITICAL RULES:

1. IDENTIFY THE TARGET OBJECT
   - Extract the object name from the user query
   - Search for it (or the closest match) in the JSON
   - If the JSON contains "contained_objects", search inside them too

2. IF THE OBJECT IS FOUND:
   - State clearly that the object is present
   - Give its location by using the "position_description" field DIRECTLY
   - Use the IMAGE to add one brief visual detail that helps the user confirm they found it (color, shape, notable feature)

3. IF THE OBJECT IS NOT FOUND:
   - Clearly state that you cannot detect it in the current scene
   - Do NOT guess or invent a position
   - You may suggest a visually similar object from the JSON if one exists

4. SPATIAL LANGUAGE: Copy the "position_description" of the target object directly. Never invent or deduce spatial relations. Do NOT mention meters, numbers, or coordinates.

5. RESPONSE STRUCTURE:
   - First sentence: Confirm whether the object was found
   - Second sentence: Location taken from "position_description"
   - Third sentence (optional): One visual detail from the IMAGE to help identify it
   - Maximum 3 sentences, concise and direct

6. Tone: Helpful and precise. The user is actively navigating — every word counts.

7. NEVER mention:
   - Terms like "video game", "virtual scenario", "game scene"
   - Raw JSON field names, numeric coordinates or meters

CRITICAL: Output your response in ENGLISH. The translation to Spanish will be done automatically."""

        elif intencion == "detalle_objeto":
            print("📝 Usando prompt para: Detalle de Objeto")
            return f"""You are an accessibility assistant providing detailed information about a specific object to a blind user in a VR scene.

USER QUERY: "{texto_usuario}"
YOUR TASK: The user wants to know more about a specific object. Identify it, find it in the JSON, and enrich the answer with visual details from the IMAGE.

SCENE OBJECTS (ground truth — fully reliable, sorted nearest first):
{contexto_json}

{spatial_docs}

CRITICAL RULES:

1. IDENTIFY THE TARGET OBJECT
   - Extract the object the user is asking about from their query
   - Find it (or the closest match) in the JSON, including inside "contained_objects"
   - If it is not in the JSON, state clearly that you cannot detect it

2. DESCRIBE THE OBJECT IN DETAIL — in this order of priority:
   a) Structural info from JSON: label, contained sub-objects, any metadata present
   b) Visual details from the IMAGE:
      - Color and finish: "dark worn leather", "shiny brass", "faded red paint"
      - Texture and material: "rough stone", "smooth polished wood", "rusty iron"
      - Shape and size (relative): "small and cylindrical", "wide and flat"
      - Condition or state: "slightly open", "cracked", "glowing faintly"
      - Notable features: handles, locks, engravings, markings, signs of use
   c) Spatial context (optional): use the "position_description" field as-is

3. CONTAINED OBJECTS:
   - If the target has "contained_objects", mention what is inside
   - Example: "The chest is closed, but its latch appears unlocked. Inside you might find..."

4. IF THE OBJECT IS NOT IN THE JSON:
   - Do NOT describe it from the image alone
   - State that you cannot confirm it is in the scene
   - Do NOT invent details

5. SPATIAL LANGUAGE (only if needed for context):
   - Copy the "position_description" field of the object. Never deduce positions.
   - Do NOT mention meters or numbers.

6. RESPONSE STRUCTURE:
   - First sentence: Identify and confirm the object
   - Second and third sentences: Rich visual and structural description
   - Fourth sentence (optional): Spatial location taken from "position_description"
   - Maximum 4 sentences, descriptive and natural
   - Your response must be conversational communicating the details directly to the user

7. Tone: Detailed and sensory-rich. Help the user form a clear mental image.

8. NEVER mention:
   - Terms like "video game", "virtual scenario", "game scene"
   - Raw JSON field names or numeric coordinates

CRITICAL: Output your response in ENGLISH. The translation to Spanish will be done automatically."""

        elif intencion == "objetos_cercanos":
            print("📝 Usando prompt para: Objetos Cercanos")
            return f"""You are an accessibility assistant helping a blind user understand their immediate surroundings inside a VR scene.

USER QUERY: "{texto_usuario}"
YOUR TASK: The user wants to know what objects are close to them right now. Focus exclusively on the nearest objects and describe them clearly.

SCENE OBJECTS (ground truth — fully reliable, sorted nearest first):
{contexto_json}

{spatial_docs}

CRITICAL RULES:

1. SELECT ONLY THE NEAREST OBJECTS
   - The JSON is already sorted by proximity. Focus on the FIRST 3–4 objects.
   - Consider an object "near" if its "distance_bucket" is within_arm_reach, very_close, or close.
   - You may mention an object with distance_bucket = "medium" only if there are fewer than 2 nearer objects.
   - IGNORE objects with distance_bucket = "far".
   - IGNORE objects where "is_in_front" is false, unless nothing relevant is in front.

2. FOR EACH NEARBY OBJECT:
   - State its name and its location using the "position_description" field
   - Add one or two visual details from the IMAGE: color, material, size, condition
   - If it has "contained_objects", briefly mention what is inside

3. SPATIAL LANGUAGE: Copy the "position_description" field of each object. Never deduce. Never mention meters or numbers.

4. PRIORITIZATION:
   - The first object in the JSON is the closest — mention it first
   - Mention at most 3–4 objects to avoid overwhelming the user

5. RESPONSE STRUCTURE:
   - First sentence: Quick count and general impression ("There are two objects very close to you...")
   - One sentence per nearby object: name + position (from position_description) + one visual detail
   - Last sentence (optional): Brief note if nothing is immediately close
   - Maximum 5 sentences, clear and easy to act on

6. Tone: Practical and spatial. The user may be about to move — help them navigate safely.

7. NEVER mention:
   - Objects that are far away or behind (unless nothing is close)
   - Terms like "video game", "virtual scenario", "game scene"
   - Raw JSON field names, numeric coordinates or meters

CRITICAL: Output your response in ENGLISH. The translation to Spanish will be done automatically."""

        elif intencion == "navegacion":
            print("📝 Usando prompt para: Navegación")
            return f"""You are an accessibility assistant helping a blind user move safely through a VR scene.

USER QUERY: "{texto_usuario}"
YOUR TASK: The user wants to move somewhere, reach an object, or understand how to get from their current position to a destination. Give clear, actionable movement instructions based on the scene data.

SCENE OBJECTS (ground truth — fully reliable, sorted nearest first):
{contexto_json}

{spatial_docs}

CRITICAL RULES:

1. IDENTIFY THE NAVIGATION GOAL
   - Determine where the user wants to go or what they want to reach
   - Find the target object in the JSON
   - If the destination is not in the JSON, state that you cannot locate it

2. GIVE STEP-BY-STEP MOVEMENT INSTRUCTIONS:
   - Use the "direction" and "distance_bucket" of the target
     - "direction" tells you which way to turn and walk
     - "distance_bucket" tells you how far in natural terms
   - Break the path into 1–2 simple, sequential steps
   - If the target is already "is_in_front": true, say the path is direct

3. WARN ABOUT OBSTACLES:
   - Check other objects whose "direction" is the same as the target's and whose "horizontal_distance_m" is smaller than the target's.
   - Those are potential obstacles on the way. Mention them so the user can avoid them.

4. SPATIAL LANGUAGE:
   - Translate "direction" into natural instructions: e.g. "front-left" -> "turn slightly left and walk forward".
   - Translate "distance_bucket" into natural phrases (use "position_description" if in doubt).
   - Never mention meters, numbers, or coordinates.

5. USE THE IMAGE for additional navigation cues:
   - Visible landmarks, light sources, or distinctive colors that help orientation
   - Example: "Head toward the brighter area on your left — that is where the exit is."
   - DO NOT invent objects not present in the JSON.

6. RESPONSE STRUCTURE:
   - First sentence: Confirm the target and its general direction
   - Second and third sentences: Step-by-step movement instructions
   - Fourth sentence (optional): Obstacle warning or a visual landmark to confirm arrival
   - Maximum 4 sentences, clear and action-oriented

7. Tone: Calm, confident, and guiding. The user is moving through a space they cannot see — be their eyes.

8. NEVER mention:
   - Terms like "video game", "virtual scenario", "game scene"
   - Raw JSON field names, numeric coordinates or meters

CRITICAL: Output your response in ENGLISH. The translation to Spanish will be done automatically."""

        else:  # Fallback
            print("📝 Usando prompt para: Fallback")
            return f"""You are an accessibility assistant helping a blind user in a VR scene.

USER QUERY: "{texto_usuario}"
YOUR TASK: Answer the user's question as best you can using the available scene data and image.

SCENE OBJECTS (ground truth — fully reliable, sorted nearest first):
{contexto_json}

{spatial_docs}

Use the "position_description" field directly whenever you describe where something is. Do NOT compute positions from numbers. Do NOT mention meters or coordinates.
Do NOT mention "video game", "virtual scenario", or "game scene".

CRITICAL: Output your response in ENGLISH. The translation to Spanish will be done automatically."""

    # -----------------------------------------------------------------
    # LLAMADAS A MODELOS
    # -----------------------------------------------------------------

    def _query_vision_model(self, prompt: str, ruta_imagen: str) -> str:
        """
        Consulta el modelo de visión multimodal con el prompt y la imagen.
        """
        print(f"🧠 Analizando escena con {self.modelo_vision}...")
        try:
            response = ollama.chat(
                model=self.modelo_vision,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [ruta_imagen]
                }],
                options={'temperature': 0.2}
            )
            return response['message']['content']
        except Exception as e:
            print(f"⚠️ Error en la consulta a Ollama: {e}")
            return "Error al generar la descripción desde el modelo de visión."

    def _translate_to_spanish(self, texto_ingles: str) -> str:
        """
        Traduce el texto de inglés a español.
        """
        print("🌐 Traduciendo a Español con Helsinki-NLP...")
        try:
            resultado = self.translator(texto_ingles, max_length=512)
            return resultado[0]['translation_text']
        except Exception as e:
            print(f"⚠️ Error en la traducción: {e}")
            return texto_ingles

    # -----------------------------------------------------------------
    # PIPELINE COMPLETO
    # -----------------------------------------------------------------

    def process(self, texto_usuario: str, ruta_imagen: str, objetos_visibles: list) -> dict:
        """
        Orquesta el proceso completo:
          1) Preprocesamiento espacial determinista (IA simbólica) -> enriquece JSON
          2) Clasificación de intención
          3) Generación de prompt dinámico con JSON enriquecido
          4) Consulta al MLLM (Qwen 2.5 VL)
          5) Traducción al español
        """
        # 1. Enriquecer espacialmente ANTES de cualquier otra cosa
        objetos_enriquecidos = self._enrich_spatial_context(objetos_visibles)
        print(f"🧭 Preprocesamiento espacial: {objetos_enriquecidos}")

        # 2. Clasificar la intención del usuario
        intencion = self._classify_intent(texto_usuario)

        # 3. Generar el prompt dinámico basado en la intención (con JSON enriquecido)
        prompt = self._generate_prompt(intencion, texto_usuario, objetos_enriquecidos)

        # 4. Consultar al modelo de visión (Ollama)
        descripcion_ingles = self._query_vision_model(prompt, ruta_imagen)

        # 5. Traducir la descripción al español
        descripcion_espanol = self._translate_to_spanish(descripcion_ingles)

        return {
            "descripcion": descripcion_espanol,
            "intencion": intencion
        }