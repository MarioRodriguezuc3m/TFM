"""
Orquestador del asistente VR.

Responsabilidad única: coordinar los componentes que procesan una consulta
del usuario. Toda la lógica específica vive fuera de este archivo:

  - Enriquecimiento espacial         -> spatial_enricher.py
  - Plantillas de prompt             -> prompts/*.txt
  - Pesos del clasificador           -> modelo_vr_guardado/
  - Inferencia del MLLM              -> Ollama (Qwen 2.5 VL)
  - Traducción                       -> Helsinki-NLP/opus-mt-en-es

El pipeline completo está en `process()` y tiene 6 pasos numerados.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import ollama
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)

import spatial_enricher


# =====================================================================
# CONFIGURACIÓN
# =====================================================================

PROMPTS_DIR = Path(__file__).parent / "prompts"
MODELO_CLASIFICADOR_DIR = "./modelo_vr_guardado"
MODELO_TRADUCTOR_ID = "Helsinki-NLP/opus-mt-en-es"

# Umbral por debajo del cual una predicción del clasificador se trata como
# fuera de dominio aunque la etiqueta asignada sea una de las "válidas".
CONFIDENCE_THRESHOLD = 0.55
OOD_LABEL = "fuera_dominio"

OOD_RESPONSE_ES = (
    "Lo siento, solo puedo ayudarte con preguntas sobre la escena en la "
    "que te encuentras: describirla, localizar objetos, darte detalles "
    "de lo que hay cerca o guiarte para moverte. ¿Puedes reformular tu "
    "pregunta en esos términos?"
)

# Mapeo intención -> nombre del archivo de prompt (sin extensión).
# Las intenciones no listadas caen en "fallback".
INTENT_TO_PROMPT: Dict[str, str] = {
    "descripcion_escena":  "descripcion_escena",
    "localizacion_objeto": "localizacion_objeto",
    "detalle_objeto":      "detalle_objeto",
    "objetos_cercanos":    "objetos_cercanos",
    "navegacion":          "navegacion",
    # "fuera_dominio"      -> fallback
}


class QueryProcessor:
    """
    Encapsula todo el pipeline: clasificación de intención, enriquecimiento
    espacial, construcción del prompt, consulta al MLLM y traducción al
    español. El frontend solo necesita llamar a `process()`.
    """

    # -----------------------------------------------------------------
    # CICLO DE VIDA
    # -----------------------------------------------------------------

    def __init__(self, modelo_vision: str = "qwen2.5vl:latest"):
        self.modelo_vision = modelo_vision

        print("🔄 Cargando clasificador de intenciones (fine-tuned)...")
        self._intent_classifier = pipeline(
            "text-classification",
            model=MODELO_CLASIFICADOR_DIR,
            tokenizer=MODELO_CLASIFICADOR_DIR,
            device=-1,  # CPU: BETO es ligero y esto libera GPU para Qwen
        )

        print(f"🔄 Cargando traductor {MODELO_TRADUCTOR_ID}...")
        tok = AutoTokenizer.from_pretrained(MODELO_TRADUCTOR_ID)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(MODELO_TRADUCTOR_ID)
        self._translator = pipeline(
            "translation_en_to_es",
            model=mdl,
            tokenizer=tok,
            device=0,
        )

        print("🔄 Cargando plantillas de prompt...")
        self._prompt_templates = self._load_all_prompts()

        print("✅ QueryProcessor listo.")

    @staticmethod
    def _load_all_prompts() -> Dict[str, str]:
        """
        Carga todas las plantillas a memoria al arrancar. Si falta algún
        archivo, el error se dispara aquí (fail-fast) en lugar de en la
        primera consulta del usuario.
        """
        requeridos = set(INTENT_TO_PROMPT.values()) | {"fallback"}
        templates: Dict[str, str] = {}

        for nombre in requeridos:
            ruta = PROMPTS_DIR / f"{nombre}.txt"
            if not ruta.is_file():
                raise FileNotFoundError(
                    f"Plantilla de prompt no encontrada: {ruta}. "
                    f"Revisa la carpeta {PROMPTS_DIR}."
                )
            templates[nombre] = ruta.read_text(encoding="utf-8")

        # El bloque de docs espaciales se inyecta en todos los prompts
        spatial_docs_path = PROMPTS_DIR / "_spatial_docs.txt"
        if not spatial_docs_path.is_file():
            raise FileNotFoundError(
                f"Falta el bloque de documentación espacial: {spatial_docs_path}"
            )
        templates["_spatial_docs"] = spatial_docs_path.read_text(encoding="utf-8")

        return templates

    # -----------------------------------------------------------------
    # PASO 1: CLASIFICACIÓN DE INTENCIÓN (con umbral OOD)
    # -----------------------------------------------------------------

    def _classify_intent(self, texto_usuario: str) -> Tuple[str, float]:
        """
        Devuelve (intencion, confianza). Si la confianza es inferior al
        umbral, la intención se reetiqueta como OOD para que el pipeline
        la trate como consulta fuera de dominio.
        """
        resultado = self._intent_classifier(texto_usuario)[0]
        intencion = resultado["label"]
        confianza = float(resultado["score"])

        if confianza < CONFIDENCE_THRESHOLD and intencion != OOD_LABEL:
            print(
                f"⚠️  Confianza baja ({confianza:.2f} < {CONFIDENCE_THRESHOLD}). "
                f"Reetiquetando '{intencion}' como '{OOD_LABEL}'."
            )
            intencion = OOD_LABEL

        print(f"🔀 Intención: {intencion} (confianza {confianza:.2f})")
        return intencion, confianza

    # -----------------------------------------------------------------
    # PASO 3: CONSTRUCCIÓN DEL PROMPT
    # -----------------------------------------------------------------

    def _build_prompt(self, intencion: str, texto_usuario: str, objetos_enriquecidos: list) -> str:
        """Selecciona la plantilla según la intención y la renderiza."""
        template_name = INTENT_TO_PROMPT.get(intencion, "fallback")
        template = self._prompt_templates[template_name]

        print(f"📝 Prompt: {template_name}.txt")
        return template.format(
            texto_usuario=texto_usuario,
            contexto_json=json.dumps(objetos_enriquecidos, indent=2, ensure_ascii=False),
            spatial_docs=self._prompt_templates["_spatial_docs"],
        )

    # -----------------------------------------------------------------
    # PASO 4: CONSULTA AL MLLM
    # -----------------------------------------------------------------

    def _query_mllm(self, prompt: str, ruta_imagen: str) -> str:
        """Envía prompt + imagen al modelo de visión multimodal."""
        print(f"🧠 Consultando {self.modelo_vision}...")
        try:
            response = ollama.chat(
                model=self.modelo_vision,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [ruta_imagen],
                }],
                options={"temperature": 0.2},
            )
            return response["message"]["content"]
        except Exception as e:
            print(f"⚠️  Error en Ollama: {e}")
            return "Error al generar la descripción desde el modelo de visión."

    # -----------------------------------------------------------------
    # PASO 5: TRADUCCIÓN
    # -----------------------------------------------------------------

    def _translate(self, texto_ingles: str) -> str:
        print("🌐 Traduciendo al español...")
        try:
            return self._translator(texto_ingles, max_length=512)[0]["translation_text"]
        except Exception as e:
            print(f"⚠️  Error en traducción: {e}")
            return texto_ingles

    # -----------------------------------------------------------------
    # PIPELINE COMPLETO (entry point público)
    # -----------------------------------------------------------------

    def process(self, texto_usuario: str, ruta_imagen: str, objetos_visibles: list) -> dict:
        """
        Pipeline completo:
          1) Clasificar intención (con umbral OOD)
          2) Si es OOD -> respuesta canned, fin
          3) Enriquecer objetos espacialmente
          4) Construir prompt a partir de la plantilla
          5) Consultar al MLLM
          6) Traducir al español
        """
        # 1. Intención
        intencion, confianza = self._classify_intent(texto_usuario)

        # 2. Short-circuit: fuera de dominio no merece invocar al MLLM
        if intencion == OOD_LABEL:
            print("🛑 Consulta fuera de dominio. Respuesta canned.")
            return {
                "descripcion": OOD_RESPONSE_ES,
                "intencion": intencion,
                "confianza": confianza,
                "ood": True,
            }

        # 3. Enriquecer contexto espacial
        objetos_enriquecidos = spatial_enricher.enrich_objects(objetos_visibles)

        # 4-6. Prompt -> MLLM -> traducción
        prompt = self._build_prompt(intencion, texto_usuario, objetos_enriquecidos)
        descripcion_ingles = self._query_mllm(prompt, ruta_imagen)
        descripcion_espanol = self._translate(descripcion_ingles)

        return {
            "descripcion": descripcion_espanol,
            "intencion": intencion,
            "confianza": confianza,
            "ood": False,
        }