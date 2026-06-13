"""
Orquestador del asistente VR.

Versión para el Benchmark 1 (Efecto del contexto): acepta `context_level`
∈ {"C1","C2","C3","C4"}. Cada nivel usa un PROMPT COMPLETO E INDEPENDIENTE,
definido en core/prompts.py (el bloque C4 es el prompt original de producción).

Niveles de contexto:
  C1 -> solo imagen           (el prompt no menciona objetos)
  C2 -> imagen + lista         (label + description, sin posición)
  C3 -> imagen + coords crudas (relative_position en metros)
  C4 -> imagen + enriquecido   (spatial_enricher + spatial_docs)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, List, Any

import ollama
from transformers import pipeline

from core import spatial_enricher
from core.prompts import PROMPTS, SPATIAL_DOCS
from paths import MODELO_DIR


# =====================================================================
# CONFIGURACIÓN
# =====================================================================

MODELO_CLASIFICADOR_DIR = str(MODELO_DIR)

CONFIDENCE_THRESHOLD = 0.15
OOD_LABEL = "fuera_dominio"

OOD_RESPONSE_ES = (
    "Lo siento, solo puedo ayudarte con preguntas sobre la escena en la "
    "que te encuentras: describirla, localizar objetos, darte detalles "
    "de lo que hay cerca o guiarte para moverte. ¿Puedes reformular tu "
    "pregunta en esos términos?"
)

# Mapeo intención -> categoría de prompt. Las no listadas caen en "fallback".
INTENT_TO_PROMPT: Dict[str, str] = {
    "descripcion_escena":  "descripcion_escena",
    "localizacion_objeto": "localizacion_objeto",
    "detalle_objeto":      "detalle_objeto",
    "objetos_cercanos":    "objetos_cercanos",
    "navegacion":          "navegacion",
}

CONTEXT_LEVELS = {"C1", "C2", "C3", "C4"}
DEFAULT_CONTEXT_LEVEL = "C4"   # comportamiento en producción

# Niveles que incluyen el placeholder {contexto_json} en su prompt.
_LEVELS_WITH_CONTEXT = {"C2", "C3", "C4"}
# Niveles que incluyen el placeholder {spatial_docs}.
_LEVELS_WITH_SPATIAL_DOCS = {"C4"}


class QueryProcessor:

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
            device=-1,
        )

        print("🔄 Cargando prompts y docs espaciales...")
        self._validate_prompts()
        self._spatial_docs = SPATIAL_DOCS

        print("✅ QueryProcessor listo.")

    @staticmethod
    def _validate_prompts() -> None:
        """
        Fail-fast: comprueba al arrancar que cada categoría operativa tiene los
        4 niveles definidos en core/prompts.py. 'fallback' también debe estar.
        """
        requeridas = set(INTENT_TO_PROMPT.values()) | {"fallback"}
        for cat in requeridas:
            if cat not in PROMPTS:
                raise KeyError(f"Falta la categoría de prompt '{cat}' en core/prompts.py")
            faltan = CONTEXT_LEVELS - set(PROMPTS[cat].keys())
            if faltan:
                raise KeyError(
                    f"La categoría '{cat}' no define los niveles {sorted(faltan)} "
                    f"en core/prompts.py"
                )

    # -----------------------------------------------------------------
    # PASO 1: CLASIFICACIÓN DE INTENCIÓN (con umbral OOD)
    # -----------------------------------------------------------------

    def _classify_intent(self, texto_usuario: str) -> Tuple[str, float]:
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
    # PASO 2: CONSTRUCCIÓN DEL CONTEXTO (JSON de objetos) SEGÚN NIVEL
    # -----------------------------------------------------------------

    def _build_context_json(
        self,
        objetos_visibles: List[Dict[str, Any]],
        level: str,
    ) -> str:
        """
        Devuelve el string JSON de objetos que se inyecta en {contexto_json}.
        C1 no usa contexto (devuelve cadena vacía). C2/C3/C4 producen distintos
        niveles de detalle. La forma del prompt la decide core/prompts.py.
        """
        if level == "C1":
            return ""

        if level == "C2":
            objetos = [
                {"label": o.get("label", "unknown"),
                 "description": o.get("description", "")}
                for o in objetos_visibles
            ]
            return json.dumps(objetos, indent=2, ensure_ascii=False)

        if level == "C3":
            objetos: List[Dict[str, Any]] = []
            for o in objetos_visibles:
                obj_out = {
                    "label": o.get("label", "unknown"),
                    "description": o.get("description", ""),
                }
                if "relative_position" in o:
                    obj_out["relative_position"] = o["relative_position"]
                if o.get("contained_objects"):
                    obj_out["contained_objects"] = [
                        {"label": s.get("label"),
                         "description": s.get("description", "")}
                        for s in o["contained_objects"]
                    ]
                objetos.append(obj_out)
            return json.dumps(objetos, indent=2, ensure_ascii=False)

        if level == "C4":
            enriquecidos = spatial_enricher.enrich_objects(objetos_visibles)
            return json.dumps(enriquecidos, indent=2, ensure_ascii=False)

        raise ValueError(
            f"Nivel de contexto desconocido: {level!r}. "
            f"Esperado uno de {sorted(CONTEXT_LEVELS)}."
        )

    # -----------------------------------------------------------------
    # PASO 3: SELECCIÓN Y RENDER DEL PROMPT
    # -----------------------------------------------------------------

    def _build_prompt(
        self,
        intencion: str,
        texto_usuario: str,
        objetos_visibles: List[Dict[str, Any]],
        level: str,
    ) -> Tuple[str, str]:
        """
        Selecciona el prompt completo de (categoría, nivel) y lo rellena solo
        con los placeholders que ese nivel usa. Devuelve (prompt, categoria).
        """
        categoria = INTENT_TO_PROMPT.get(intencion, "fallback")
        template = PROMPTS[categoria][level]

        # Construir solo los placeholders que el prompt de este nivel necesita.
        fmt: Dict[str, str] = {"texto_usuario": texto_usuario}
        if level in _LEVELS_WITH_CONTEXT:
            fmt["contexto_json"] = self._build_context_json(objetos_visibles, level)
        if level in _LEVELS_WITH_SPATIAL_DOCS:
            fmt["spatial_docs"] = self._spatial_docs

        print(f"📝 Prompt: {categoria} ({level})")
        # format_map ignora placeholders ausentes en el texto, y como solo
        # pasamos los que el nivel usa, no hay riesgo de KeyError.
        prompt = template.format(**fmt)
        print(prompt)
        return prompt, categoria

    # -----------------------------------------------------------------
    # PASO 4: CONSULTA AL MLLM
    # -----------------------------------------------------------------

    def _query_mllm(self, prompt: str, ruta_imagen: str, temperature: float, seed: int) -> str:
        print(f"🧠 Consultando {self.modelo_vision}...")
        try:
            response = ollama.chat(
                model=self.modelo_vision,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [ruta_imagen],
                }],
                options={"temperature": temperature, "seed": seed},
            )
            return response["message"]["content"]
        except Exception as e:
            print(f"⚠️  Error en Ollama: {e}")
            return "Error al generar la descripción desde el modelo de visión."

    # -----------------------------------------------------------------
    # PIPELINE COMPLETO
    # -----------------------------------------------------------------

    def process(
        self,
        texto_usuario: str,
        ruta_imagen: str,
        objetos_visibles: list,
        context_level: str = DEFAULT_CONTEXT_LEVEL,
        temperature: float = 0.1,
        seed: int = 42,
    ) -> dict:
        if context_level not in CONTEXT_LEVELS:
            raise ValueError(
                f"context_level inválido: {context_level!r}. "
                f"Esperado uno de {sorted(CONTEXT_LEVELS)}."
            )

        # 1. Intención
        intencion, confianza = self._classify_intent(texto_usuario)

        # 2. OOD short-circuit
        if intencion == OOD_LABEL:
            print("🛑 Consulta fuera de dominio. Respuesta canned.")
            return {
                "descripcion": OOD_RESPONSE_ES,
                "intencion": intencion,
                "confianza": confianza,
                "ood": True,
                "context_level": context_level,
                "prompt_template": None,
            }

        # 3-4. Prompt completo del nivel + render
        prompt, categoria = self._build_prompt(
            intencion, texto_usuario, objetos_visibles, context_level
        )

        # 5. MLLM (genera la respuesta directamente en español)
        descripcion = self._query_mllm(prompt, ruta_imagen, temperature, seed)

        return {
            "descripcion": descripcion,
            "intencion": intencion,
            "confianza": confianza,
            "ood": False,
            "context_level": context_level,
            "prompt_template": categoria,
        }