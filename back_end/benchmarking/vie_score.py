"""
VIEScore para descripciones de texto (adaptación estilo AskVR), con Gemini
como único juez.

VIEScore (Ku et al., ACL 2024, arXiv:2312.14867) es un evaluador
"MLLM-as-judge" que NO requiere entrenamiento. Produce dos ejes en escala
0-10 y los combina mediante media geométrica:

    SC (Semantic Consistency)  = min(sub-scores de consistencia semántica)
    PQ (Perceptual Quality)    = min(sub-scores de calidad perceptual)
    O  (Overall)               = sqrt(SC * PQ)

VIEScore se diseñó originalmente para evaluar IMÁGENES generadas. AskVR
(Fernandez et al., MMM 2026) lo reutiliza para evaluar DESCRIPCIONES DE
TEXTO, reinterpretando los ejes como:

    SC -> exactitud del contenido y alineamiento con la semántica de la imagen
    PQ -> realismo, fluidez y comprensibilidad de la descripción generada

Este módulo replica ese enfoque. El juez (Gemini) recibe en una sola pasada:
imagen de la escena + consulta del usuario + (opcional) contexto estructurado
+ la RESPUESTA DEL SISTEMA a evaluar. Devuelve sub-scores con su rationale.

Como tu sistema genera las respuestas con Qwen, usar Gemini como juez evita el
sesgo de auto-evaluación (juez != generador), lo cual es metodológicamente más
defendible.

Requisitos:
    pip install google-genai
    Una o varias API keys (gratis en https://aistudio.google.com/apikey),
    definidas en el entorno o en un .env:
      - GEMINI_API_KEYS = "key1,key2,key3"  (varias cuentas, separadas por comas)
      - o GEMINI_API_KEY = "key"            (una sola)
    Con varias keys, el juez rota automáticamente a la siguiente cuando una
    agota su cuota DIARIA, lo que permite evaluar más ítems por día.

Nota sobre modelos: el juez usa por defecto el modelo Flash más capaz del nivel
gratuito. Los nombres de Gemini cambian con frecuencia; si Google renombra el
modelo, ajusta DEFAULT_MODEL o pásalo con --model. Comprueba los vigentes en
https://ai.google.dev/gemini-api/docs/models
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# Modelo juez por defecto: versión GA estable de Flash (no preview), para que
# los nombres no cambien a mitad de las pasadas del benchmark. Alternativa más
# barata: "gemini-3.1-flash-lite". Comprueba los vigentes en
# https://ai.google.dev/gemini-api/docs/models
DEFAULT_MODEL = "gemini-3.5-flash"


# =====================================================================
# RÚBRICAS (sub-aspectos de cada eje). Editar aquí para recalibrar.
# Cada sub-aspecto se puntúa de 0 a 10. SC y PQ toman el MÍNIMO de sus
# sub-aspectos (un resultado es tan bueno como su peor dimensión).
# =====================================================================

# Sub-aspectos de Semantic Consistency (alineamiento con la imagen + consulta).
# spatial_correctness se aplica a TODAS las categorías: el juez solo lo evalúa
# si la respuesta menciona alguna dirección/posición; si no menciona ninguna,
# no penaliza (lo considera plenamente satisfecho).
SC_ASPECTS_BASE = [
    ("grounding",
     "Factual grounding: every object, attribute and fact stated in the "
     "response must be actually present in the scene image. Penalise "
     "hallucinations (things mentioned that are not in the image)."),
    ("query_relevance",
     "Query relevance: the response must actually address the user's "
     "question/intent, not give unrelated information."),
    ("spatial_correctness",
     "Spatial correctness: IF the response mentions ANY direction "
     "(left/right/front/behind/above/below) or relative position/distance, "
     "you MUST verify it against the object's real position in the scene "
     "image and the structured scene data, and penalise inverted or wrong "
     "directions. IF the response makes no spatial claim at all (e.g. it only "
     "describes an object's appearance or colour), this aspect does not apply: "
     "consider it fully satisfied and give it 10."),
]

# Sub-aspectos de Perceptual Quality (realismo, fluidez, comprensibilidad)
PQ_ASPECTS = [
    ("fluency",
     "Fluency and naturalness: the text must be well-formed, grammatical and "
     "read naturally for the target language."),
    ("comprehensibility",
     "Comprehensibility and usefulness for a blind/low-vision user: clear, "
     "unambiguous, with an appropriate level of detail (neither too sparse "
     "nor overly verbose), and easy to act upon when listened to."),
]


# =====================================================================
# CONSTRUCCIÓN DE PROMPTS PARA EL JUEZ
# =====================================================================

_SYSTEM_PREAMBLE = (
    "You are a strict evaluator of accessibility descriptions for blind and "
    "low-vision (BLV) users in a virtual-reality scene. You will be shown the "
    "scene image, the user's spoken request, optional structured scene data, "
    "and an AI-generated text response. You must rate the response."
)

_SCORE_INSTRUCTIONS = (
    "Rate EACH listed aspect on an integer scale from 0 to 10 "
    "(0 = completely fails the aspect, 10 = perfectly satisfies it). "
    "First think briefly, then output ONLY a JSON object on the last line, "
    "with no markdown fences, exactly in this format:\n"
    '{{"score": [{score_slots}], "reasoning": "<one short sentence>"}}\n'
    "The score list must have exactly {n} integers, in the same order as the "
    "aspects listed above."
)


def _format_aspects(aspects: List[tuple]) -> str:
    return "\n".join(f"  {i+1}. {name}: {desc}"
                     for i, (name, desc) in enumerate(aspects))


def build_sc_prompt(query: str, response: str, context: str,
                    aspects: List[tuple]) -> str:
    score_slots = ", ".join("score" for _ in aspects)
    return (
        f"{_SYSTEM_PREAMBLE}\n\n"
        f"EVALUATION AXIS: SEMANTIC CONSISTENCY (SC).\n"
        f"Judge how accurately the response reflects the scene image and how "
        f"well it answers the request. Evaluate these aspects:\n"
        f"{_format_aspects(aspects)}\n\n"
        f"USER REQUEST:\n{query}\n\n"
        f"STRUCTURED SCENE DATA (ground-truth objects, may be empty):\n"
        f"{context or '[none provided]'}\n\n"
        f"AI-GENERATED RESPONSE TO EVALUATE:\n{response}\n\n"
        + _SCORE_INSTRUCTIONS.format(score_slots=score_slots, n=len(aspects))
    )


def build_pq_prompt(query: str, response: str,
                    aspects: List[tuple]) -> str:
    score_slots = ", ".join("score" for _ in aspects)
    return (
        f"{_SYSTEM_PREAMBLE}\n\n"
        f"EVALUATION AXIS: PERCEPTUAL QUALITY (PQ).\n"
        f"Judge the realism, fluency and comprehensibility of the response as "
        f"a piece of spoken accessibility feedback. Do NOT judge factual "
        f"accuracy here (that is handled separately). Evaluate these aspects:\n"
        f"{_format_aspects(aspects)}\n\n"
        f"USER REQUEST:\n{query}\n\n"
        f"AI-GENERATED RESPONSE TO EVALUATE:\n{response}\n\n"
        + _SCORE_INSTRUCTIONS.format(score_slots=score_slots, n=len(aspects))
    )


# =====================================================================
# PARSEO ROBUSTO DE LA SALIDA DEL JUEZ
# =====================================================================

def parse_score_json(raw: str, n_expected: int) -> Dict[str, Any]:
    """
    Extrae el último objeto JSON con la forma {"score": [...], "reasoning": ...}.
    Es tolerante a fences de markdown y a texto previo (rationale).
    Devuelve {"scores": [float,...], "reasoning": str}. Si falla, scores=None.
    """
    cleaned = raw.replace("```json", "").replace("```", "")
    # Busca bloques tipo {... score ...} con comillas dobles o simples
    candidates = re.findall(r"\{[^{}]*['\"]score['\"][^{}]*\}", cleaned, flags=re.DOTALL)
    for blob in reversed(candidates):
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            try:
                data = json.loads(blob.replace("'", '"'))
            except json.JSONDecodeError:
                continue
        scores = data.get("score")
        if isinstance(scores, list) and len(scores) >= 1:
            try:
                scores = [float(s) for s in scores]
            except (TypeError, ValueError):
                continue
            return {"scores": scores, "reasoning": str(data.get("reasoning", ""))}
    return {"scores": None, "reasoning": raw.strip()[:300]}


# =====================================================================
# JUEZ GEMINI (Google Gen AI SDK)
# =====================================================================

def _media_type_for(image_path: str) -> str:
    ext = image_path.lower().rsplit(".", 1)[-1]
    return {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "gif": "image/gif", "webp": "image/webp",
    }.get(ext, "image/jpeg")


class GeminiJudge:
    """
    Juez Gemini vía el SDK google-genai. Requiere `pip install google-genai`.

    Soporta UNA O VARIAS API keys (p.ej. de cuentas distintas) para sortear la
    cuota DIARIA del free tier. Comportamiento ante un 429:
      - Si es límite POR MINUTO  -> espera el retryDelay y reintenta con la
        MISMA key (no tiene sentido rotar: las otras keys también esperarían).
      - Si es límite DIARIO       -> rota a la siguiente key. Si no quedan keys,
        propaga el error.

    Las keys se leen, por orden de preferencia, de:
      1) el parámetro `api_keys` (lista), o
      2) la variable de entorno GEMINI_API_KEYS (varias separadas por comas),
      3) la variable GEMINI_API_KEY / GOOGLE_API_KEY (una sola).

    Modelos juez: comprueba los nombres vigentes en
    https://ai.google.dev/gemini-api/docs/models
    """
    def __init__(self, model: str = DEFAULT_MODEL,
                 temperature: float = 0.0, max_tokens: int = 2048,
                 requests_per_minute: int = 5, max_retries: int = 6,
                 server_retry_max: int = 0,
                 api_keys: Optional[List[str]] = None):
        from google import genai
        self._genai = genai

        self.api_keys = api_keys if api_keys is not None else self._load_keys()
        if not self.api_keys:
            raise RuntimeError(
                "No se encontró ninguna API key. Define GEMINI_API_KEYS "
                "(varias separadas por comas) o GEMINI_API_KEY en el entorno/.env."
            )
        self._key_idx = 0
        self._exhausted_keys: set = set()   # índices de keys con cuota diaria agotada
        self._client = genai.Client(api_key=self.api_keys[self._key_idx])
        print(f"   🔑 {len(self.api_keys)} API key(s) cargada(s); "
              f"usando la #1.")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Control de ritmo: el free tier limita las peticiones por minuto
        # (p.ej. 5/min para gemini-3.5-flash). Espaciamos las llamadas para no
        # superarlo de forma preventiva, y además reintentamos ante un 429.
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self.max_retries = max_retries
        # Reintentos ante errores transitorios de servidor (503/500/502/504).
        # 0 = ilimitado (reintenta hasta que Google atienda la petición).
        self.server_retry_max = server_retry_max
        self._last_call_ts = 0.0

    # ---- gestión de keys ------------------------------------------------

    @staticmethod
    def _load_keys() -> List[str]:
        """Lee las keys del entorno. GEMINI_API_KEYS (lista separada por comas)
        tiene prioridad; si no, cae a GEMINI_API_KEY / GOOGLE_API_KEY."""
        import os
        multi = os.environ.get("GEMINI_API_KEYS", "")
        if multi.strip():
            return [k.strip() for k in multi.split(",") if k.strip()]
        single = (os.environ.get("GEMINI_API_KEY")
                  or os.environ.get("GOOGLE_API_KEY") or "").strip()
        return [single] if single else []

    def _rotate_key(self) -> bool:
        """Marca la key actual como agotada y pasa a la siguiente disponible.
        Devuelve True si quedaba otra key; False si se han agotado todas."""
        self._exhausted_keys.add(self._key_idx)
        for i in range(len(self.api_keys)):
            if i not in self._exhausted_keys:
                self._key_idx = i
                self._client = self._genai.Client(api_key=self.api_keys[i])
                # reiniciamos el reloj de ritmo: la key nueva tiene su propio
                # presupuesto por minuto.
                self._last_call_ts = 0.0
                print(f"   🔄 Cuota diaria agotada; rotando a la API key "
                      f"#{i + 1} de {len(self.api_keys)}.")
                return True
        return False

    # ---- control de ritmo y clasificación de errores --------------------

    def _throttle(self) -> None:
        """Espera lo necesario para no exceder requests_per_minute."""
        if self.min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_call_ts
        wait = self.min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.monotonic()

    @staticmethod
    def _retry_delay_from_error(err: Exception, attempt: int) -> float:
        """Extrae el retryDelay sugerido por la API; si no, backoff exponencial."""
        msg = str(err)
        m = re.search(r"retry(?:Delay)?['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s",
                      msg, flags=re.IGNORECASE)
        if m:
            # margen extra para no volver a topar justo en el borde
            return float(m.group(1)) + 1.0
        # backoff exponencial con jitter: 2,4,8,16,... s (+ 0-1s aleatorio)
        return min(2 ** attempt, 60) + random.random()

    @staticmethod
    def _is_rate_limit(err: Exception) -> bool:
        msg = str(err).lower()
        return "429" in msg or "resource_exhausted" in msg or "quota" in msg

    @staticmethod
    def _is_server_error(err: Exception) -> bool:
        """True si es un error transitorio de servidor de Google (no es culpa
        nuestra ni de la cuota): 503 UNAVAILABLE por alta demanda, o 500/502/504.
        Estos NO consumen cuota y conviene reintentarlos sin rotar de key."""
        msg = str(err).lower()
        return (
            "503" in msg or "unavailable" in msg
            or "500" in msg or "internal" in msg
            or "502" in msg or "504" in msg
            or "overloaded" in msg or "high demand" in msg
        )

    @staticmethod
    def _is_daily_limit(err: Exception) -> bool:
        """True si el 429 es por cuota DIARIA (no por minuto). Gemini lo indica
        en el quotaId/métrica del error: 'PerDay' vs 'PerMinute'. Si no se puede
        distinguir, se trata como NO diario (se espera, no se rota)."""
        msg = str(err).lower()
        if "perday" in msg or "per_day" in msg or "requestsperday" in msg:
            return True
        # algunos mensajes solo dicen 'daily'
        if "daily" in msg and "limit" in msg:
            return True
        return False

    # ---- llamada principal ---------------------------------------------

    def generate(self, prompt: str, image_path: str) -> str:
        from google.genai import types
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        last_err: Optional[Exception] = None
        attempt = 0            # cuenta solo reintentos por límite POR MINUTO
        server_attempt = 0     # cuenta solo reintentos por error de servidor (503)
        while attempt <= self.max_retries:
            self._throttle()
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=[
                        types.Part.from_bytes(
                            data=img_bytes,
                            mime_type=_media_type_for(image_path),
                        ),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                    ),
                )
                # response.text concatena el texto de la respuesta. Puede ser
                # None si el modelo no devolvió texto (p.ej. bloqueo de
                # seguridad); se protege.
                return response.text or ""
            except Exception as e:
                last_err = e

                # 503/500/502/504: error transitorio de Google (alta demanda).
                # No consume cuota; reintentamos con la MISMA key, sin gastar el
                # presupuesto del límite por minuto. server_retry_max=0 => infinito.
                if self._is_server_error(e):
                    server_attempt += 1
                    if self.server_retry_max and server_attempt > self.server_retry_max:
                        print("   ⛔ Demasiados errores de servidor; abandono.")
                        raise
                    # backoff creciente con techo de 60s + jitter
                    delay = min(2 ** min(server_attempt, 6), 60) + random.random()
                    print(f"   🌐 Servidor saturado (503/alta demanda); esperando "
                          f"{delay:.1f}s y reintentando (intento {server_attempt}"
                          f"{'' if not self.server_retry_max else '/' + str(self.server_retry_max)})...")
                    time.sleep(delay)
                    continue

                if not self._is_rate_limit(e):
                    raise

                # 429 por cuota DIARIA -> intentar rotar de key.
                if self._is_daily_limit(e):
                    if self._rotate_key():
                        # con la key nueva no consumimos un reintento: el
                        # presupuesto por minuto se reinicia.
                        continue
                    print("   ⛔ Todas las API keys han agotado su cuota diaria.")
                    raise

                # 429 por cuota POR MINUTO -> esperar y reintentar misma key.
                if attempt < self.max_retries:
                    delay = self._retry_delay_from_error(e, attempt)
                    print(f"   ⏳ Límite por minuto alcanzado; esperando "
                          f"{delay:.1f}s (reintento {attempt + 1}/"
                          f"{self.max_retries})...")
                    time.sleep(delay)
                    attempt += 1
                    continue
                raise
        # Agotados los reintentos: propaga el último error.
        raise last_err  # type: ignore[misc]


# =====================================================================
# RESULTADO Y EVALUADOR PRINCIPAL
# =====================================================================

@dataclass
class VIEResult:
    sc: Optional[float]            # min de sub-scores SC (0-10)
    pq: Optional[float]            # min de sub-scores PQ (0-10)
    overall: Optional[float]      # sqrt(SC*PQ) (0-10)
    sc_subscores: List[float] = field(default_factory=list)
    pq_subscores: List[float] = field(default_factory=list)
    sc_reasoning: str = ""
    pq_reasoning: str = ""

    def normalized(self) -> Dict[str, Optional[float]]:
        """Escala 0-1 (como reporta AskVR)."""
        return {
            "sc_01": round(self.sc / 10, 4) if self.sc is not None else None,
            "pq_01": round(self.pq / 10, 4) if self.pq is not None else None,
            "overall_01": round(self.overall / 10, 4) if self.overall is not None else None,
        }


class VIEScore:
    """
    Evaluador VIEScore para descripciones de texto, con Gemini como juez.

    Uso:
        judge = VIEScore(model="gemini-3.5-flash")
        res = judge.evaluate(
            image_path="scene.jpg",
            query="¿Qué hay a mi derecha?",
            response="A tu derecha, muy cerca, hay un cofre del tesoro.",
            context="[{...objetos...}]",
            intent="localizacion_objeto",
        )
        print(res.sc, res.pq, res.overall)
    """

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.0,
                 requests_per_minute: int = 5, max_tokens: int = 2048,
                 server_retry_max: int = 0,
                 api_keys: Optional[List[str]] = None):
        self.judge = GeminiJudge(model=model, temperature=temperature,
                                 requests_per_minute=requests_per_minute,
                                 max_tokens=max_tokens,
                                 server_retry_max=server_retry_max,
                                 api_keys=api_keys)

    def _sc_aspects(self, intent: Optional[str]) -> List[tuple]:
        # spatial_correctness ya está en SC_ASPECTS_BASE y aplica a TODAS las
        # categorías (el propio aspecto no penaliza si la respuesta no menciona
        # ninguna dirección). El parámetro intent se conserva por compatibilidad
        # con quien llama, pero ya no altera los aspectos.
        return list(SC_ASPECTS_BASE)

    def evaluate(
        self,
        image_path: str,
        query: str,
        response: str,
        context: str = "",
        intent: Optional[str] = None,
    ) -> VIEResult:
        # --- SC ---
        sc_aspects = self._sc_aspects(intent)
        sc_prompt = build_sc_prompt(query, response, context, sc_aspects)
        sc_raw = self.judge.generate(sc_prompt, image_path)
        sc_parsed = parse_score_json(sc_raw, len(sc_aspects))

        # --- PQ ---
        pq_prompt = build_pq_prompt(query, response, PQ_ASPECTS)
        pq_raw = self.judge.generate(pq_prompt, image_path)
        pq_parsed = parse_score_json(pq_raw, len(PQ_ASPECTS))

        sc_subs = sc_parsed["scores"]
        pq_subs = pq_parsed["scores"]

        # VIEScore: cada eje = MÍNIMO de sus sub-scores
        sc = float(min(sc_subs)) if sc_subs else None
        pq = float(min(pq_subs)) if pq_subs else None

        # Overall = media geométrica. Se calcula POR ÍTEM (clave: luego se
        # promedia O entre ítems, NO se recompone desde SC y PQ medias).
        if sc is not None and pq is not None:
            overall = math.sqrt(sc * pq)
        else:
            overall = None

        return VIEResult(
            sc=sc, pq=pq, overall=overall,
            sc_subscores=sc_subs or [],
            pq_subscores=pq_subs or [],
            sc_reasoning=sc_parsed["reasoning"],
            pq_reasoning=pq_parsed["reasoning"],
        )