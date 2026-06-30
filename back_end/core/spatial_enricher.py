"""
Preprocesamiento espacial determinista (IA simbólica).

Este módulo convierte coordenadas locales (x, y, z) en campos semánticos de
alto nivel (dirección cardinal, cubo de distancia, posición vertical, frase
natural lista para usar). La idea es descargar al MLLM de una tarea en la
que suele fallar — razonar sobre números — y entregarle las relaciones
espaciales ya resueltas.

Convención de coordenadas (THREE.js, sistema local de la cámara):
    -Z = delante    +Z = detrás
    -X = izquierda  +X = derecha
    -Y = abajo      +Y = arriba

Todas las funciones son puras (sin estado ni I/O) y testeables de forma
aislada, lo cual simplifica el capítulo de evaluación del TFM.
"""

from __future__ import annotations

import math
from typing import List, Dict, Any


# =====================================================================
# UMBRALES (único punto de recalibración de la semántica espacial)
# =====================================================================

# Distancia horizontal en metros (sqrt(x^2 + z^2))
DIST_ARM_REACH = 2.0    # dentro del alcance del brazo
DIST_VERY_CLOSE = 5.0   # muy cerca
DIST_CLOSE = 10.0       # a unos pasos
DIST_MEDIUM = 20.0      # distancia media
# > 20 m se clasifica como "far"

# Umbrales verticales en metros
Y_ABOVE = 1.5           # por encima de la cabeza
Y_BELOW = -0.8          # en el suelo

# Semiángulo (en grados) del sector "front". Cuanto mayor, más ancho es "delante"
# y más lateral hay que estar para que algo cuente como front-left / front-right.
# Con 35°, un objeto a ~25° del centro (p. ej. el cañón) se considera "front".
FRONT_HALF_DEG = 35.0

# Ángulo a partir del cual algo deja de ser diagonal (front-right/front-left) y
# pasa a ser lateral puro (right/left). Más bajo => banda diagonal más estrecha y
# "right"/"left" más amplios. Con 55°, un objeto a ~58° (p. ej. el barco) es "right".
DIAG_SIDE_DEG = 55.0


# =====================================================================
# MAPEOS DE DIRECCIÓN / DISTANCIA / VERTICAL → FRASE NATURAL
# =====================================================================

_DIRECTION_PHRASE = {
    "front":        "directly in front of you",
    "front-right":  "ahead and to your right",
    "right":        "directly to your right",
    "back-right":   "behind you to your right",
    "behind":       "directly behind you",
    "back-left":    "behind you to your left",
    "left":         "directly to your left",
    "front-left":   "ahead and to your left",
}

_DISTANCE_PHRASE = {
    "within_arm_reach": "within arm's reach",
    "very_close":       "very close",
    "close":            "a few steps away",
    "medium":           "at medium distance",
    "far":              "far away",
}

_VERTICAL_SUFFIX = {
    "above_eye_level": " (above your eye level)",
    "eye_level":       "",
    "below_eye_level": " (on the floor level)",
}

_COMPASS_MAP = {
    "front":       "N",
    "front-right": "NE",
    "right":       "E",
    "back-right":  "SE",
    "behind":      "S",
    "back-left":   "SW",
    "left":        "W",
    "front-left":  "NW",
}

_FRONT_DIRECTIONS = {"front", "front-left", "front-right"}


# =====================================================================
# FUNCIONES DE CLASIFICACIÓN (puras)
# =====================================================================

def compute_direction(x: float, z: float) -> str:
    """
    Clasifica la posición (x, z) en uno de los 8 sectores angulares.

    Se usa atan2(x, -z) porque -Z es 'delante' en coordenadas locales de la
    cámara. El ángulo queda en [-180, 180], con 0° = delante, 90° = derecha,
    ±180° = detrás, -90° = izquierda.
    """
    angle = math.degrees(math.atan2(x, -z))

    # El sector frontal es ±FRONT_HALF_DEG (más ancho que los 45° uniformes) y la
    # banda diagonal front-right/front-left va de FRONT_HALF_DEG a DIAG_SIDE_DEG;
    # más allá ya es lateral puro (right/left). Así los diagonales solo aplican en
    # una franja estrecha y "delante" / "a un lado" son más amplios.
    F = FRONT_HALF_DEG
    D = DIAG_SIDE_DEG

    if -F <= angle < F:
        return "front"
    if F <= angle < D:
        return "front-right"
    if D <= angle < 112.5:
        return "right"
    if 112.5 <= angle < 157.5:
        return "back-right"
    if angle >= 157.5 or angle < -157.5:
        return "behind"
    if -157.5 <= angle < -112.5:
        return "back-left"
    if -112.5 <= angle < -D:
        return "left"
    return "front-left"  # -D <= angle < -F


def compute_distance_bucket(horizontal_dist: float) -> str:
    """Clasifica la distancia horizontal (en el plano XZ) en categorías discretas."""
    if horizontal_dist < DIST_ARM_REACH:
        return "within_arm_reach"
    if horizontal_dist < DIST_VERY_CLOSE:
        return "very_close"
    if horizontal_dist < DIST_CLOSE:
        return "close"
    if horizontal_dist < DIST_MEDIUM:
        return "medium"
    return "far"


def compute_vertical_position(y: float) -> str:
    """Clasifica la altura relativa del objeto respecto a la cabeza del usuario."""
    if y > Y_ABOVE:
        return "above_eye_level"
    if y < Y_BELOW:
        return "below_eye_level"
    return "eye_level"


def build_position_description(direction: str, dist_bucket: str, vert_pos: str) -> str:
    """Construye una frase natural en inglés lista para que el MLLM la copie tal cual."""
    return (
        f"{_DISTANCE_PHRASE[dist_bucket]}, "
        f"{_DIRECTION_PHRASE[direction]}"
        f"{_VERTICAL_SUFFIX[vert_pos]}"
    )


# =====================================================================
# PIPELINE COMPLETO: cruda -> enriquecida
# =====================================================================

def enrich_objects(objetos_visibles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convierte la lista cruda de objetos (con `relative_position` en x, y, z)
    en una lista enriquecida con campos semánticos ya calculados.

    La salida va ORDENADA por distancia horizontal ascendente (lo más cercano
    primero), lo cual ayuda al MLLM a priorizar de forma natural sin tener
    que reordenar él mismo.

    Estructura de cada objeto enriquecido:
        {
            "label":                 "Treasure Chest",
            "description":           "An old wooden chest...",
            "position_description":  "very close, ahead and to your right",
            "direction":             "front-right",
            "distance_bucket":       "very_close",
            "angular_sector":        "NE",
            "vertical_position":     "eye_level",
            "is_in_front":           true,
            "distance_m":            3.21,   # distancia radial √(x²+z²)
            "lateral_distance_m":    3.20,   # |x|: cuánto a un lado
            "depth_distance_m":      0.50,   # |z|: cuánto adelante/atrás
            "contained_objects":     [ ... ]    # opcional, si venía en la entrada
        }
    """
    enriched: List[Dict[str, Any]] = []

    for obj in objetos_visibles:
        # Edge case: objeto sin posición -> lo pasamos tal cual
        if "relative_position" not in obj:
            enriched.append(dict(obj))
            continue

        pos = obj["relative_position"]
        x = float(pos.get("x", 0.0))
        y = float(pos.get("y", 0.0))
        z = float(pos.get("z", 0.0))

        horizontal_dist = math.sqrt(x * x + z * z)
        direction = compute_direction(x, z)
        dist_bucket = compute_distance_bucket(horizontal_dist)
        vert_pos = compute_vertical_position(y)

        enriched_obj: Dict[str, Any] = {
            "label": obj.get("label", "unknown"),
            "description": obj.get("description", ""),
            "position_description": build_position_description(
                direction, dist_bucket, vert_pos
            ),
            "direction": direction,
            "distance_bucket": dist_bucket,
            "angular_sector": _COMPASS_MAP[direction],
            "vertical_position": vert_pos,
            "is_in_front": direction in _FRONT_DIRECTIONS,
            "distance_m": round(horizontal_dist, 2),
            "lateral_distance_m": round(abs(x), 2),
            "depth_distance_m": round(abs(z), 2),
        }

        # Conservamos los sub-objetos (heredan la posición del padre)
        sub = obj.get("contained_objects")
        if sub:
            enriched_obj["contained_objects"] = [
                {"label": s.get("label"), "description": s.get("description", "")}
                for s in sub
            ]

        enriched.append(enriched_obj)

    # Ordenar por distancia horizontal (más cerca primero)
    enriched.sort(key=lambda o: o.get("distance_m", float("inf")))
    return enriched


def enrich_nearest_object(nearest: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """
    Enriquece el OBJETO más cercano que YA ha elegido el front-end (es quien tiene
    las posiciones de los sub-objetos dentro de la escena 3D). El front envía un
    dict con `label`, `description`, `relative_position` y, si la pieza pertenece a
    un grupo, `group`. Aquí solo se le añaden los campos espaciales (frase de
    posición, dirección, distancia). Devuelve None si no llega ninguno.
    """
    if not nearest or "relative_position" not in nearest:
        return None

    p = nearest["relative_position"]
    x = float(p.get("x", 0.0))
    y = float(p.get("y", 0.0))
    z = float(p.get("z", 0.0))
    d = math.sqrt(x * x + z * z)
    direction = compute_direction(x, z)

    out: Dict[str, Any] = {
        "label": nearest.get("label", "unknown"),
        "description": nearest.get("description", ""),
        "position_description": build_position_description(
            direction, compute_distance_bucket(d), compute_vertical_position(y)
        ),
        "direction": direction,
        "distance_bucket": compute_distance_bucket(d),
        "distance_m": round(d, 2),
    }
    if nearest.get("group"):
        out["group"] = nearest["group"]
    return out