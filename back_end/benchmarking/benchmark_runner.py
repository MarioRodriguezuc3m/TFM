"""
Runner para el Benchmark 1 — Efecto del contexto.

Ejecuta cada consulta del dataset bajo los 4 niveles de contexto
(C1, C2, C3, C4) manteniendo TODO lo demás fijo (mismo clasificador BETO,
mismas plantillas especializadas) y guarda los resultados en un CSV listo
para analizar.

Ahora soporta VARIOS modelos de visión a la vez: se ejecuta el barrido
completo (consultas × niveles) para cada modelo y se guarda UN CSV por
modelo. Por defecto corre los tres modelos locales del proyecto:
    - qwen2.5vl:latest
    - llava:7b
    - blaifa/InternVL3_5:8B

Uso:
    python -m benchmarking.benchmark_runner --dataset benchmarking/dataset/dataset.json --out resultados/

    # Subconjunto de modelos:
    python -m benchmarking.benchmark_runner --dataset ... --modelos llava:7b qwen2.5vl:latest

Formato del dataset (JSON, lista de objetos):
[
  {
    "id": "q001",
    "query": "¿Qué hay a mi derecha?",
    "expected_intent": "localizacion_objeto",
    "image_path": "scene_01.jpg",
    "objetos_visibles": [
      {
        "label": "Treasure Chest",
        "description": "An old wooden chest with brass fittings.",
        "relative_position": {"x": 3.2, "y": 0.0, "z": -0.5}
      },
      ...
    ]
  },
  ...
]

Nota sobre las imágenes: `image_path` puede ser solo el nombre del fichero
(p.ej. "scene_01.jpg"); se resuelve contra la carpeta central de imágenes
definida en paths.BENCHMARK_IMAGES_DIR. También se aceptan rutas absolutas o
rutas relativas ya existentes (compatibilidad hacia atrás).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any

from core.query_processor import QueryProcessor, CONTEXT_LEVELS
from paths import BENCHMARK_IMAGES_DIR


# Orden fijo para garantizar que las columnas del CSV son siempre las mismas
LEVELS_ORDER = ["C1", "C2", "C3", "C4"]

# Modelos de visión a evaluar (uno por CSV de salida).
DEFAULT_MODELS = [
    "qwen2.5vl:latest",
    "llava:7b",
    "blaifa/InternVL3_5:8B",
]


def model_tag(modelo: str) -> str:
    """Convierte el nombre de un modelo en un sufijo seguro para fichero.

    Ej.: "blaifa/InternVL3_5:8B" -> "InternVL3_5_8B"
         "qwen2.5vl:latest"      -> "qwen2.5vl_latest"
         "llava:7b"              -> "llava_7b"
    """
    tag = modelo.split("/")[-1]          # quita prefijo de repo (blaifa/...)
    tag = tag.replace(":", "_")          # separador de tag de Ollama
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)  # cualquier otro carácter raro
    return tag


def dataset_tag(dataset_path: Path) -> str:
    """Convierte el nombre del dataset en un sufijo seguro para fichero.

    Permite distinguir los CSV de salida cuando se corre el benchmark sobre
    varios datasets (dataset_1, dataset_2, ...), que es justo lo que se hace
    para repartir las consultas y esquivar el límite diario de la API del juez.

    Ej.: "dataset_1.json" -> "dataset_1"
         "dataset.json"   -> "dataset"
    """
    return re.sub(r"[^A-Za-z0-9_.-]", "_", dataset_path.stem)


def resolve_image(image_path: str) -> str:
    """Resuelve la ruta de imagen contra la carpeta central de benchmarking.

    - Si viene vacía → "".
    - Si es absoluta o ya existe tal cual → se respeta.
    - En otro caso (solo nombre de fichero o ruta antigua) → se toma el nombre
      de fichero y se ancla en BENCHMARK_IMAGES_DIR.
    """
    if not image_path:
        return ""
    p = Path(image_path)
    if p.is_absolute() or p.exists():
        return str(p)
    return str(BENCHMARK_IMAGES_DIR / p.name)


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("El dataset debe ser una lista JSON de consultas.")
    return data


def run_one(
    qp: QueryProcessor,
    item: Dict[str, Any],
    level: str,
    modelo: str,
) -> Dict[str, Any]:
    """Ejecuta una consulta bajo un nivel de contexto y devuelve la fila CSV."""
    img = resolve_image(item.get("image_path", ""))
    start = time.perf_counter()
    try:
        result = qp.process(
            texto_usuario=item["query"],
            ruta_imagen=img,
            objetos_visibles=item.get("objetos_visibles", []),
            context_level=level,
            temperature=0.0,
            seed=42,
        )
        error = ""
    except Exception as e:
        result = {
            "descripcion": "",
            "intencion": "ERROR",
            "confianza": 0.0,
            "ood": False,
            "context_level": level,
            "prompt_template": None,
        }
        error = str(e)
    elapsed = time.perf_counter() - start

    return {
        "model":                  modelo,  # modelo de visión usado en esta fila
        "query_id":               item["id"],
        "query":                  item["query"],
        "image_path":             img,  # ruta ya resuelta (la usa el juez VIEScore)
        "expected_intent":        item.get("expected_intent", ""),
        "predicted_intent":       result["intencion"],
        "intent_correct":         int(result["intencion"] == item.get("expected_intent", "")),
        "intent_confidence":      round(result.get("confianza", 0.0), 4),
        "context_level":          level,
        "prompt_template":        result.get("prompt_template") or "",
        "response_es":            result["descripcion"],
        "latency_s":              round(elapsed, 3),
        "ood":                    int(bool(result.get("ood", False))),
        "error":                  error,
    }


def run_one_model(
    dataset: List[Dict[str, Any]],
    out_dir: Path,
    modelo_vision: str,
    levels: List[str],
    ds_tag: str,
) -> Path:
    """Ejecuta el barrido completo (consultas × niveles) para UN modelo
    y guarda su propio CSV. Devuelve la ruta del CSV generado.

    El nombre del CSV crudo incluye el dataset y el modelo
    (benchmark1_raw_<dataset>_<modelo>.csv) para no pisar resultados al
    correr el benchmark sobre varios datasets."""
    raw_csv = out_dir / f"benchmark1_raw_{ds_tag}_{model_tag(modelo_vision)}.csv"

    print(f"\n🚀 Cargando QueryProcessor (modelo={modelo_vision})...")
    qp = QueryProcessor(modelo_vision=modelo_vision)

    n_total = len(dataset) * len(levels)
    print(f"\n📊 Benchmark 1 [{modelo_vision}] — {len(dataset)} consultas × "
          f"{len(levels)} niveles = {n_total} ejecuciones\n")

    rows: List[Dict[str, Any]] = []
    counter = 0
    for item in dataset:
        for level in levels:
            counter += 1
            print(f"\n[{modelo_vision}] [{counter}/{n_total}] "
                  f"id={item['id']} level={level}")
            print(f"   query: {item['query']!r}")
            row = run_one(qp, item, level, modelo_vision)
            rows.append(row)
            # Guardado incremental para no perder datos si se cuelga algo
            _write_csv(raw_csv, rows)

    print(f"\n✅ [{modelo_vision}] Resultados guardados en {raw_csv}")
    return raw_csv


def run_benchmark(
    dataset_path: Path,
    out_dir: Path,
    modelos_vision: List[str] = None,
    levels: List[str] = None,
) -> List[Path]:
    """Ejecuta el benchmark para cada modelo de la lista y devuelve la lista
    de CSVs generados (uno por modelo)."""
    modelos_vision = modelos_vision or DEFAULT_MODELS
    levels = levels or LEVELS_ORDER
    for lv in levels:
        if lv not in CONTEXT_LEVELS:
            raise ValueError(f"Nivel inválido: {lv}")

    dataset = load_dataset(dataset_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_tag = dataset_tag(dataset_path)

    print(f"🧪 Modelos a evaluar ({len(modelos_vision)}): "
          f"{', '.join(modelos_vision)}")
    print(f"📁 Dataset: {dataset_path.name} (tag='{ds_tag}')")

    csv_paths: List[Path] = []
    for modelo in modelos_vision:
        csv_path = run_one_model(dataset, out_dir, modelo, levels, ds_tag)
        csv_paths.append(csv_path)

    print("\n🏁 Todos los modelos completados. CSVs generados:")
    for p in csv_paths:
        print(f"   - {p}")
    return csv_paths


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Benchmark 1 — Efecto del contexto.")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Ruta al JSON con las consultas y ground truth.")
    parser.add_argument("--out", type=Path, default=Path("resultados"),
                        help="Directorio donde guardar los CSV de resultados.")
    parser.add_argument("--modelos", nargs="+", default=DEFAULT_MODELS,
                        help="Modelos de visión (Ollama) a evaluar; un CSV por "
                             "modelo. Por defecto: los tres modelos locales.")
    parser.add_argument("--levels", nargs="+", default=LEVELS_ORDER,
                        help="Subconjunto de niveles a ejecutar (e.g. --levels C2 C4).")
    args = parser.parse_args()

    run_benchmark(args.dataset, args.out, args.modelos, args.levels)


if __name__ == "__main__":
    main()