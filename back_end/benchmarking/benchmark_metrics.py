"""
Métricas del Benchmark 1 — únicamente VIEScore (estilo AskVR).

Lee el CSV crudo producido por benchmark_runner.py, vuelve a unir cada fila
con su imagen y su contexto (desde el dataset JSON), y puntúa la RESPUESTA del
sistema con VIEScore (SC, PQ, O). Produce:

  - benchmark1_viescore.csv  : una fila por (consulta × nivel) con SC, PQ, O
                               (en 0-10 y normalizado 0-1) + rationale del juez.
  - benchmark1_viescore_summary.csv : agregado por nivel × intención con
                               SC, PQ y O medios. O se promedia POR ÍTEM
                               (igual que AskVR), no se recompone de medias.

Uso:
    python -m benchmarking.benchmark_metrics resultados/benchmark1_raw.csv \
        --dataset benchmarking/dataset/dataset.json \
        --model gemini-3.5-flash \
        --out resultados/

Requisitos: `pip install google-genai python-dotenv`. Define en el entorno o en
un .env en la raíz del proyecto (back_end/) UNA de estas opciones:
  - GEMINI_API_KEYS = "key1,key2,key3"  (varias cuentas → el juez rota cuando
    una agota su cuota diaria, permitiendo evaluar más ítems por día)
  - GEMINI_API_KEY  = "key"             (una sola)

Nota: el juez (Gemini) NO es el mismo modelo que genera las respuestas (Qwen),
con lo que se evita el sesgo de auto-evaluación.

Nota sobre las imágenes: el `image_path` (del CSV crudo o del dataset) se
resuelve contra paths.BENCHMARK_IMAGES_DIR si no es absoluto ni existe tal cual,
igual que en benchmark_runner.py. El juez VIEScore necesita leer la imagen del
disco para codificarla en base64.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from benchmarking.vie_score import VIEScore
from paths import BENCHMARK_IMAGES_DIR


# =====================================================================
# RESOLUCIÓN DE RUTA DE IMAGEN
# =====================================================================

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


# =====================================================================
# CARGA Y UNIÓN CON EL DATASET
# =====================================================================

def index_dataset(dataset_path: Path) -> Dict[str, Dict[str, Any]]:
    """Indexa el dataset por id para recuperar imagen y contexto por consulta."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


def context_string(item: Dict[str, Any]) -> str:
    """Contexto estructurado (objetos crudos) que se pasa al juez como apoyo."""
    objs = item.get("objetos_visibles", [])
    return json.dumps(objs, indent=2, ensure_ascii=False)


# =====================================================================
# EVALUACIÓN VIEScore FILA A FILA
# =====================================================================

def evaluate_rows(
    raw_csv: Path,
    dataset_idx: Dict[str, Dict[str, Any]],
    judge: VIEScore,
    response_field: str = "response_es",
) -> List[Dict[str, Any]]:
    with open(raw_csv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows: List[Dict[str, Any]] = []
    n = len(rows)
    for i, row in enumerate(rows, 1):
        qid = row["query_id"]
        item = dataset_idx.get(qid, {})
        image_path = resolve_image(
            row.get("image_path") or item.get("image_path", "")
        )
        query = row.get("query", "")
        response = row.get(response_field, "") or ""
        intent = row.get("predicted_intent", "") or row.get("expected_intent", "")
        context = context_string(item) if item else ""

        print(f"[{i}/{n}] VIEScore id={qid} level={row.get('context_level')} ...")

        # OOD o respuesta vacía: no se evalúa (no aplica VIEScore)
        if not response or row.get("ood") == "1":
            res = None
            err = "ood_or_empty"
        else:
            try:
                res = judge.evaluate(
                    image_path=image_path,
                    query=query,
                    response=response,
                    context=context,
                    intent=intent,
                )
                err = ""
            except Exception as e:
                res = None
                err = _short_error(e)

        out = dict(row)  # conserva todo lo del CSV crudo
        if res is not None:
            norm = res.normalized()
            out.update({
                "vie_sc":          _r(res.sc),
                "vie_pq":          _r(res.pq),
                "vie_overall":     _r(res.overall),
                "vie_sc_01":       norm["sc_01"],
                "vie_pq_01":       norm["pq_01"],
                "vie_overall_01":  norm["overall_01"],
                "vie_sc_subscores": "|".join(str(s) for s in res.sc_subscores),
                "vie_pq_subscores": "|".join(str(s) for s in res.pq_subscores),
                "vie_sc_reasoning": res.sc_reasoning,
                "vie_pq_reasoning": res.pq_reasoning,
                "vie_error":        "",
            })
        else:
            out.update({
                "vie_sc": "", "vie_pq": "", "vie_overall": "",
                "vie_sc_01": "", "vie_pq_01": "", "vie_overall_01": "",
                "vie_sc_subscores": "", "vie_pq_subscores": "",
                "vie_sc_reasoning": "", "vie_pq_reasoning": "",
                "vie_error": err,
            })
        out_rows.append(out)

    return out_rows


def _r(v: Optional[float]) -> Any:
    return round(v, 3) if v is not None else ""


def _short_error(e: Exception) -> str:
    """Resume errores largos (p.ej. el volcado JSON de un 429) a una etiqueta
    corta y legible, para que no ensucien el CSV."""
    msg = str(e)
    low = msg.lower()
    if "429" in msg or "resource_exhausted" in low or "quota" in low:
        return "quota_exceeded_429"
    # Cualquier otro error: primera línea, recortada.
    first = msg.strip().splitlines()[0] if msg.strip() else "unknown_error"
    return first[:200]


# =====================================================================
# AGREGACIÓN POR NIVEL × INTENCIÓN
# =====================================================================

def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def grab(rs, key):
        return [float(r[key]) for r in rs if r.get(key) not in ("", None)]

    def mean(vals):
        return round(sum(vals) / len(vals), 3) if vals else ""

    buckets: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["context_level"], r.get("expected_intent", ""))].append(r)

    summary: List[Dict[str, Any]] = []
    for (level, intent), rs in sorted(buckets.items()):
        summary.append(_summary_row(level, intent, rs, grab, mean))

    # Fila __ALL__ por nivel
    by_level: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_level[r["context_level"]].append(r)
    for level, rs in sorted(by_level.items()):
        summary.append(_summary_row(level, "__ALL__", rs, grab, mean))

    return summary


def _summary_row(level, intent, rs, grab, mean):
    sc = grab(rs, "vie_sc")
    pq = grab(rs, "vie_pq")
    ov = grab(rs, "vie_overall")        # O promediada por ítem (clave)
    return {
        "context_level":   level,
        "expected_intent": intent,
        "n":               len(rs),
        "n_scored":        len(ov),
        "vie_sc_mean":     mean(sc),
        "vie_pq_mean":     mean(pq),
        "vie_overall_mean": mean(ov),
        "vie_sc_mean_01":   round(mean(sc) / 10, 4) if sc else "",
        "vie_pq_mean_01":   round(mean(pq) / 10, 4) if pq else "",
        "vie_overall_mean_01": round(mean(ov) / 10, 4) if ov else "",
    }


# =====================================================================
# I/O
# =====================================================================

# Columnas relevantes del CSV de detalle (en este orden). El resto del CSV
# crudo (latencias, plantilla, confianza, response_en/es completos, etc.) se
# conserva aparte en el CSV "_full". Aquí queda lo necesario para leer y
# analizar VIEScore de un vistazo.
RELEVANT_COLUMNS = [
    "query_id",
    "context_level",
    "expected_intent",
    "predicted_intent",
    "query",
    "response_es",
    "vie_sc", "vie_pq", "vie_overall",
    "vie_sc_01", "vie_pq_01", "vie_overall_01",
    "vie_sc_subscores", "vie_pq_subscores",
    "vie_sc_reasoning", "vie_pq_reasoning",
    "vie_error",
]


def select_columns(rows: List[Dict[str, Any]],
                   columns: List[str]) -> List[Dict[str, Any]]:
    """Devuelve cada fila con SOLO las columnas indicadas (las ausentes se
    rellenan vacías)."""
    return [{c: r.get(c, "") for c in columns} for r in rows]


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    # Unión de todas las claves preservando orden de la primera fila
    fieldnames = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =====================================================================
# CLI
# =====================================================================

def main():
    # Carga GEMINI_API_KEY desde back_end/.env (o desde el entorno si ya está
    # definida). Anclado a la raíz del proyecto para que funcione sin importar
    # desde qué carpeta se ejecute.
    from dotenv import load_dotenv
    from paths import ROOT
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Métricas del Benchmark 1 con VIEScore (estilo AskVR).")
    parser.add_argument("raw_csv", type=Path,
                        help="CSV crudo de benchmark_runner.py")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Dataset JSON (para recuperar imagen y contexto).")
    parser.add_argument("--model", default="gemini-3.5-flash",
                        help="Modelo Gemini juez (p.ej. gemini-3.5-flash, "
                             "gemini-3.1-flash-lite).")
    parser.add_argument("--response-field", default="response_es",
                        choices=["response_es", "response_en"],
                        help="Qué respuesta evaluar: la final en español (default) "
                             "o la inglesa pre-traducción.")
    parser.add_argument("--rpm", type=int, default=5,
                        help="Peticiones por minuto permitidas por el juez "
                             "(free tier gemini-3.5-flash = 5; cada ítem usa 2 "
                             "peticiones). El runner espaciará las llamadas para "
                             "no superarlo.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out or args.raw_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deriva el tag del modelo del nombre del crudo: benchmark1_raw_<tag>.csv -> <tag>
    stem = args.raw_csv.stem                      # p.ej. "benchmark1_raw_llava_7b"
    tag = stem.replace("benchmark1_raw_", "", 1) if stem.startswith("benchmark1_raw_") else stem

    detail_csv      = out_dir / f"benchmark1_viescore_{tag}.csv"
    detail_full_csv = out_dir / f"benchmark1_viescore_full_{tag}.csv"
    summary_csv     = out_dir / f"benchmark1_viescore_summary_{tag}.csv"

    print(f"⚖️  Cargando juez VIEScore (Gemini, model={args.model}, "
          f"rpm={args.rpm})")
    judge = VIEScore(model=args.model, requests_per_minute=args.rpm)

    print(f"📥 Dataset: {args.dataset}")
    dataset_idx = index_dataset(args.dataset)

    rows = evaluate_rows(args.raw_csv, dataset_idx, judge, args.response_field)

    # CSV completo (todo lo del crudo + métricas) por si hace falta auditar.
    write_csv(rows, detail_full_csv)
    # CSV de detalle reducido a columnas relevantes (el que se lee a diario).
    write_csv(select_columns(rows, RELEVANT_COLUMNS), detail_csv)
    print(f"   → detalle (columnas clave) en {detail_csv}")
    print(f"   → detalle completo en {detail_full_csv}")

    summary = summarize(rows)
    write_csv(summary, summary_csv)
    print(f"📊 Resumen (nivel × intención) en {summary_csv}\n")

    print("--- VIEScore por nivel (0-1, como AskVR) ---")
    for r in summary:
        if r["expected_intent"] == "__ALL__":
            print(f"   {r['context_level']}  ALL   "
                  f"SC={r['vie_sc_mean_01']}  "
                  f"PQ={r['vie_pq_mean_01']}  "
                  f"O={r['vie_overall_mean_01']}  "
                  f"(n_scored={r['n_scored']})")


if __name__ == "__main__":
    main()