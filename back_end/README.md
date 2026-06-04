# back_end — estructura refactorizada

Tres paquetes con responsabilidades separadas, más carpetas de datos/artefactos
en la raíz. **Todo se ejecuta desde esta carpeta raíz** (`back_end/`) usando
`python -m paquete.modulo`, para que los imports entre paquetes funcionen.

```
back_end/
├── paths.py                 # rutas centralizadas (única fuente de verdad)
├── requirements.txt
│
├── core/                    # backend principal (funcionalidad en producción)
│   ├── backend.py           # servidor FastAPI  → punto de entrada
│   ├── query_processor.py   # orquestador del pipeline
│   ├── spatial_enricher.py  # preprocesamiento espacial
│   └── prompts/             # plantillas .txt  (PEGA AQUÍ TUS .txt REALES)
│
├── finetuning/              # clasificador de intenciones BETO
│   ├── fine_tuning_modelo.py    # entrena y guarda el modelo final
│   ├── comparacion_finetuning.py# comparativa de encoders (gráficos memoria)
│   ├── test_finetuning.py       # smoke test del clasificador
│   ├── validacion_dataset.py    # herramienta de anotación (CLI con --csv)
│   └── dataset_consultas.csv
│
├── benchmarking/            # los benchmarks + VIEScore (juez Claude)
│   ├── benchmark_runner.py  # genera respuestas (Qwen) por nivel C1–C4
│   ├── benchmark_metrics.py # puntúa con VIEScore
│   ├── vie_score.py         # juez Claude (Anthropic API)
│   └── dataset/             # consultas e imágenes de evaluación
│
├── modelo_vr_guardado/      # pesos del clasificador (salida de finetuning)
├── checkpoints/             # checkpoints de entrenamiento
├── resultados/
├── resultados_comparativa/
└── current_input/           # I/O temporal del servidor
```

## Rutas: paths.py
Ningún script usa ya rutas relativas frágiles tipo `"./modelo_vr_guardado"`.
Todas las carpetas se resuelven en `paths.py` respecto a la raíz del proyecto,
así da igual desde dónde mires: el import `from paths import MODELO_DIR` siempre
apunta al sitio correcto.

## Cómo ejecutar cada parte (siempre desde la raíz back_end/)

```bash
# 1) Entrenar el clasificador (genera modelo_vr_guardado/)
python -m finetuning.fine_tuning_modelo

# 2) Probar el clasificador
python -m finetuning.test_finetuning

# 3) Comparativa de encoders (gráficos para la memoria)
python -m finetuning.comparacion_finetuning

# 4) Arrancar el backend
python -m core.backend
#   (o: uvicorn core.backend:app --host 0.0.0.0 --port 3000)

# 5) Benchmark 1 — generar respuestas por nivel de contexto
python -m benchmarking.benchmark_runner \
    --dataset benchmarking/dataset/dataset.json \
    --out resultados/ --modelo qwen2.5vl:latest

# 6) Puntuar con VIEScore (juez Claude; necesita ANTHROPIC_API_KEY)
python -m benchmarking.benchmark_metrics resultados/benchmark1_raw.csv \
    --dataset benchmarking/dataset/dataset.json \
    --model claude-opus-4-7 --out resultados/
```

## IMPORTANTE: faltan tus plantillas de prompt
La carpeta `core/prompts/` está vacía porque los `.txt` no venían en los
archivos compartidos. Copia ahí tus plantillas reales (ver
`core/prompts/_LEER_IMPORTANTE.txt`). Sin ellas, `query_processor` lanzará
`FileNotFoundError` al arrancar (fail-fast, intencionado).
