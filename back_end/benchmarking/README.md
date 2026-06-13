# Benchmark 1 — Efecto del contexto · métricas con VIEScore (estilo AskVR)

Implementación del primer experimento de tu propuesta: medir cuánto aporta
cada nivel de información estructurada (C1→C4) fijando todo lo demás
(Qwen 2.5 VL, clasificador BETO, plantillas especializadas). La calidad de
las respuestas se mide **únicamente con VIEScore**, igual que hace AskVR
(Fernandez et al., MMM 2026) para evaluar descripciones de texto.

## Qué es VIEScore aquí

VIEScore (Ku et al., ACL 2024) es un evaluador *MLLM-as-judge* sin
entrenamiento. Da dos ejes en escala 0–10 y los combina por media geométrica:

    SC (Semantic Consistency) = min(sub-scores SC)
    PQ (Perceptual Quality)   = min(sub-scores PQ)
    O  (Overall)              = sqrt(SC · PQ)

VIEScore nació para imágenes generadas. **AskVR lo reutiliza para texto**,
reinterpretando los ejes así (cita textual del paper):
- **SC** = "the accuracy of content understanding and alignment with image semantics"
- **PQ** = "the realism, fluency and user comprehensibility of generated descriptions"

Este código replica ese enfoque usando **Claude como único juez** (vía la
Anthropic Messages API). Como las respuestas se generan con Qwen, usar Claude
de juez evita el sesgo de auto-evaluación. Las rúbricas concretas (sub-aspectos
de cada eje) están en `vie_score.py` y son editables:
- **SC**: grounding (sin alucinaciones), relevancia a la consulta y —solo para
  intenciones espaciales— corrección direccional/distancia.
- **PQ**: fluidez del lenguaje y comprensibilidad/utilidad para una persona
  ciega.

Detalle importante replicado de AskVR: la **O se calcula por ítem** y luego se
promedia. NO se recompone desde las medias de SC y PQ (por eso en su Tabla 1
O=0.477 ≠ √(0.761·0.406)).

## Archivos

- `query_processor.py` — pipeline parametrizado por `context_level` (C1–C4).
- `spatial_enricher.py` — tu módulo de enriquecimiento (sin cambios).
- `benchmark_runner.py` — ejecuta consultas × niveles y guarda CSV crudo
  (incluye `image_path`, que el juez VIEScore necesita).
- `vie_score.py` — **núcleo VIEScore**: juez Claude (Anthropic Messages API),
  rúbricas SC/PQ, parser robusto y fórmula O=√(SC·PQ). Requiere
  `pip install anthropic` y la variable `ANTHROPIC_API_KEY`.
- `benchmark_metrics.py` — aplica VIEScore a cada respuesta del CSV y agrega
  por nivel × intención.
- `dataset_example.json` — esqueleto del dataset.

## Flujo de uso

```bash
# 1. Generar respuestas bajo los 4 niveles de contexto
python benchmark_runner.py \
    --dataset benchmark_data/dataset.json \
    --out results/ --modelo qwen2.5vl:latest

# 2. Puntuar con VIEScore (juez Claude). Necesita ANTHROPIC_API_KEY.
python benchmark_metrics.py results/benchmark1_raw.csv \
    --dataset benchmark_data/dataset.json \
    --model claude-opus-4-7 --out results/
```

Salidas:
- `results/benchmark1_raw.csv` — respuestas, latencia, intención.
- `results/benchmark1_viescore.csv` — + SC, PQ, O (0–10 y 0–1) y el rationale
  del juez por cada respuesta.
- `results/benchmark1_viescore_summary.csv` — SC/PQ/O medios por nivel ×
  intención. Es la tabla que va en la memoria.

## Recomendaciones para que el experimento sea válido

1. **Juez ≠ generador.** Las respuestas se generan con Qwen y se juzgan con
   Claude, evitando que el modelo se auto-favorezca (crítica habitual al
   MLLM-as-judge; AskVR no aborda este punto).
2. **`temperature=0` en el juez** (ya por defecto) para reproducibilidad.
3. **Repite la evaluación** (p.ej. 3 pasadas) y reporta media ± desviación:
   el juez tiene varianza aunque T=0.
4. Las consultas OOD y las respuestas vacías se marcan `vie_error=ood_or_empty`
   y NO se puntúan (VIEScore no aplica a la respuesta canned).
5. El modelo de visión genera las respuestas directamente en español
   (`response_es`); no hay paso de traducción.

## Contraste de hipótesis

Con `benchmark1_viescore_summary.csv`:
- **C1→C2** (¿la lista de objetos mejora todo?): compara `vie_sc_mean` en la
  fila `__ALL__` entre C1 y C2.
- **C3→C4** (¿el preprocesamiento espacial ayuda sobre todo en localización?):
  compara `vie_overall_mean` en `localizacion_objeto` y `objetos_cercanos`
  frente a `descripcion_escena`.

Para significancia formal: Wilcoxon signed-rank sobre la O por ítem entre pares
de niveles (mismo test que ENVISIONVR usó entre NVR y EVR).

## Modelo juez

Por defecto `claude-opus-4-7` (el más capaz). Para abaratar pasadas repetidas
puedes usar `claude-sonnet-4-6` o `claude-haiku-4-5-20251001` con `--model`.
Comprueba los nombres vigentes en
https://docs.claude.com/en/docs/about-claude/models
