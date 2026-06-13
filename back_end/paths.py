"""
Resolución centralizada de rutas del proyecto.

Todas las carpetas de datos/artefactos viven en la RAÍZ del proyecto
(back_end/), no dentro de los paquetes. Cualquier script, sin importar desde
qué carpeta se ejecute, importa estas constantes en lugar de usar rutas
relativas frágiles como "./modelo_vr_guardado".

Uso:
    from paths import MODELO_DIR, DATASET_CONSULTAS, CURRENT_INPUT
"""

from pathlib import Path

# Raíz del proyecto = carpeta que contiene este archivo (back_end/)
ROOT = Path(__file__).resolve().parent

# Artefactos del clasificador (salida del finetuning, entrada del core)
MODELO_DIR = ROOT / "modelo_vr_guardado"
CHECKPOINTS_DIR = ROOT / "checkpoints"

# Datos de finetuning
DATASET_CONSULTAS = ROOT / "finetuning" / "dataset_consultas.csv"

# Resultados
RESULTADOS_DIR = ROOT / "resultados"
RESULTADOS_COMPARATIVA_DIR = ROOT / "resultados_comparativa"

# I/O temporal del servidor
CURRENT_INPUT = ROOT / "current_input"

# Logs de sesión (una carpeta por arranque del servidor, una entrada por consulta)
SESSION_LOGS_DIR = ROOT / "session_logs"

# Frontend (escena A-Frame) servido como estáticos por el backend
FRONTEND_DIR = ROOT.parent / "front_end"

# Datos de benchmarking
BENCHMARK_DATASET_DIR = ROOT / "benchmarking" / "dataset"
BENCHMARK_IMAGES_DIR = BENCHMARK_DATASET_DIR / "images"