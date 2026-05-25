"""
Fine-tuning + comparativa de encoders para el clasificador de intenciones (6 clases).

Diferencias respecto a fine_tuning_v3.py
----------------------------------------
Esta versión NO entrena un único modelo, sino que ejecuta el MISMO pipeline de
fine-tuning sobre VARIOS encoders manteniendo fijos:
  - El dataset (mismo CSV)
  - El split estratificado 70/15/15 (mismo SEED -> mismos índices)
  - Los hiperparámetros (lr, batch size, epochs, weight decay, early stopping)
  - Las métricas de evaluación (mismas que v3: accuracy, precision, recall,
    F1, ROC-AUC y PR-AUC, macro y weighted)

Lo único que varía entre experimentos es el ENCODER base. De esta forma, las
diferencias en las métricas se pueden atribuir al modelo y no al setup.

Encoders comparados:
  - BETO (uncased)
  - RoBERTuito (uncased, informal en español)
  - XLM-RoBERTa-base (multilingüe)
  - mDeBERTa-v3-base (multilingüe)
  - DistilBETO (rápido, frontera de Pareto)

Salidas (un subdirectorio por modelo):
  ./resultados_comparativa/<modelo_safe_name>/
      classification_report.txt
      confusion_matrix.png
      confusion_matrix.csv
      roc_curves.png
      learning_curve.png
      metrics.json
  ./resultados_comparativa/
      comparativa_resumen.csv        <- tabla maestra para la memoria
      comparativa_resumen.json
      comparativa_f1_macro.png       <- gráfico de barras comparativo
      comparativa_accuracy_vs_params.png  <- frontera de Pareto

Esta versión NO guarda los pesos de ningún modelo: solo recopila métricas
para la comparativa. El modelo de producción se decide más tarde con los
resultados a la vista.
"""

import gc
import json
import os
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# =====================================================================
# 0. CONFIGURACIÓN GLOBAL Y REPRODUCIBILIDAD
# =====================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATASET_PATH = "dataset_consultas.csv"
RESULTADOS_DIR = Path("./resultados_comparativa")
RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

# Categorías operativas válidas (las 5 que sí disparan el pipeline MLLM).
# Cualquier categoría fuera de este conjunto se reetiqueta como 'fuera_dominio'
# antes de entrenar. Esto absorbe la antigua 'consulta_general' y cualquier
# etiqueta huérfana que pudiera quedar en el CSV.
CATEGORIAS_OPERATIVAS = {
    "descripcion_escena",
    "localizacion_objeto",
    "detalle_objeto",
    "objetos_cercanos",
    "navegacion",
}
OOD_LABEL = "fuera_dominio"

# Modelos a comparar. Para cada uno se entrena el MISMO pipeline.
#
# Notas prácticas:
#   - 'uncased' indica si el tokenizador hace lowercasing. Solo es informativo
#     aquí; el AutoTokenizer ya hace lo correcto a partir de la config del repo.
#   - Algunos modelos multilingües son notablemente más grandes (xlm-r,
#     mDeBERTa). Si tu GPU es limitada, baja per_device_train_batch_size en
#     CONFIG_ENTRENAMIENTO o usa fp16=True.
MODELOS_A_COMPARAR = [
    {
        "alias": "BETO-uncased",
        "model_id": "dccuchile/bert-base-spanish-wwm-uncased",
    },
    {
        "alias": "RoBERTuito",
        "model_id": "pysentimiento/robertuito-base-uncased",
    },
    {
        "alias": "XLM-RoBERTa-base",
        "model_id": "FacebookAI/xlm-roberta-base",
    },
    {
        "alias": "mDeBERTa-v3-base",
        "model_id": "microsoft/mdeberta-v3-base",
    },
    {
        "alias": "DistilBETO",
        "model_id": "dccuchile/distilbert-base-spanish-uncased",
    },
]

# Hiperparámetros compartidos por TODOS los experimentos (fairness).
# Si cambias algo aquí, afecta a todos los modelos por igual.
CONFIG_ENTRENAMIENTO = {
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "num_train_epochs": 8,
    "weight_decay": 0.01,
    "max_length": 64,
    "early_stopping_patience": 2,
}


# =====================================================================
# 1. CARGA, FUSIÓN OOD Y SPLIT (UNA SOLA VEZ, COMPARTIDO POR TODOS LOS MODELOS)
# =====================================================================
print("=" * 70)
print("Comparativa de encoders para clasificación de intenciones (6 clases)")
print("=" * 70)

print("\n📊 Cargando dataset...")
df = pd.read_csv(DATASET_PATH)

# Fusión: cualquier categoría que no esté en CATEGORIAS_OPERATIVAS pasa a OOD.
# Esto cubre 'consulta_general' (ya eliminada del dataset) y cualquier
# etiqueta huérfana que pudiera aparecer.
n_fusionados = (~df["categoria"].isin(CATEGORIAS_OPERATIVAS | {OOD_LABEL})).sum()
df["categoria"] = df["categoria"].where(
    df["categoria"].isin(CATEGORIAS_OPERATIVAS), OOD_LABEL
)
if n_fusionados > 0:
    print(
        f"   ℹ  {n_fusionados} ejemplos de categorías no operativas "
        f"reetiquetados como '{OOD_LABEL}'."
    )

etiquetas = sorted(df["categoria"].unique().tolist())
label2id = {label: i for i, label in enumerate(etiquetas)}
id2label = {i: label for i, label in enumerate(etiquetas)}
df["label"] = df["categoria"].map(label2id)

print(f"   Total ejemplos: {len(df)}")
print(f"   Clases ({len(etiquetas)}): {etiquetas}")
print("   Distribución por clase:")
for cat, n in df["categoria"].value_counts().sort_index().items():
    print(f"     · {cat:22s} -> {n}")

# Split estratificado 70/15/15 con SEED fijo -> los índices son los mismos
# para todos los modelos. Esto es lo que garantiza una comparación justa.
df_train, df_temp = train_test_split(
    df, test_size=0.30, random_state=SEED, stratify=df["label"]
)
df_val, df_test = train_test_split(
    df_temp, test_size=0.50, random_state=SEED, stratify=df_temp["label"]
)
print(f"\n   Train: {len(df_train)}  |  Val: {len(df_val)}  |  Test: {len(df_test)}")

# Guardamos los DataFrames "crudos" (texto + label). La tokenización se hace
# DENTRO del bucle porque cada modelo tiene su propio tokenizer.


# =====================================================================
# 2. UTILIDADES COMPARTIDAS
# =====================================================================
metric_acc = evaluate.load("accuracy")


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax estable por filas."""
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def compute_metrics(eval_pred):
    """Métricas durante entrenamiento: accuracy, precision, recall, F1 y ROC-AUC."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = _softmax(logits)

    acc = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
    precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
    precision_weighted = precision_score(
        labels, preds, average="weighted", zero_division=0
    )
    recall_macro = recall_score(labels, preds, average="macro", zero_division=0)
    recall_weighted = recall_score(labels, preds, average="weighted", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")

    # ROC-AUC One-vs-Rest. Si en el batch de eval falta alguna clase, se
    # captura para no tirar el entrenamiento.
    try:
        roc_auc_macro = roc_auc_score(
            labels, probs, multi_class="ovr", average="macro"
        )
        roc_auc_weighted = roc_auc_score(
            labels, probs, multi_class="ovr", average="weighted"
        )
    except ValueError:
        roc_auc_macro = float("nan")
        roc_auc_weighted = float("nan")

    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted,
    }


def safe_name(s: str) -> str:
    """Convierte un alias o model_id en un nombre de carpeta seguro."""
    return s.replace("/", "_").replace(" ", "_")


def liberar_memoria():
    """Libera memoria de GPU/CPU entre experimentos para evitar OOM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _safe_float(x):
    """Convierte a float, mapeando NaN a None para que JSON lo serialice limpio."""
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


# =====================================================================
# 3. FUNCIÓN PRINCIPAL: ENTRENA UN MODELO Y DEVUELVE SUS MÉTRICAS
# =====================================================================
def entrenar_y_evaluar(alias: str, model_id: str, output_dir: Path) -> dict:
    """
    Ejecuta el pipeline completo de fine-tuning + evaluación + plots
    para un encoder concreto. Devuelve un dict con las métricas resumidas
    listas para acumular en la tabla comparativa.
    """
    print("\n" + "#" * 70)
    print(f"# Entrenando: {alias}  ({model_id})")
    print("#" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # 3.1 Tokenización (específica de este encoder)
    # -----------------------------------------------------------------
    print(f"🔤 Cargando tokenizador {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tokenizar(ejemplos):
        return tokenizer(
            ejemplos["texto"],
            truncation=True,
            padding="max_length",
            max_length=CONFIG_ENTRENAMIENTO["max_length"],
        )

    ds_train = Dataset.from_pandas(
        df_train[["texto", "label"]], preserve_index=False
    ).map(tokenizar, batched=True)
    ds_val = Dataset.from_pandas(
        df_val[["texto", "label"]], preserve_index=False
    ).map(tokenizar, batched=True)
    ds_test = Dataset.from_pandas(
        df_test[["texto", "label"]], preserve_index=False
    ).map(tokenizar, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # -----------------------------------------------------------------
    # 3.2 Modelo
    # -----------------------------------------------------------------
    modelo = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(etiquetas),
        id2label=id2label,
        label2id=label2id,
    )
    num_params = sum(p.numel() for p in modelo.parameters())
    num_params_trainable = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"   Parámetros totales:    {num_params:,}")
    print(f"   Parámetros entrenables:{num_params_trainable:,}")

    # -----------------------------------------------------------------
    # 3.3 Entrenamiento
    # -----------------------------------------------------------------
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=CONFIG_ENTRENAMIENTO["learning_rate"],
        per_device_train_batch_size=CONFIG_ENTRENAMIENTO["per_device_train_batch_size"],
        per_device_eval_batch_size=CONFIG_ENTRENAMIENTO["per_device_eval_batch_size"],
        num_train_epochs=CONFIG_ENTRENAMIENTO["num_train_epochs"],
        weight_decay=CONFIG_ENTRENAMIENTO["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=SEED,
        report_to="none",
        save_total_limit=1,  # solo nos quedamos con el best, ahorra disco
    )

    trainer = Trainer(
        model=modelo,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=CONFIG_ENTRENAMIENTO["early_stopping_patience"]
            )
        ],
    )

    print("🚀 Entrenando...")
    t0 = time.time()
    trainer.train()
    tiempo_entrenamiento = time.time() - t0
    print(f"   ⏱  Tiempo de entrenamiento: {tiempo_entrenamiento:.1f}s")

    # -----------------------------------------------------------------
    # 3.4 Evaluación en TEST (métricas ampliadas, simétricas a v3)
    # -----------------------------------------------------------------
    print("📈 Evaluando en TEST...")
    pred_output = trainer.predict(ds_test)
    y_true = pred_output.label_ids
    logits_test = pred_output.predictions
    y_pred = np.argmax(logits_test, axis=-1)
    y_proba = _softmax(logits_test)

    # Binarización para AUC One-vs-Rest
    y_true_bin = label_binarize(y_true, classes=list(range(len(etiquetas))))

    # Métricas agregadas
    acc_test = float((y_true == y_pred).mean())
    precision_macro_test = precision_score(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted_test = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    recall_macro_test = recall_score(y_true, y_pred, average="macro", zero_division=0)
    recall_weighted_test = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    f1_macro_test = float(f1_score(y_true, y_pred, average="macro"))
    f1_weighted_test = float(f1_score(y_true, y_pred, average="weighted"))

    try:
        roc_auc_macro_test = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro"
        )
        roc_auc_weighted_test = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="weighted"
        )
    except ValueError as e:
        print(f"   ⚠ No se pudo calcular ROC-AUC global: {e}")
        roc_auc_macro_test = float("nan")
        roc_auc_weighted_test = float("nan")

    try:
        pr_auc_macro_test = average_precision_score(y_true_bin, y_proba, average="macro")
        pr_auc_weighted_test = average_precision_score(
            y_true_bin, y_proba, average="weighted"
        )
    except ValueError as e:
        print(f"   ⚠ No se pudo calcular PR-AUC global: {e}")
        pr_auc_macro_test = float("nan")
        pr_auc_weighted_test = float("nan")

    # Métricas por clase
    precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(etiquetas))), zero_division=0
    )

    # ROC-AUC por clase (One-vs-Rest)
    roc_auc_pc = {}
    for i, nombre in enumerate(etiquetas):
        try:
            roc_auc_pc[nombre] = float(roc_auc_score(y_true_bin[:, i], y_proba[:, i]))
        except ValueError:
            roc_auc_pc[nombre] = float("nan")

    per_class = {
        etiquetas[i]: {
            "precision": float(precision_pc[i]),
            "recall": float(recall_pc[i]),
            "f1": float(f1_pc[i]),
            "roc_auc": roc_auc_pc[etiquetas[i]],
            "support": int(support_pc[i]),
        }
        for i in range(len(etiquetas))
    }

    print(f"   Accuracy              : {acc_test:.4f}")
    print(f"   Precision macro       : {precision_macro_test:.4f}")
    print(f"   Recall macro          : {recall_macro_test:.4f}")
    print(f"   F1 macro              : {f1_macro_test:.4f}")
    print(f"   F1 weighted           : {f1_weighted_test:.4f}")
    print(f"   ROC-AUC macro (OvR)   : {roc_auc_macro_test:.4f}")
    print(f"   PR-AUC macro          : {pr_auc_macro_test:.4f}")

    reporte = classification_report(
        y_true, y_pred, target_names=etiquetas, zero_division=0, digits=3
    )
    reporte_extra = "\nROC-AUC por clase (One-vs-Rest):\n"
    for nombre, valor in roc_auc_pc.items():
        reporte_extra += f"  {nombre:22s} : {valor:.4f}\n"
    (output_dir / "classification_report.txt").write_text(
        reporte + reporte_extra, encoding="utf-8"
    )

    # -----------------------------------------------------------------
    # 3.5 Matriz de confusión
    # -----------------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(etiquetas))))
    cm_df = pd.DataFrame(cm, index=etiquetas, columns=etiquetas)
    cm_df.to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(etiquetas)))
    ax.set_yticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, rotation=45, ha="right")
    ax.set_yticklabels(etiquetas)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"{alias} - Test (acc={acc_test:.3f}, F1 macro={f1_macro_test:.3f})")
    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(len(etiquetas)):
        for j in range(len(etiquetas)):
            color = "white" if cm[i, j] > vmax / 2 else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------
    # 3.6 Curvas ROC por clase (One-vs-Rest)
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, nombre in enumerate(etiquetas):
        if np.isnan(roc_auc_pc[nombre]):
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, label=f"{nombre} (AUC = {roc_auc_pc[nombre]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Azar")
    ax.set_xlabel("Tasa de falsos positivos (FPR)")
    ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
    ax.set_title(f"{alias} - Curvas ROC (OvR)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------
    # 3.7 Curva de aprendizaje
    # -----------------------------------------------------------------
    log_history = trainer.state.log_history
    train_losses, eval_losses, eval_f1 = [], [], []
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry and "epoch" in entry:
            train_losses.append((entry["epoch"], entry["loss"]))
        if "eval_loss" in entry and "epoch" in entry:
            eval_losses.append((entry["epoch"], entry["eval_loss"]))
        if "eval_f1_macro" in entry and "epoch" in entry:
            eval_f1.append((entry["epoch"], entry["eval_f1_macro"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    if train_losses:
        ep, lo = zip(*train_losses)
        ax1.plot(ep, lo, "o-", label="Train loss", color="tab:blue")
    if eval_losses:
        ep, lo = zip(*eval_losses)
        ax1.plot(ep, lo, "s-", label="Val loss", color="tab:orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{alias} - Curva de aprendizaje")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if eval_f1:
        ep, f1v = zip(*eval_f1)
        ax2.plot(ep, f1v, "^-", color="tab:green", label="Val F1 macro")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 macro")
    ax2.set_title(f"{alias} - F1 macro en validación")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curve.png", dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------
    # 3.8 Gap de overfitting
    # -----------------------------------------------------------------
    overfitting_gap = None
    if train_losses and eval_losses:
        last_train = train_losses[-1][1]
        last_eval = eval_losses[-1][1]
        overfitting_gap = float(last_eval - last_train)

    # -----------------------------------------------------------------
    # 3.9 Latencia de inferencia (útil para Pareto velocidad/calidad)
    # -----------------------------------------------------------------
    print("⏱  Midiendo latencia de inferencia en CPU...")
    modelo.eval()
    modelo_cpu = modelo.to("cpu")
    ejemplos_latencia = df_test["texto"].sample(
        n=min(50, len(df_test)), random_state=SEED
    ).tolist()
    # Warm-up
    with torch.no_grad():
        for txt in ejemplos_latencia[:3]:
            inputs = tokenizer(txt, return_tensors="pt", truncation=True,
                               padding="max_length", max_length=CONFIG_ENTRENAMIENTO["max_length"])
            _ = modelo_cpu(**inputs)
    # Medición
    t0 = time.time()
    with torch.no_grad():
        for txt in ejemplos_latencia:
            inputs = tokenizer(txt, return_tensors="pt", truncation=True,
                               padding="max_length", max_length=CONFIG_ENTRENAMIENTO["max_length"])
            _ = modelo_cpu(**inputs)
    latencia_ms = (time.time() - t0) / len(ejemplos_latencia) * 1000
    print(f"   Latencia media CPU: {latencia_ms:.2f} ms/consulta")

    # -----------------------------------------------------------------
    # 3.10 Serializar métricas del modelo
    # -----------------------------------------------------------------
    metricas = {
        "alias": alias,
        "model_id": model_id,
        "num_params_total": int(num_params),
        "num_params_trainable": int(num_params_trainable),
        "tiempo_entrenamiento_s": float(tiempo_entrenamiento),
        "latencia_cpu_ms_por_consulta": float(latencia_ms),
        "test_metrics": {
            "accuracy": _safe_float(acc_test),
            "precision_macro": _safe_float(precision_macro_test),
            "precision_weighted": _safe_float(precision_weighted_test),
            "recall_macro": _safe_float(recall_macro_test),
            "recall_weighted": _safe_float(recall_weighted_test),
            "f1_macro": _safe_float(f1_macro_test),
            "f1_weighted": _safe_float(f1_weighted_test),
            "roc_auc_macro_ovr": _safe_float(roc_auc_macro_test),
            "roc_auc_weighted_ovr": _safe_float(roc_auc_weighted_test),
            "pr_auc_macro": _safe_float(pr_auc_macro_test),
            "pr_auc_weighted": _safe_float(pr_auc_weighted_test),
            "per_class": per_class,
        },
        "overfitting_gap_val_minus_train": _safe_float(overfitting_gap),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    # -----------------------------------------------------------------
    # 3.11 (omitido) No guardamos los pesos del modelo: esta versión solo
    # mide y compara. Si más adelante quieres recuperar un modelo concreto
    # para producción, descomenta el bloque siguiente:
    #
    #   trainer.save_model(str(output_dir / "modelo"))
    #   tokenizer.save_pretrained(str(output_dir / "modelo"))
    # -----------------------------------------------------------------

    # Limpieza
    del trainer, modelo, modelo_cpu, ds_train, ds_val, ds_test
    liberar_memoria()

    return metricas


# =====================================================================
# 4. BUCLE PRINCIPAL: ENTRENA TODOS LOS MODELOS
# =====================================================================
resultados_globales = []

for cfg in MODELOS_A_COMPARAR:
    alias = cfg["alias"]
    model_id = cfg["model_id"]
    sub_dir = RESULTADOS_DIR / safe_name(alias)

    try:
        metricas = entrenar_y_evaluar(alias, model_id, sub_dir)
        resultados_globales.append(metricas)
    except Exception as e:
        # Si un modelo falla (p. ej. por OOM o por no estar en el cache), lo
        # registramos y continuamos con el siguiente. Así una sola caída no
        # tira toda la comparativa.
        print(f"\n❌ ERROR entrenando {alias}: {e}")
        resultados_globales.append({
            "alias": alias,
            "model_id": model_id,
            "error": str(e),
        })
        liberar_memoria()


# =====================================================================
# 5. TABLA COMPARATIVA Y GRÁFICOS RESUMEN
# =====================================================================
print("\n" + "=" * 70)
print("📊 Generando tabla comparativa...")
print("=" * 70)

filas = []
for r in resultados_globales:
    if "error" in r:
        filas.append({
            "alias": r["alias"],
            "model_id": r["model_id"],
            "accuracy": None,
            "precision_macro": None,
            "recall_macro": None,
            "f1_macro": None,
            "f1_weighted": None,
            "roc_auc_macro": None,
            "pr_auc_macro": None,
            "num_params_M": None,
            "tiempo_entrenamiento_s": None,
            "latencia_cpu_ms": None,
            "overfitting_gap": None,
            "error": r["error"],
        })
    else:
        tm = r["test_metrics"]
        filas.append({
            "alias": r["alias"],
            "model_id": r["model_id"],
            "accuracy": tm["accuracy"],
            "precision_macro": tm["precision_macro"],
            "recall_macro": tm["recall_macro"],
            "f1_macro": tm["f1_macro"],
            "f1_weighted": tm["f1_weighted"],
            "roc_auc_macro": tm["roc_auc_macro_ovr"],
            "pr_auc_macro": tm["pr_auc_macro"],
            "num_params_M": r["num_params_total"] / 1e6,
            "tiempo_entrenamiento_s": r["tiempo_entrenamiento_s"],
            "latencia_cpu_ms": r["latencia_cpu_ms_por_consulta"],
            "overfitting_gap": r["overfitting_gap_val_minus_train"],
            "error": None,
        })

resumen_df = pd.DataFrame(filas).sort_values("f1_macro", ascending=False, na_position="last")
resumen_df.to_csv(RESULTADOS_DIR / "comparativa_resumen.csv", index=False, encoding="utf-8")
with open(RESULTADOS_DIR / "comparativa_resumen.json", "w", encoding="utf-8") as f:
    json.dump(filas, f, ensure_ascii=False, indent=2)

print("\nResumen (ordenado por F1 macro):")
print(resumen_df.to_string(index=False))

# --------- Gráfico 1: barras de F1 macro y accuracy ----------
df_ok = resumen_df.dropna(subset=["f1_macro"]).copy()
if len(df_ok) > 0:
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(df_ok))
    w = 0.35
    ax.bar(x - w / 2, df_ok["f1_macro"], w, label="F1 macro", color="tab:blue")
    ax.bar(x + w / 2, df_ok["accuracy"], w, label="Accuracy", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(df_ok["alias"], rotation=20, ha="right")
    ax.set_ylabel("Métrica")
    ax.set_ylim(0, 1.05)
    ax.set_title("Comparativa de encoders en test (mismo dataset, mismo split)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for i, (f1v, accv) in enumerate(zip(df_ok["f1_macro"], df_ok["accuracy"])):
        ax.text(i - w / 2, f1v + 0.01, f"{f1v:.3f}", ha="center", fontsize=9)
        ax.text(i + w / 2, accv + 0.01, f"{accv:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTADOS_DIR / "comparativa_f1_macro.png", dpi=150)
    plt.close(fig)

    # --------- Gráfico 2: Pareto calidad vs latencia ----------
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df_ok["latencia_cpu_ms"], df_ok["f1_macro"], s=120, color="tab:green")
    for _, row in df_ok.iterrows():
        ax.annotate(
            row["alias"],
            (row["latencia_cpu_ms"], row["f1_macro"]),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=10,
        )
    ax.set_xlabel("Latencia CPU (ms/consulta)  →  más rápido a la izquierda")
    ax.set_ylabel("F1 macro en test  →  mejor arriba")
    ax.set_title("Frontera de Pareto: calidad vs. velocidad")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTADOS_DIR / "comparativa_accuracy_vs_params.png", dpi=150)
    plt.close(fig)

# =====================================================================
# 6. RESUMEN FINAL
# =====================================================================
# Nota: esta versión NO promociona ningún modelo a ./modelo_vr_guardado/.
# El objetivo es solo medir y comparar; el modelo de producción se decide
# después a la vista de los resultados.
if len(df_ok) > 0:
    ganador_alias = df_ok.iloc[0]["alias"]
    ganador_f1 = df_ok.iloc[0]["f1_macro"]
    print("\n" + "=" * 70)
    print(f"🏆 Mejor F1 macro: {ganador_alias} ({ganador_f1:.4f})")
    print("=" * 70)
else:
    print("\n⚠️  Ningún modelo terminó correctamente.")

print("\n" + "=" * 70)
print("✅ Comparativa completa.")
print(f"   Detalle por modelo:  {RESULTADOS_DIR}/<alias>/")
print(f"   Resumen global:      {RESULTADOS_DIR}/comparativa_resumen.csv")
print(f"   Gráficos:            {RESULTADOS_DIR}/comparativa_*.png")
print("=" * 70)