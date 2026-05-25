"""
Fine-tuning del clasificador de intenciones (6 clases).

Versión "producción": entrena el modelo final con BETO y lo guarda en
./modelo_vr_guardado/. Las métricas se calculan y se imprimen por pantalla,
pero NO se persisten en disco (ni JSON, ni PNG, ni CSV).

El análisis comparativo y la generación de gráficos para la memoria del TFM
se hace en el script aparte de comparativa de encoders. Aquí solo nos
interesa producir el artefacto que cargará query_processor.py.

Pasos:
  1. Carga del CSV y fusión de cualquier etiqueta no operativa en
     'fuera_dominio'. Resultado: 6 clases.
  2. Split estratificado 70/15/15 con SEED fijo.
  3. Tokenización con BETO.
  4. Entrenamiento con early stopping sobre f1_macro.
  5. Evaluación sobre el test set: se imprimen accuracy, precision, recall,
     F1, ROC-AUC, PR-AUC (macro y weighted), classification_report y
     ROC-AUC por clase. Más diagnóstico de overfitting (gap val - train).
  6. Guardado del modelo + tokenizer en ./modelo_vr_guardado/.
"""

import random
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------
# 0. CONFIGURACIÓN Y REPRODUCIBILIDAD
# ---------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MODELO_ID = "dccuchile/bert-base-spanish-wwm-uncased"
DATASET_PATH = "dataset_consultas.csv"
MODELO_DIR = Path("./modelo_vr_guardado")

# Categorías operativas válidas (las que sí disparan el pipeline MLLM).
# Cualquier categoría fuera de este conjunto se reetiqueta como 'fuera_dominio'.
CATEGORIAS_OPERATIVAS = {
    "descripcion_escena",
    "localizacion_objeto",
    "detalle_objeto",
    "objetos_cercanos",
    "navegacion",
}
OOD_LABEL = "fuera_dominio"

print("=" * 70)
print("Fine-tuning del clasificador de intenciones (6 clases)")
print("=" * 70)

# ---------------------------------------------------------
# 1. CARGA, FUSIÓN OOD Y SPLIT
# ---------------------------------------------------------
print("\n📊 Cargando dataset...")
df = pd.read_csv(DATASET_PATH)

# Cualquier categoría que no esté en CATEGORIAS_OPERATIVAS se considera OOD.
# Esto cubre tanto 'consulta_general' (eliminada) como 'fuera_dominio'
# y cualquier etiqueta no contemplada que pudiera aparecer en el CSV.
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

# Split estratificado 70/15/15 (train/val/test)
df_train, df_temp = train_test_split(
    df, test_size=0.30, random_state=SEED, stratify=df["label"]
)
df_val, df_test = train_test_split(
    df_temp, test_size=0.50, random_state=SEED, stratify=df_temp["label"]
)
print(f"\n   Train: {len(df_train)}  |  Val: {len(df_val)}  |  Test: {len(df_test)}")

ds_train = Dataset.from_pandas(df_train[["texto", "label"]], preserve_index=False)
ds_val = Dataset.from_pandas(df_val[["texto", "label"]], preserve_index=False)
ds_test = Dataset.from_pandas(df_test[["texto", "label"]], preserve_index=False)

# ---------------------------------------------------------
# 2. TOKENIZACIÓN
# ---------------------------------------------------------
print("\n🔤 Cargando tokenizador BETO...")
tokenizer = AutoTokenizer.from_pretrained(MODELO_ID)


def tokenizar(ejemplos):
    return tokenizer(
        ejemplos["texto"],
        truncation=True,
        padding="max_length",
        max_length=64,
    )


ds_train_tok = ds_train.map(tokenizar, batched=True)
ds_val_tok = ds_val.map(tokenizar, batched=True)
ds_test_tok = ds_test.map(tokenizar, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------------------------------------------------------
# 3. MODELO Y MÉTRICAS DURANTE EL ENTRENAMIENTO
# ---------------------------------------------------------
modelo = AutoModelForSequenceClassification.from_pretrained(
    MODELO_ID,
    num_labels=len(etiquetas),
    id2label=id2label,
    label2id=label2id,
)

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

    # ROC-AUC One-vs-Rest. Si en este batch de eval falta alguna clase
    # (split pequeño puede ocurrir en val), sklearn lanza error; lo capturamos.
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


# ---------------------------------------------------------
# 4. ENTRENAMIENTO CON EARLY STOPPING
# ---------------------------------------------------------
args = TrainingArguments(
    output_dir="./checkpoints",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=8,  # generoso, lo limita el early stopping
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    seed=SEED,
    report_to="none",
    save_total_limit=2,
)

trainer = Trainer(
    model=modelo,
    args=args,
    train_dataset=ds_train_tok,
    eval_dataset=ds_val_tok,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("\n🚀 Iniciando entrenamiento...")
trainer.train()

# ---------------------------------------------------------
# 5. EVALUACIÓN SOBRE EL TEST SET (solo impresión por pantalla)
# ---------------------------------------------------------
print("\n📈 Evaluando sobre el TEST set (sin tocarlo antes)...")

pred_output = trainer.predict(ds_test_tok)
y_true = pred_output.label_ids
logits_test = pred_output.predictions
y_pred = np.argmax(logits_test, axis=-1)
y_proba = _softmax(logits_test)

# One-hot de y_true para AUC por clase (equivalente a sklearn.label_binarize
# pero sin arrastrar el import).
n_clases = len(etiquetas)
y_true_bin = np.eye(n_clases, dtype=int)[y_true]

# Métricas agregadas
acc_test = float((y_true == y_pred).mean())
precision_macro_test = precision_score(y_true, y_pred, average="macro", zero_division=0)
precision_weighted_test = precision_score(
    y_true, y_pred, average="weighted", zero_division=0
)
recall_macro_test = recall_score(y_true, y_pred, average="macro", zero_division=0)
recall_weighted_test = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1_macro_test = f1_score(y_true, y_pred, average="macro")
f1_weighted_test = f1_score(y_true, y_pred, average="weighted")

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
    y_true, y_pred, labels=list(range(n_clases)), zero_division=0
)

# ROC-AUC por clase (One-vs-Rest)
roc_auc_pc = {}
for i, nombre in enumerate(etiquetas):
    try:
        roc_auc_pc[nombre] = float(roc_auc_score(y_true_bin[:, i], y_proba[:, i]))
    except ValueError:
        # Si en test no hay ejemplos positivos o negativos de la clase,
        # ROC-AUC no está definido.
        roc_auc_pc[nombre] = float("nan")

# Resumen por pantalla
print(f"\n   --- Métricas globales (test) ---")
print(f"   Accuracy              : {acc_test:.4f}")
print(f"   Precision macro       : {precision_macro_test:.4f}")
print(f"   Precision weighted    : {precision_weighted_test:.4f}")
print(f"   Recall macro          : {recall_macro_test:.4f}")
print(f"   Recall weighted       : {recall_weighted_test:.4f}")
print(f"   F1 macro              : {f1_macro_test:.4f}")
print(f"   F1 weighted           : {f1_weighted_test:.4f}")
print(f"   ROC-AUC macro (OvR)   : {roc_auc_macro_test:.4f}")
print(f"   ROC-AUC weighted (OvR): {roc_auc_weighted_test:.4f}")
print(f"   PR-AUC macro          : {pr_auc_macro_test:.4f}")
print(f"   PR-AUC weighted       : {pr_auc_weighted_test:.4f}")

print("\n   --- Reporte por clase (test) ---")
reporte = classification_report(
    y_true, y_pred, target_names=etiquetas, zero_division=0, digits=3
)
print(reporte)

print("   ROC-AUC por clase (One-vs-Rest):")
for nombre, valor in roc_auc_pc.items():
    print(f"     {nombre:22s} : {valor:.4f}")

# ---------------------------------------------------------
# 6. DIAGNÓSTICO DE OVERFITTING (impresión, no se guarda)
# ---------------------------------------------------------
log_history = trainer.state.log_history

train_losses = []
eval_losses = []
for entry in log_history:
    if "loss" in entry and "eval_loss" not in entry and "epoch" in entry:
        train_losses.append((entry["epoch"], entry["loss"]))
    if "eval_loss" in entry and "epoch" in entry:
        eval_losses.append((entry["epoch"], entry["eval_loss"]))

if train_losses and eval_losses:
    last_train = train_losses[-1][1]
    last_eval = eval_losses[-1][1]
    overfitting_gap = last_eval - last_train
    print(f"\n🔍 Diagnóstico de overfitting (últimos valores):")
    print(f"   Train loss final : {last_train:.4f}")
    print(f"   Val loss final   : {last_eval:.4f}")
    print(f"   Gap (val - train): {overfitting_gap:.4f}")
    if overfitting_gap > 0.3:
        print(
            "   ⚠  Gap > 0.3 sugiere overfitting moderado. "
            "Si empeora, reducir epochs, añadir dropout o más datos."
        )
    elif overfitting_gap < 0.1:
        print("   ✅ Gap pequeño: el modelo generaliza bien.")
    else:
        print("   ℹ  Gap moderado, aceptable para este tamaño de dataset.")

# ---------------------------------------------------------
# 7. GUARDAR MODELO FINAL (único artefacto que persiste)
# ---------------------------------------------------------
trainer.save_model(str(MODELO_DIR))
tokenizer.save_pretrained(str(MODELO_DIR))
print(f"\n✅ Modelo final guardado en {MODELO_DIR}")
print("=" * 70)