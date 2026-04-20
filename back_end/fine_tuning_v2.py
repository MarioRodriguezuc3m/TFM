"""
Fine-tuning del clasificador de intenciones (7 clases) con métricas robustas.

Cambios respecto a la versión anterior:
  1. Dataset ampliado: 1050 ejemplos, 7 categorías (se añade 'fuera_dominio').
  2. Split estratificado 70/15/15 (train/validación/test) para preservar la
     distribución de clases en cada partición.
  3. Métricas reportadas (no solo accuracy):
       - accuracy global
       - F1 macro, F1 weighted
       - precision/recall/F1 por clase
       - matriz de confusión (heatmap PNG + CSV)
       - curva de aprendizaje (train_loss vs eval_loss por epoch → PNG)
  4. Early stopping sobre eval_loss para mitigar overfitting.
  5. Cálculo explícito del gap train-vs-eval al final (señal de overfitting).
  6. Todas las métricas se serializan a metrics.json para la memoria del TFM.

Todo se guarda en:
  ./modelo_vr_guardado/        (pesos + tokenizer)
  ./resultados/metrics.json    (métricas finales)
  ./resultados/confusion_matrix.png
  ./resultados/confusion_matrix.csv
  ./resultados/learning_curve.png
  ./resultados/classification_report.txt
"""

import json
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin interfaz gráfica (reproducible)
import matplotlib.pyplot as plt

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
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
DATASET_PATH = "dataset_intent_vr_1050.csv"
RESULTADOS_DIR = Path("./resultados")
RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
MODELO_DIR = Path("./modelo_vr_guardado")

print("=" * 70)
print("Fine-tuning del clasificador de intenciones (7 clases)")
print("=" * 70)

# ---------------------------------------------------------
# 1. CARGA Y SPLIT ESTRATIFICADO
# ---------------------------------------------------------
print("\n📊 Cargando dataset...")
df = pd.read_csv(DATASET_PATH)

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
# 3. MODELO Y MÉTRICAS
# ---------------------------------------------------------
modelo = AutoModelForSequenceClassification.from_pretrained(
    MODELO_ID,
    num_labels=len(etiquetas),
    id2label=id2label,
    label2id=label2id,
)

metric_acc = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """Métricas durante entrenamiento: accuracy + F1 macro + F1 weighted."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
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
# 5. EVALUACIÓN SOBRE EL TEST SET
# ---------------------------------------------------------
print("\n📈 Evaluando sobre el TEST set (sin tocarlo antes)...")

pred_output = trainer.predict(ds_test_tok)
y_true = pred_output.label_ids
y_pred = np.argmax(pred_output.predictions, axis=-1)

acc_test = (y_true == y_pred).mean()
f1_macro_test = f1_score(y_true, y_pred, average="macro")
f1_weighted_test = f1_score(y_true, y_pred, average="weighted")

precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=list(range(len(etiquetas))), zero_division=0
)

per_class = {
    etiquetas[i]: {
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "f1": float(f1[i]),
        "support": int(support[i]),
    }
    for i in range(len(etiquetas))
}

print(f"\n   Accuracy test   : {acc_test:.4f}")
print(f"   F1 macro test   : {f1_macro_test:.4f}")
print(f"   F1 weighted test: {f1_weighted_test:.4f}")
print("\n   Reporte por clase (test):")
reporte = classification_report(
    y_true, y_pred, target_names=etiquetas, zero_division=0, digits=3
)
print(reporte)
(RESULTADOS_DIR / "classification_report.txt").write_text(reporte, encoding="utf-8")

# ---------------------------------------------------------
# 6. MATRIZ DE CONFUSIÓN (PNG + CSV)
# ---------------------------------------------------------
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(etiquetas))))
cm_df = pd.DataFrame(cm, index=etiquetas, columns=etiquetas)
cm_df.to_csv(RESULTADOS_DIR / "confusion_matrix.csv", encoding="utf-8")

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(etiquetas)))
ax.set_yticks(range(len(etiquetas)))
ax.set_xticklabels(etiquetas, rotation=45, ha="right")
ax.set_yticklabels(etiquetas)
ax.set_xlabel("Predicho")
ax.set_ylabel("Real")
ax.set_title(f"Matriz de confusión - Test set (accuracy={acc_test:.3f})")

# Anotar cada celda
vmax = cm.max() if cm.max() > 0 else 1
for i in range(len(etiquetas)):
    for j in range(len(etiquetas)):
        color = "white" if cm[i, j] > vmax / 2 else "black"
        ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color)

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "confusion_matrix.png", dpi=150)
plt.close(fig)
print(f"   ✔ Matriz de confusión guardada en {RESULTADOS_DIR}/confusion_matrix.png")

# ---------------------------------------------------------
# 7. CURVA DE APRENDIZAJE (train_loss vs eval_loss por epoch)
# ---------------------------------------------------------
log_history = trainer.state.log_history

train_losses = []   # (epoch, train_loss)
eval_losses = []    # (epoch, eval_loss)
eval_f1 = []        # (epoch, f1_macro)

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
    ax1.plot(ep, lo, "s-", label="Validation loss", color="tab:orange")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Curva de aprendizaje (loss)")
ax1.legend()
ax1.grid(True, alpha=0.3)

if eval_f1:
    ep, f1v = zip(*eval_f1)
    ax2.plot(ep, f1v, "^-", color="tab:green", label="Val F1 macro")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("F1 macro")
ax2.set_title("F1 macro de validación")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "learning_curve.png", dpi=150)
plt.close(fig)
print(f"   ✔ Curva de aprendizaje guardada en {RESULTADOS_DIR}/learning_curve.png")

# ---------------------------------------------------------
# 8. DIAGNÓSTICO DE OVERFITTING
# ---------------------------------------------------------
overfitting_gap = None
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
# 9. SERIALIZAR MÉTRICAS A JSON
# ---------------------------------------------------------
metricas_finales = {
    "config": {
        "modelo_base": MODELO_ID,
        "num_clases": len(etiquetas),
        "clases": etiquetas,
        "tamano_dataset_total": len(df),
        "tamano_train": len(df_train),
        "tamano_val": len(df_val),
        "tamano_test": len(df_test),
        "seed": SEED,
    },
    "test_metrics": {
        "accuracy": float(acc_test),
        "f1_macro": float(f1_macro_test),
        "f1_weighted": float(f1_weighted_test),
        "per_class": per_class,
    },
    "overfitting_gap_val_minus_train": (
        float(overfitting_gap) if overfitting_gap is not None else None
    ),
}
with open(RESULTADOS_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metricas_finales, f, ensure_ascii=False, indent=2)
print(f"   ✔ Métricas finales guardadas en {RESULTADOS_DIR}/metrics.json")

# ---------------------------------------------------------
# 10. GUARDAR MODELO FINAL
# ---------------------------------------------------------
trainer.save_model(str(MODELO_DIR))
tokenizer.save_pretrained(str(MODELO_DIR))
print(f"\n✅ Modelo final guardado en {MODELO_DIR}")
print("=" * 70)