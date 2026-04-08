import pandas as pd
import numpy as np
import torch
import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)

# ---------------------------------------------------------
# 1. CARGA Y PREPARACIÓN DE DATOS
# ---------------------------------------------------------
print("📊 Cargando el dataset...")
df = pd.read_csv("dataset_intent_vr_500.csv")

# Extraer etiquetas únicas y crear diccionarios de mapeo
etiquetas = df['categoria'].unique().tolist()
label2id = {label: i for i, label in enumerate(etiquetas)}
id2label = {i: label for i, label in enumerate(etiquetas)}

"""
Las categorías son las siguientes:
- descripcion_escena: El usuario quiere una descripción general de la escena.
- localizacion_objeto: El usuario pregunta por la ubicación de un objeto específico.
- detalle_objeto: El usuario quiere detalles visuales de un objeto (color, textura, etc.).
- objetos_cercanos: El usuario pregunta qué objetos hay cerca de él.
- consulta_general: Preguntas generales que no encajan en las anteriores (ej. "¿Puedo salir por aquí?").
- navegacion: El usuario da instrucciones de movimiento (ej. "Llévame a la puerta").
"""
# Convertir la columna de texto a enteros
df['label'] = df['categoria'].map(label2id)

# Crear dataset de HuggingFace y dividir en Entrenamiento (80%) y Validación (20%)
dataset = Dataset.from_pandas(df[['texto', 'label']])
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# ---------------------------------------------------------
# 2. TOKENIZACIÓN (Convertir texto a números)
# ---------------------------------------------------------
print("🔤 Descargando tokenizador y modelo base (BETO)...")
# Usamos un modelo BERT base en español, muy ligero y preciso
modelo_id = "dccuchile/bert-base-spanish-wwm-uncased"
tokenizer = AutoTokenizer.from_pretrained(modelo_id)

def tokenizar_funcion(ejemplos):
    return tokenizer(ejemplos["texto"], truncation=True, padding="max_length", max_length=64)

tokenized_datasets = dataset.map(tokenizar_funcion, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------------------------------------------------------
# 3. CONFIGURACIÓN DEL MODELO Y MÉTRICAS
# ---------------------------------------------------------
modelo = AutoModelForSequenceClassification.from_pretrained(
    modelo_id, 
    num_labels=len(etiquetas),
    id2label=id2label,
    label2id=label2id
)

# Métrica para ver cómo de bien aprende (Precisión)
metrica = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predicciones = np.argmax(logits, axis=-1)
    return metrica.compute(predictions=predicciones, references=labels)

# ---------------------------------------------------------
# 4. ENTRENAMIENTO (Fine-Tuning)
# ---------------------------------------------------------
argumentos_entrenamiento = TrainingArguments(
    output_dir="./resultados",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5, # 5 pasadas completas por el dataset
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir='./logs',
)

trainer = Trainer(
    model=modelo,
    args=argumentos_entrenamiento,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("🚀 ¡Iniciando el entrenamiento! (Esto puede tardar unos minutos)...")
trainer.train()

# ---------------------------------------------------------
# 5. GUARDAR EL MODELO FINAL ENTRENADO
# ---------------------------------------------------------
ruta_guardado = "./modelo_vr_guardado"
trainer.save_model(ruta_guardado)
tokenizer.save_pretrained(ruta_guardado)
print(f"✅ ¡Entrenamiento completado! Modelo guardado en la carpeta '{ruta_guardado}'")