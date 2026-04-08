from transformers import pipeline

print("🔄 Cargando tu modelo entrenado...")
# Usamos el pipeline estándar de HuggingFace apuntando a TU carpeta
clasificador_vr = pipeline(
    "text-classification", 
    model="./modelo_vr_guardado", 
    tokenizer="./modelo_vr_guardado",
    device=-1 # Usamos CPU, verás que es instantáneo
)
print("✅ Modelo cargado y listo.\n")

# --- ¡PON A PRUEBA TU MODELO AQUÍ! ---
frases_de_prueba =[
    "¿A cuántos metros de distancia está la puerta?",
    "Dime de qué color es la pared de esta sala",
    "Creo que me he perdido, llévame al principio",
    "¿Hay algo justo frente a mí?",
    "No te entiendo bien, ¿puedes repetir?"
]

for frase in frases_de_prueba:
    # Pasamos la frase por nuestro modelo entrenado
    resultado = clasificador_vr(frase)
    
    intencion = resultado[0]['label']
    confianza = resultado[0]['score'] * 100 # Pasado a porcentaje
    
    print("-" * 50)
    print(f"🗣️ Usuario:  '{frase}'")
    print(f"🔀 Intención: {intencion} (Confianza: {confianza:.1f}%)")

print("-" * 50)

# Pruébalo con una variable string interactiva
mifrase = "¿Dónde se encuentra el cofre del tesoro?"
print(f"\nPrueba personalizada: '{mifrase}'")
print("Resultado:", clasificador_vr(mifrase)[0]['label'])