"""
Smoke test del clasificador de intenciones fine-tuneado (7 clases).

Cubre:
  - Las 6 clases originales con frases no vistas en entrenamiento.
  - La nueva clase 'fuera_dominio' (chistes, aritmética, saludos, etc.).
  - Casos borderline con confianza baja para comprobar el umbral OOD
    aplicado en query_processor.py.

Ejecutar con:
    python test_finetuning.py
"""

from transformers import pipeline

# Debe coincidir con CONFIDENCE_THRESHOLD en query_processor.py
CONFIDENCE_THRESHOLD = 0.55
OOD_LABEL = "fuera_dominio"

print("🔄 Cargando tu modelo entrenado...")
clasificador_vr = pipeline(
    "text-classification",
    model="./modelo_vr_guardado",
    tokenizer="./modelo_vr_guardado",
    device=-1,  # CPU es más que suficiente para BETO en inferencia
)
print("✅ Modelo cargado y listo.\n")


# -------------------------------------------------------------------
# Frases de prueba organizadas por la etiqueta que esperamos obtener
# -------------------------------------------------------------------
frases_de_prueba = [
    # --- Clases del dominio VR ---
    ("¿A cuántos metros de distancia está la puerta?", "localizacion_objeto"),
    ("Dime de qué color es la pared de esta sala",    "detalle_objeto"),
    ("Creo que me he perdido, llévame al principio",  "navegacion"),
    ("¿Hay algo justo frente a mí?",                  "objetos_cercanos"),
    ("No te entiendo bien, ¿puedes repetir?",         "consulta_general"),
    ("Descríbeme esta habitación, por favor",         "descripcion_escena"),

    # --- Nueva clase: fuera de dominio ---
    ("Cuéntame un chiste",                            "fuera_dominio"),
    ("¿Cuánto es siete por ocho?",                    "fuera_dominio"),
    ("¿Qué tiempo hace hoy en Madrid?",               "fuera_dominio"),
    ("Hola, ¿cómo estás?",                            "fuera_dominio"),
    ("Recomiéndame una película",                     "fuera_dominio"),
    ("¿Quién es el presidente de España?",            "fuera_dominio"),

    # --- Borderline / ambiguas (sin etiqueta esperada fija) ---
    ("¿Puedo?",                                       None),
    ("Eso",                                           None),
    ("Hazlo",                                         None),
]


def aplicar_umbral(etiqueta: str, confianza: float) -> str:
    """Replica la lógica de query_processor.py: si la confianza es baja,
    reetiquetamos como fuera_dominio aunque la clase predicha sea válida."""
    if confianza < CONFIDENCE_THRESHOLD and etiqueta != OOD_LABEL:
        return OOD_LABEL
    return etiqueta


aciertos = 0
total_con_esperada = 0

for frase, esperada in frases_de_prueba:
    resultado = clasificador_vr(frase)[0]
    etiqueta_raw = resultado["label"]
    confianza = resultado["score"]
    etiqueta_final = aplicar_umbral(etiqueta_raw, confianza)

    print("-" * 70)
    print(f"🗣️  Usuario:        '{frase}'")
    print(f"🔀 Intención raw:  {etiqueta_raw}  (confianza {confianza*100:.1f}%)")
    if etiqueta_final != etiqueta_raw:
        print(f"🛑 Post-umbral:    {etiqueta_final}  (confianza < {CONFIDENCE_THRESHOLD})")
    if esperada is not None:
        total_con_esperada += 1
        ok = "✅" if etiqueta_final == esperada else "❌"
        if etiqueta_final == esperada:
            aciertos += 1
        print(f"🎯 Esperado:       {esperada}   {ok}")

print("-" * 70)
if total_con_esperada > 0:
    print(
        f"\n📊 Resumen: {aciertos}/{total_con_esperada} aciertos "
        f"({100*aciertos/total_con_esperada:.1f} %) sobre las frases con "
        f"etiqueta esperada."
    )

# --- Prueba personalizada ---
mifrase = "¿Dónde se encuentra el cofre del tesoro?"
resultado = clasificador_vr(mifrase)[0]
etiqueta_final = aplicar_umbral(resultado["label"], resultado["score"])
print(f"\nPrueba personalizada: '{mifrase}'")
print(f"Resultado: {etiqueta_final}  (raw={resultado['label']}, "
      f"score={resultado['score']:.2f})")