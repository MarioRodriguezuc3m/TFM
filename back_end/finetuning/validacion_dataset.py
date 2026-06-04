"""
Script de anotación manual para validación del dataset de consultas VR.

Procedimiento:
  1. Carga el dataset original (texto, categoria_original).
  2. Selecciona una muestra ALEATORIA estratificada (N registros por clase).
     - La aleatoriedad cambia en cada ejecución (sin seed fijo), salvo que se
       indique uno con --seed para reproducibilidad.
  3. Presenta cada consulta SIN mostrar la etiqueta original (anotación a ciegas).
  4. El anotador (tú) introduce la categoría que cree correcta.
  5. Al final calcula y muestra:
       - Accuracy global
       - Accuracy por categoría
       - Matriz de confusión
       - Kappa de Cohen (anotador humano vs etiqueta original del dataset)
  6. Guarda los resultados en un CSV y un TXT de métricas con timestamp para poder comparar
     varias rondas de anotación (intra-annotator agreement / test-retest).

Uso:
    python anotar_dataset.py --csv dataset_intent_vr_1050.csv --por_clase 15
    python anotar_dataset.py --csv dataset_intent_vr_1050.csv --total 100
    python anotar_dataset.py --csv dataset_intent_vr_1050.csv --total 100 --seed 42

Requisitos:
    pip install pandas scikit-learn
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)


# =====================================================================
# CONFIGURACIÓN
# =====================================================================

# Se elimina "consulta_general" según lo solicitado
CATEGORIAS = [
    "descripcion_escena",
    "localizacion_objeto",
    "detalle_objeto",
    "objetos_cercanos",
    "navegacion",
    "fuera_dominio",
]

# Atajos numéricos para anotar más rápido (1-6 en lugar de escribir el nombre)
ATAJOS = {str(i + 1): cat for i, cat in enumerate(CATEGORIAS)}

OUTPUT_DIR = Path("./anotaciones")


# =====================================================================
# UTILIDADES
# =====================================================================

def mostrar_menu() -> None:
    """Muestra la leyenda de categorías con sus atajos."""
    print("\n" + "=" * 60)
    print("CATEGORÍAS DISPONIBLES (escribe el número o el nombre):")
    print("=" * 60)
    for num, cat in ATAJOS.items():
        print(f"  [{num}] {cat}")
    print("  [s] saltar este registro")
    print("  [b] volver atrás (re-anotar el registro anterior)")
    print("  [l] listar todas las anotaciones hechas hasta ahora")
    print("  [c N] corregir la anotación del registro número N (ej. 'c 7')")
    print("  [?] mostrar este menú de nuevo")
    print("  [q] salir y guardar lo anotado hasta ahora")
    print("=" * 60)


def normalizar_input(entrada: str) -> str | None:
    """Convierte input del usuario en nombre de categoría válido."""
    entrada = entrada.strip().lower()
    if entrada in ATAJOS:
        return ATAJOS[entrada]
    if entrada in CATEGORIAS:
        return entrada
    # Aceptar también prefijos únicos (p.ej. 'nav' -> 'navegacion')
    coincidencias = [c for c in CATEGORIAS if c.startswith(entrada)]
    if len(coincidencias) == 1:
        return coincidencias[0]
    return None


def obtener_muestra_estratificada(
    df: pd.DataFrame,
    por_clase: int | None,
    total: int | None,
    seed: int | None,
) -> pd.DataFrame:
    """
    Devuelve una muestra estratificada por la columna 'categoria'.

    Si se indica `por_clase`, toma exactamente ese número por categoría.
    Si se indica `total`, reparte equitativamente entre las clases.
    """
    if por_clase is None and total is None:
        por_clase = 15  # default razonable: 15 * 6 = 90 registros

    if por_clase is None:
        # Repartir 'total' entre las clases lo más equitativo posible
        por_clase = max(1, total // len(CATEGORIAS))

    # groupby + sample respeta la cantidad por clase.
    partes = []
    for cat, grupo in df.groupby("categoria"):
        n = min(por_clase, len(grupo))
        partes.append(grupo.sample(n=n, random_state=seed))
    muestra = pd.concat(partes, ignore_index=True)

    # Mezclar el orden para que el anotador no vea bloques de la misma clase
    muestra = muestra.sample(frac=1, random_state=seed).reset_index(drop=True)
    return muestra


# =====================================================================
# BUCLE DE ANOTACIÓN
# =====================================================================

def anotar(muestra: pd.DataFrame) -> pd.DataFrame:
    """
    Itera por la muestra pidiendo al usuario la categoría de cada consulta.
    Devuelve el DataFrame ampliado con la columna 'categoria_anotada'.

    Comandos disponibles en cualquier momento:
      - número 1-6 o nombre de categoría: anota el registro actual
      - 's' : salta el registro actual (queda como None)
      - 'b' : vuelve al registro anterior para re-anotarlo
      - 'l' : lista todas las anotaciones hechas hasta ahora
      - 'c N' : corrige la anotación del registro número N
      - '?' : muestra el menú de nuevo
      - 'q' : sale guardando lo anotado
    """
    total = len(muestra)
    # Inicializamos con None: así un índice cualquiera siempre es accesible
    # tanto para anotar por primera vez como para corregir más tarde.
    anotaciones: list[str | None] = [None] * total

    mostrar_menu()
    print(f"\n🚀 Vas a anotar {total} consultas. ¡Ánimo!\n")

    i = 0  # índice (0-based) del registro actual
    while i < total:
        texto_actual = muestra.iloc[i]["texto"]
        previa = anotaciones[i]
        prefijo_previa = f"  (ya anotada antes como: {previa}) " if previa else ""

        print(f"\n[{i + 1}/{total}] Consulta: {texto_actual!r}")
        if previa:
            print(prefijo_previa.rstrip())

        entrada = input("  Tu categoría: ").strip()
        entrada_lower = entrada.lower()

        # --- Comandos de control --------------------------------------

        if entrada_lower == "q":
            print("\n⚠️  Saliendo. Se guardarán las anotaciones hechas hasta ahora.")
            break

        if entrada_lower in ("?", "h", "help", "ayuda"):
            mostrar_menu()
            continue  # NO avanza el índice

        if entrada_lower == "s":
            print("  ⏭️  Saltado (queda sin anotar).")
            anotaciones[i] = None
            i += 1
            continue

        if entrada_lower == "b":
            if i == 0:
                print("  ⚠️  Ya estás en el primer registro, no se puede volver atrás.")
                continue
            i -= 1
            print(f"  ↩️  Volviendo al registro [{i + 1}/{total}] para re-anotarlo.")
            continue

        if entrada_lower == "l":
            _listar_anotaciones(muestra, anotaciones)
            continue  # NO avanza el índice

        if entrada_lower.startswith("c "):
            # Sintaxis: 'c N' donde N es el número (1-based) del registro a corregir
            partes = entrada_lower.split()
            if len(partes) != 2 or not partes[1].isdigit():
                print("  ❌ Sintaxis: 'c N' donde N es el número del registro (ej. 'c 7').")
                continue
            n = int(partes[1])
            if not (1 <= n <= total):
                print(f"  ❌ El número debe estar entre 1 y {total}.")
                continue
            _corregir_registro(muestra, anotaciones, idx=n - 1)
            continue  # NO avanza el índice; sigues en el registro actual

        # --- Anotación normal -----------------------------------------

        categoria = normalizar_input(entrada)
        if categoria is None:
            print(f"  ❌ Entrada no válida: '{entrada}'. Escribe '?' para ver el menú.")
            continue

        anotaciones[i] = categoria
        print(f"  ✅ Anotado como: {categoria}")
        i += 1

    muestra = muestra.copy()
    muestra["categoria_anotada"] = anotaciones
    return muestra


def _listar_anotaciones(muestra: pd.DataFrame, anotaciones: list[str | None]) -> None:
    """Muestra todas las anotaciones hechas hasta ahora con su número de registro."""
    print("\n" + "-" * 60)
    print("📋 ANOTACIONES HASTA AHORA")
    print("-" * 60)
    hay_alguna = False
    for idx, anotacion in enumerate(anotaciones):
        if anotacion is not None:
            texto = muestra.iloc[idx]["texto"]
            # Truncar texto largo para que la lista sea legible
            texto_corto = texto if len(texto) <= 50 else texto[:47] + "..."
            print(f"  [{idx + 1:3d}] {anotacion:20s} ← {texto_corto!r}")
            hay_alguna = True
    if not hay_alguna:
        print("  (todavía no has anotado nada)")
    print("-" * 60)


def _corregir_registro(
    muestra: pd.DataFrame,
    anotaciones: list[str | None],
    idx: int,
) -> None:
    """Permite re-anotar un registro previo identificado por su índice (0-based)."""
    texto = muestra.iloc[idx]["texto"]
    actual = anotaciones[idx]
    actual_str = actual if actual is not None else "(sin anotar)"

    print(f"\n  ✏️  Corrigiendo registro [{idx + 1}]: {texto!r}")
    print(f"     Anotación actual: {actual_str}")

    while True:
        entrada = input("     Nueva categoría (o 'cancelar' para dejarla como está): ").strip()
        if entrada.lower() in ("cancelar", "cancel", "c", ""):
            print("     ↩️  Corrección cancelada, se mantiene la anotación anterior.")
            return
        nueva = normalizar_input(entrada)
        if nueva is None:
            print(f"     ❌ Entrada no válida: '{entrada}'. Inténtalo de nuevo o escribe 'cancelar'.")
            continue
        anotaciones[idx] = nueva
        print(f"     ✅ Registro [{idx + 1}] actualizado: {actual_str} → {nueva}")
        return


# =====================================================================
# MÉTRICAS Y REPORTE
# =====================================================================

def reportar_metricas(df_anotado: pd.DataFrame, ruta_txt: Path | None = None) -> None:
    """Calcula, muestra y opcionalmente guarda accuracy, kappa, matriz de confusión y reporte."""
    # Filtrar los saltados (None)
    df = df_anotado.dropna(subset=["categoria_anotada"]).copy()
    n_total = len(df_anotado)
    n_anotadas = len(df)
    n_saltadas = n_total - n_anotadas

    if n_anotadas == 0:
        print("\n⚠️  No has anotado ningún registro, no hay métricas que calcular.")
        return

    y_true = df["categoria"].tolist()
    y_pred = df["categoria_anotada"].tolist()

    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, labels=CATEGORIAS)

    # Colección de líneas para guardar en el archivo de texto
    reporte_lineas = []

    def log(msg: str = "") -> None:
        print(msg)
        reporte_lineas.append(msg)

    log("\n" + "=" * 70)
    log("📊 RESULTADOS DE LA ANOTACIÓN")
    log("=" * 70)
    log(f"Registros anotados : {n_anotadas} / {n_total} (saltados: {n_saltadas})")
    log(f"Accuracy global    : {accuracy:.3f}  ({accuracy * 100:.1f}%)")
    log(f"Cohen's kappa      : {kappa:.3f}")
    log("  Interpretación (Landis & Koch, 1977):")
    log("    < 0.20 pobre | 0.21-0.40 aceptable | 0.41-0.60 moderado")
    log("    0.61-0.80 sustancial | > 0.80 casi perfecto")

    # Reporte por categoría
    log("\n" + "-" * 70)
    log("📋 PRECISIÓN / RECALL / F1 POR CATEGORÍA")
    log("-" * 70)
    log(classification_report(y_true, y_pred, labels=CATEGORIAS, zero_division=0))

    # Matriz de confusión
    log("-" * 70)
    log("🔢 MATRIZ DE CONFUSIÓN  (filas = etiqueta original del dataset,")
    log("                         columnas = tu anotación)")
    log("-" * 70)
    cm = confusion_matrix(y_true, y_pred, labels=CATEGORIAS)
    cm_df = pd.DataFrame(cm, index=CATEGORIAS, columns=CATEGORIAS)
    
    with pd.option_context("display.max_columns", None, "display.width", 200):
        log(cm_df.to_string())

    # Detalle de los desacuerdos: útil para discusión cualitativa en el TFM
    desacuerdos = df[df["categoria"] != df["categoria_anotada"]]
    if not desacuerdos.empty:
        log("\n" + "-" * 70)
        log(f"❓ DESACUERDOS ({len(desacuerdos)} casos)")
        log("-" * 70)
        for _, fila in desacuerdos.iterrows():
            log(f"  • {fila['texto']!r}")
            log(f"      dataset → {fila['categoria']}   |   tú → {fila['categoria_anotada']}")

    # Guardar en archivo .txt si se provee una ruta válida
    if ruta_txt is not None:
        try:
            with open(ruta_txt, "w", encoding="utf-8") as f:
                f.write("\n".join(reporte_lineas))
            print(f"\n📊 Métricas guardadas en: {ruta_txt}")
        except Exception as e:
            print(f"\n❌ Error al guardar el archivo de métricas: {e}")


def guardar_resultados(df_anotado: pd.DataFrame, ruta_origen: str) -> Path:
    """Guarda el CSV de anotaciones con timestamp para permitir varias rondas y devuelve la ruta."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_base = Path(ruta_origen).stem
    salida = OUTPUT_DIR / f"anotacion_{nombre_base}_{timestamp}.csv"
    df_anotado.to_csv(salida, index=False, encoding="utf-8")
    print(f"\n💾 Anotaciones guardadas en: {salida}")
    return salida


# =====================================================================
# MAIN
# =====================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Anotación manual de validación sin la categoría consulta_general."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Ruta al CSV original con columnas 'texto' y 'categoria'.",
    )
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument(
        "--por_clase",
        type=int,
        help="Nº de registros a anotar POR CADA categoría (ej. 15).",
    )
    grupo.add_argument(
        "--total",
        type=int,
        help="Nº TOTAL aproximado de registros a anotar (se reparten por igual).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed para reproducir la misma muestra. Por defecto: aleatorio en cada ejecución.",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"❌ No se encuentra el fichero: {args.csv}")
        return 1

    df = pd.read_csv(args.csv)
    if not {"texto", "categoria"}.issubset(df.columns):
        print("❌ El CSV debe tener columnas 'texto' y 'categoria'.")
        return 1

    print(f"📂 Dataset cargado: {len(df)} registros, {df['categoria'].nunique()} clases.")

    muestra = obtener_muestra_estratificada(
        df, por_clase=args.por_clase, total=args.total, seed=args.seed,
    )
    print(f"🎯 Muestra estratificada: {len(muestra)} registros.")
    if args.seed is None:
        print("   (Aleatorio: cada ejecución sin --seed elige una muestra distinta)")
    else:
        print(f"   (Reproducible con seed={args.seed})")

    try:
        df_anotado = anotar(muestra)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrumpido por el usuario (Ctrl+C). Saliendo sin calcular métricas...")
        return 1

    # Guardar resultados en CSV
    ruta_csv = guardar_resultados(df_anotado, args.csv)
    
    # Derivar la ruta del archivo TXT a partir de la ruta del CSV para que coincida el timestamp
    nombre_txt = ruta_csv.name.replace("anotacion_", "metricas_")
    ruta_txt = ruta_csv.with_name(nombre_txt).with_suffix(".txt")
    
    # Calcular, mostrar y guardar métricas
    reportar_metricas(df_anotado, ruta_txt=ruta_txt)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())