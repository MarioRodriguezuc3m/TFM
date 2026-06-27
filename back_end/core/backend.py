# --- Archivo: backend.py ---

import os
import base64
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json

# Importa la clase del pipeline desde el paquete core
from core.query_processor import QueryProcessor
from paths import FRONTEND_DIR, SESSION_LOGS_DIR


class NoCacheStaticFiles(StaticFiles):
    """StaticFiles que desactiva la caché del navegador para el HTML y el JS.

    Así, al iterar (mismo origen, p. ej. localhost), el navegador NO sirve un
    index.html/script.js viejo de su caché tras editar el código. Los assets
    pesados (modelos GLB, imágenes, sonidos) sí se siguen cacheando."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if path in ("", ".", "index.html") or path.endswith((".html", ".js")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


app = FastAPI()
origins = ["*"]

# Se crea un solo objeto QueryProcessor cuando el servidor arranca.
# Esto asegura que los modelos se cargan una sola vez.
print("🚀 Iniciando Servidor de Asistencia VR...")
query_processor = QueryProcessor(modelo_vision="qwen2.5vl:latest")
print("-" * 50)

# Carpeta de log de esta sesión (una por arranque del servidor) y contador de consultas
SESSION_DIR = SESSION_LOGS_DIR / datetime.now().strftime("sesion_%Y%m%d_%H%M%S")
contador_consultas = 0

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Consulta(BaseModel):
    texto: str
    imagen: str
    nombre: str
    objetos_visibles: list = []
    camara: dict = {}

# Escribe el registro JSON de una consulta en la carpeta de la sesión
def guardar_registro(prefijo: str, registro: dict):
    file_path = SESSION_DIR / f"{prefijo}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(registro, f, ensure_ascii=False, indent=4)

@app.post("/api/procesar-consulta")
async def procesar_consulta(datos: Consulta):
    global contador_consultas
    contador_consultas += 1
    prefijo = f"consulta_{contador_consultas:03d}"
    inicio = time.perf_counter()

    registro = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "consulta": datos.texto,
        "objetos_visibles": datos.objetos_visibles,
        "intencion": None,
        "respuesta": None,
        "latencia_segundos": None,
        "error": None,
    }

    try:
        print("\n" + "=" * 80)
        print(f"➡️  Recibida consulta del usuario: '{datos.texto}'")
        print("=" * 80)

        # 1. Guardar la imagen en el log de la sesión (lógica de I/O se queda en la API)
        os.makedirs(SESSION_DIR, exist_ok=True)

        if "," in datos.imagen:
            _, encoded = datos.imagen.split(",", 1)
        else:
            encoded = datos.imagen

        image_data = base64.b64decode(encoded)
        file_path_img = str(SESSION_DIR / f"{prefijo}.jpg")
        with open(file_path_img, "wb") as f:
            f.write(image_data)

        print(f"✅ Imagen guardada: {file_path_img}")

        # 2. Delegar todo el procesamiento a la clase QueryProcessor
        resultado = query_processor.process(
            texto_usuario=datos.texto,
            ruta_imagen=file_path_img,
            objetos_visibles=datos.objetos_visibles
        )

        print("-" * 80)
        print("📢 DESCRIPCIÓN FINAL (Español):\n")
        print(resultado["descripcion"])
        print("-" * 80)

        # 3. Completar y guardar el registro de la consulta
        registro["intencion"] = resultado["intencion"]
        registro["respuesta"] = resultado["descripcion"]
        registro["latencia_segundos"] = round(time.perf_counter() - inicio, 2)
        guardar_registro(prefijo, registro)

        return {
            "mensaje": "OK",
            "intencion": resultado["intencion"],
            "descripcion": resultado["descripcion"]
        }

    except Exception as e:
        print(f"❌ Error grave en el endpoint: {e}")
        # Las consultas fallidas también se registran, son datos útiles del estudio
        registro["error"] = str(e)
        registro["latencia_segundos"] = round(time.perf_counter() - inicio, 2)
        try:
            os.makedirs(SESSION_DIR, exist_ok=True)
            guardar_registro(prefijo, registro)
        except Exception as log_err:
            print(f"⚠️ No se pudo guardar el registro de la consulta fallida: {log_err}")
        raise HTTPException(status_code=500, detail=str(e))

# El mount en "/" debe registrarse después de las rutas /api para no eclipsarlas
app.mount("/", NoCacheStaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    print("🌐 Servidor disponible en http://localhost:3000")
    print("-" * 50)
    uvicorn.run(app, host="0.0.0.0", port=3000)