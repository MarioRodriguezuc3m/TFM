# --- Archivo: backend.py ---

import os
import base64
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json

# Importa la clase del pipeline desde el paquete core
from core.query_processor import QueryProcessor
from paths import CURRENT_INPUT, FRONTEND_DIR

app = FastAPI()
origins = ["*"]

# Se crea un solo objeto QueryProcessor cuando el servidor arranca.
# Esto asegura que los modelos se cargan una sola vez.
print("🚀 Iniciando Servidor de Asistencia VR...")
query_processor = QueryProcessor(modelo_vision="qwen2.5vl:latest")
print("-" * 50)

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

@app.post("/api/procesar-consulta")
async def procesar_consulta(datos: Consulta):
    try:
        print("\n" + "=" * 80)
        print(f"➡️  Recibida consulta del usuario: '{datos.texto}'")
        print("=" * 80)

        # 1. Guardar archivos (lógica de I/O se queda en la API)
        folder = str(CURRENT_INPUT)
        os.makedirs(folder, exist_ok=True)

        if "," in datos.imagen:
            _, encoded = datos.imagen.split(",", 1)
        else:
            encoded = datos.imagen
        
        image_data = base64.b64decode(encoded)
        file_path_img = os.path.join(folder, datos.nombre)
        with open(file_path_img, "wb") as f:
            f.write(image_data)

        nombre_json = datos.nombre.replace(".jpg", ".json").replace(".png", ".json")
        file_path_json = os.path.join(folder, nombre_json)
        with open(file_path_json, "w", encoding="utf-8") as f:
            json.dump({"objetos_visibles": datos.objetos_visibles}, f, indent=4)
        
        print(f"✅ Archivos guardados: {datos.nombre}")

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
        
        return {
            "mensaje": "OK",
            "intencion": resultado["intencion"],
            "descripcion": resultado["descripcion"]
        }

    except Exception as e:
        print(f"❌ Error grave en el endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# El mount en "/" debe registrarse después de las rutas /api para no eclipsarlas
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    print("🌐 Servidor disponible en http://localhost:3000")
    print("-" * 50)
    uvicorn.run(app, host="0.0.0.0", port=3000)