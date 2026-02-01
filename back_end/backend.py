import os
import base64
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json

app = FastAPI()

origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Permite GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)

class Captura(BaseModel):
    imagen: str
    nombre: str
    metadatos: list = []

@app.post("/api/guardar-captura")
async def guardar_captura(datos: Captura):
    try:
        folder = "current_input"
        os.makedirs(folder, exist_ok=True)

        # 1. Se guarda la imagen
        if "," in datos.imagen:
            header, encoded = datos.imagen.split(",", 1)
        else:
            encoded = datos.imagen
        
        image_data = base64.b64decode(encoded)
        file_path_img = os.path.join(folder, datos.nombre)
        
        with open(file_path_img, "wb") as f:
            f.write(image_data)

        # 2. Se guardan los metadatos de la escena en un archivo JSON
        nombre_json = datos.nombre.replace(".jpg", ".json").replace(".png", ".json")
        file_path_json = os.path.join(folder, nombre_json)

        # Guardamos la lista de objetos en un archivo de texto
        with open(file_path_json, "w", encoding="utf-8") as f:
            json.dump({
                "imagen": datos.nombre,
                "objetos_visibles": datos.metadatos
            }, f, indent=4, ensure_ascii=False)

        print(f"✅ Guardado: {datos.nombre} + JSON descriptivo")
        return {"mensaje": "OK"}

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Arrancamos el servidor en el puerto 3000 para coincidir con tu configuración anterior
    print("🚀 Servidor corriendo en http://localhost:3000")
    uvicorn.run(app, host="0.0.0.0", port=3000)