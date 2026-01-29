import os
import base64
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Definimos el modelo de datos que esperamos recibir del Javascript
class Captura(BaseModel):
    imagen: str
    nombre: str

origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Permite GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)

# 1. API para guardar la imagen
@app.post("/api/guardar-captura")
async def guardar_captura(datos: Captura):
    try:
        # Crear carpeta 'capturas' si no existe
        folder = "capturas"
        os.makedirs(folder, exist_ok=True)

        # La imagen viene con un encabezado tipo "data:image/jpeg;base64,..."
        # Necesitamos separar el encabezado de los datos reales
        if "," in datos.imagen:
            header, encoded = datos.imagen.split(",", 1)
        else:
            encoded = datos.imagen

        # Decodificar el base64 a bytes
        image_data = base64.b64decode(encoded)

        # Ruta completa del archivo
        file_path = os.path.join(folder, datos.nombre)

        # Escribir el archivo en disco
        with open(file_path, "wb") as f:
            f.write(image_data)

        print(f"✅ Imagen guardada: {datos.nombre}")
        return {"mensaje": "Imagen guardada correctamente", "archivo": datos.nombre}

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Arrancamos el servidor en el puerto 3000 para coincidir con tu configuración anterior
    print("🚀 Servidor corriendo en http://localhost:3000")
    uvicorn.run(app, host="0.0.0.0", port=3000)