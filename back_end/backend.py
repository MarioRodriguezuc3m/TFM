import os
import base64
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import ollama

app = FastAPI()
origins = ["*"] 

MODELO_OLLAMA = "qwen2.5vl:latest"

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Captura(BaseModel):
    imagen: str
    nombre: str
    objetos_visibles: list = [] 
    camara: dict = {}

def consultar_ia(ruta_imagen, metadatos_lista):
    """
    Consulta optimizada - prioriza la imagen sobre los metadatos
    """
    print(f"🧠 Analizando escena con {MODELO_OLLAMA}...")
    
    # Convertimos la lista de objetos a texto para el prompt
    contexto_json = json.dumps(metadatos_lista, indent=2, ensure_ascii=False)
    
    prompt = f"""Eres un asistente de accesibilidad enfocado en la descripción accesible de escenar para personas ciegas que usan realidad virtual.

Tu tarea es describir lo que ves en la IMAGEN usando con el objetivo de que una persona ciega inmersa en la escena pueda interpretar la escena con claridad.

INFORMACIÓN AUXILIAR (úsala para complementar lo que ves):
{contexto_json}

Nota: Los objetos tienen "posicion_relativa" al usuario en formato (x, y, z):
- X: negativo = izquierda, positivo = derecha
- Y: negativo = abajo, positivo = arriba  
- Z: negativo = delante, positivo = detrás

Algunos objetos incluyen "objetos_contenidos" que son sub-elementos dentro de ellos.
REGLAS CRÍTICAS:

1. USA LOS OBJETOS DETECTADOS COMO BASE
   - Los objetos que aparecen en el JSON son inequívocamente confiables, están realmente en la escena
   - MENCIONA SOLO los objetos que aparecen en el JSON
   - NO inventes ni añadas objetos que no estén en la lista
   - NO confundas objetos - usa exactamente las etiquetas del JSON

2. USA LA IMAGEN SOLO PARA DETALLES VISUALES. Añade detalles que no estén presentes en el JSON, si transmiten información relevante para describir la escena a la persona ciega:
   - Añade colores específicos que veas en la escena : "marrón oscuro", "verde brillante", "azul celeste"
   - Describe texturas: "madera envejecida", "metal oxidado"
   - Menciona iluminación: "bien iluminado", "sombras suaves", "luz brillante"
   - Describe el ambiente general: cielo, terreno, atmósfera

3. NUNCA menciones:
   - Términos como:  "videojuego", "escenario virtual", "escena de juego"
   - El usuario YA SABE que está en inmerso VR, no hace falta recordárselo

4. Orientación espacial clara (basada en posicion_relativa):
   - X negativo grande (< -5): "bastante a tu izquierda"
   - X negativo pequeño (-5 a -1): "a tu izquierda"
   - X casi cero (-1 a 1): "frente a ti" o "delante de ti"
   - X positivo pequeño (1 a 5): "a tu derecha"
   - X positivo grande (> 5): "bastante a tu derecha"
   
   - Z negativo: "delante", más negativo = "más cerca"
   - Z positivo: "detrás"
   
   - Usa expresiones como "muy cerca", "cerca", "a media distancia", "lejos", "al fondo"
   - NO uses números ni metros exactos

5. Estructura de la descripción:
   - Primera frase: Contexto general (dónde estás, ambiente)
   - Segunda frase: Objetos principales más cercanos
   - Tercera frase: Objetos secundarios o más lejanos
   - Cuarta frase (opcional): Detalles del entorno general
   - Máximo 4 oraciones, lenguaje natural y fluido

6. ORDEN de prioridad en la descripción:
   - Prioriza objetos más cercanos al usuario
   - Luego por relevancia o tamaño
   - Menciona la posición de cada objeto (izquierda/derecha/frente)

7. Tono: Descriptivo, directo y útil. Recuerda que estás generando una descripción para una persona ciega."""

    try:
        response = ollama.chat(
            model=MODELO_OLLAMA,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [ruta_imagen]
            }]
        )
        return response['message']['content']
        
    except Exception as e:
        print(f"⚠️ Error en Ollama: {e}")
        return "Error al generar la descripción."

@app.post("/api/guardar-captura")
async def guardar_captura(datos: Captura):
    try:
        folder = "current_input"
        os.makedirs(folder, exist_ok=True)

        # 1. Guardar la imagen
        if "," in datos.imagen:
            header, encoded = datos.imagen.split(",", 1)
        else:
            encoded = datos.imagen
        
        image_data = base64.b64decode(encoded)
        file_path_img = os.path.join(folder, datos.nombre)
        
        with open(file_path_img, "wb") as f:
            f.write(image_data)

        # 2. Guardar los metadatos completos de la escena
        nombre_json = datos.nombre.replace(".jpg", ".json").replace(".png", ".json")
        file_path_json = os.path.join(folder, nombre_json)

        with open(file_path_json, "w", encoding="utf-8") as f:
            json.dump({
                "objetos_visibles": datos.objetos_visibles 
            }, f, indent=4, ensure_ascii=False)

        print(f"✅ Guardado: {datos.nombre} + JSON con info de cámara")

        # Generar descripción optimizada
        descripcion_generada = consultar_ia(file_path_img, datos.objetos_visibles)

        print("-" * 80)
        print("📢 DESCRIPCIÓN DE AUDIO:\n")
        print(descripcion_generada)
        print("-" * 80)
        
        return {
            "mensaje": "OK",
            "descripcion": descripcion_generada
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Endpoint para verificar que el servidor está funcionando"""
    return {"status": "ok", "modelo": MODELO_OLLAMA}

if __name__ == "__main__":
    print("🚀 Servidor de Asistencia VR para Ciegos")
    print("📍 http://localhost:3000")
    print(f"🤖 Modelo: {MODELO_OLLAMA}")
    print("-" * 50)
    uvicorn.run(app, host="0.0.0.0", port=3000)