import os
import base64
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import ollama
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()
origins = ["*"] 

MODELO_OLLAMA = "qwen2.5vl:latest"

# Initialize Helsinki-NLP translator (loaded once at startup)
print("🔄 Loading translation model Helsinki-NLP/opus-mt-en-es...")
model_checkpoint = "Helsinki-NLP/opus-mt-en-es"

# Cargamos el tokenizador y el modelo de forma explícita
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

translator = pipeline(
    "translation_en_to_es",           # Usamos el nombre de tarea genérico
    model=model,             # Pasamos el objeto del modelo
    tokenizer=tokenizer,     # Pasamos el objeto del tokenizador
    device=0                # IMPORTANTE: -1 es CPU, 0 es GPU
)
print("✅ Translation model loaded successfully")

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

def traducir_a_espanol(texto_ingles):
    """
    Translate English text to Spanish using Helsinki-NLP model
    Fast and accurate translation (~50-200ms)
    """
    try:
        # Split long texts into chunks if needed (max 512 tokens per chunk)
        if len(texto_ingles) > 2000:
            # Split by sentences for long texts
            sentences = texto_ingles.split('. ')
            translated_sentences = []
            
            for sentence in sentences:
                if sentence.strip():
                    result = translator(sentence + '.', max_length=512)
                    translated_sentences.append(result[0]['translation_text'])
            
            return ' '.join(translated_sentences)
        else:
            # Translate directly for short texts
            result = translator(texto_ingles, max_length=512)
            return result[0]['translation_text']
            
    except Exception as e:
        print(f"⚠️ Translation error: {e}")
        print("📝 Returning original text as fallback")
        return texto_ingles

def consultar_ia(ruta_imagen, metadatos_lista):
    """
    Optimized query - prioritizes image over metadata
    Generates description in English, then translates to Spanish
    """
    print(f"🧠 Analyzing scene with {MODELO_OLLAMA}...")
    
    # Convert object list to text for prompt
    contexto_json = json.dumps(metadatos_lista, indent=2, ensure_ascii=False)
    
    # PROMPT IN ENGLISH (better model performance)
    prompt = f"""You are an accessibility assistant focused on describing VR scenes for blind users.

Your task is to describe what you see in the IMAGE so that a blind person immersed in the scene can clearly understand it.

AUXILIARY INFORMATION (use to complement what you see):
{contexto_json}

Note: Objects have "relative_position" to the user in (x, y, z) format:
- X: negative = left, positive = right
- Y: negative = down, positive = up  
- Z: negative = in front, positive = behind

Some objects include "contained_objects" which are sub-elements within them.

CRITICAL RULES:

1. USE DETECTED OBJECTS AS YOUR BASE
   - Objects appearing in the JSON are unequivocally reliable, they are really in the scene
   - MENTION ONLY objects that appear in the JSON
   - DO NOT invent or add objects not in the list
   - DO NOT confuse objects - use exactly the labels from the JSON

2. USE THE IMAGE ONLY FOR VISUAL DETAILS. Add details not present in the JSON, if they convey relevant information to describe the scene to the blind person:
   - Add specific colors you see in the scene: "dark brown", "bright green", "sky blue"
   - Describe textures: "aged wood", "rusty metal"
   - Mention lighting: "well lit", "soft shadows", "bright light"
   - Describe the general environment: sky, terrain, atmosphere

3. NEVER mention:
   - Terms like: "video game", "virtual scenario", "game scene"
   - The user ALREADY KNOWS they are immersed in VR, no need to remind them

4. Clear spatial orientation (based on relative_position):
   - Large negative X (< -5): "well to your left"
   - Small negative X (-5 to -1): "to your left"
   - Nearly zero X (-1 to 1): "in front of you" or "ahead of you"
   - Small positive X (1 to 5): "to your right"
   - Large positive X (> 5): "well to your right"
   
   - Negative Z: "in front", more negative = "closer"
   - Positive Z: "behind"
   
   - Use expressions like "very close", "close", "at medium distance", "far", "in the background"
   - DO NOT use numbers or exact meters

5. Description structure:
   - First sentence: General context (where you are, atmosphere)
   - Second sentence: Main closest objects
   - Third sentence: Secondary or more distant objects
   - Fourth sentence (optional): General environment details
   - Maximum 4 sentences, natural and fluid language

6. Priority ORDER in description:
   - Prioritize objects closest to the user
   - Then by relevance or size
   - Mention the position of each object (left/right/front)

7. Tone: Descriptive, direct and useful. Remember you are generating a description for a blind person.

CRITICAL: Output your description in ENGLISH. The translation to Spanish will be done automatically."""

    try:
        # Generate description in English
        print("🎨 Generating English description...")
        response = ollama.chat(
            model=MODELO_OLLAMA,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [ruta_imagen]
            }],
            options={
                'temperature': 0.2,
                'top_p': 0.9,
                'top_k': 40,
                'repeat_penalty': 1.1,
            }
        )
        
        descripcion_ingles = response['message']['content']
        print("📝 English description generated")
        print(f"   Length: {len(descripcion_ingles)} characters")
        
        # Translate to Spanish using Helsinki-NLP
        print("🌐 Translating to Spanish with Helsinki-NLP...")
        descripcion_espanol = traducir_a_espanol(descripcion_ingles)
        print("✅ Translation complete")
        
        return descripcion_espanol
        
    except Exception as e:
        print(f"⚠️ Error in Ollama: {e}")
        return "Error al generar la descripción."

@app.post("/api/guardar-captura")
async def guardar_captura(datos: Captura):
    try:
        folder = "current_input"
        os.makedirs(folder, exist_ok=True)

        # 1. Save the image
        if "," in datos.imagen:
            header, encoded = datos.imagen.split(",", 1)
        else:
            encoded = datos.imagen
        
        image_data = base64.b64decode(encoded)
        file_path_img = os.path.join(folder, datos.nombre)
        
        with open(file_path_img, "wb") as f:
            f.write(image_data)

        # 2. Save complete scene metadata
        nombre_json = datos.nombre.replace(".jpg", ".json").replace(".png", ".json")
        file_path_json = os.path.join(folder, nombre_json)

        with open(file_path_json, "w", encoding="utf-8") as f:
            json.dump({
                "objetos_visibles": datos.objetos_visibles 
            }, f, indent=4, ensure_ascii=False)

        print(f"✅ Saved: {datos.nombre} + JSON with camera info")

        # Generate optimized description (English -> Spanish)
        descripcion_generada = consultar_ia(file_path_img, datos.objetos_visibles)

        print("-" * 80)
        print("📢 AUDIO DESCRIPTION (Spanish):\n")
        print(descripcion_generada)
        print("-" * 80)
        
        return {
            "mensaje": "OK",
            "descripcion": descripcion_generada
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 VR Assistance Server for Blind Users")
    print("🌐 http://localhost:3000")
    print(f"🤖 Vision Model: {MODELO_OLLAMA}")
    print(f"🌍 Translation: Helsinki-NLP/opus-mt-en-es")
    print("-" * 50)
    uvicorn.run(app, host="0.0.0.0", port=3000)