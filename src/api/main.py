"""
API Principal - Reconocimiento de Caracteres Manuscritos

Este modulo expone los endpoints REST para:
- /api/v2/predict: Prediccion de caracteres desde canvas (modelo v2_finetuned)
- /api/v3/predict: Prediccion de caracteres desde documentos (modelo v3_super)
- /api/v4/predict_text: Reconocimiento de texto completo (TrOCR)

Autor: ZaoGabo
Version: 3.0.0
"""

import sys
import base64
import io
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Cargar variables de entorno antes de cualquier otra importacion
load_dotenv()

# Configurar path del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Importaciones internas del proyecto
from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    TrOCRTextResponse,
    ImageBase64Request
)
from src.label_map import LabelMap
from src.config import CUSTOM_LABELS
from src.utils.logger import app_logger as logger
from src.core.constants import MODELS_DIR


# ---------------------------------------------------------------------------
# Variables globales para modelos
# ---------------------------------------------------------------------------
model_v2 = None      # CNN v2 finetuned - optimizado para canvas
model_v3 = None      # CNN v3 super - optimizado para documentos
model_trocr = None   # TrOCR - reconocimiento de texto completo
label_map = None     # Mapeo de indices a caracteres


# ---------------------------------------------------------------------------
# Ciclo de vida de la aplicacion
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida de la API.
    
    Al iniciar:
        - Carga los tres modelos de prediccion
        - Inicializa el mapa de etiquetas
    
    Al cerrar:
        - Libera recursos (si aplica)
    """
    global model_v2, model_v3, model_trocr, label_map
    
    logger.info("Iniciando carga de modelos...")
    
    # Modelo v2_finetuned (canvas)
    try:
        from src.cnn_predictor_v2_finetuned import CNNPredictor_v2_Finetuned
        v2_path = ROOT_DIR / MODELS_DIR / 'cnn_modelo_v2_finetuned'
        model_v2 = CNNPredictor_v2_Finetuned(model_dir=str(v2_path))
        logger.info("[OK] Modelo v2_finetuned cargado")
    except Exception as e:
        logger.error(f"Error cargando v2_finetuned: {e}")
        model_v2 = None
    
    # Modelo v3_super (documentos)
    try:
        from src.cnn_predictor_v3_super import CNNPredictor_v3_SUPER
        v3_path = ROOT_DIR / MODELS_DIR / 'cnn_modelo_v3_super'
        model_v3 = CNNPredictor_v3_SUPER(model_dir=str(v3_path))
        logger.info("[OK] Modelo v3_super cargado")
    except Exception as e:
        logger.error(f"Error cargando v3_super: {e}")
        model_v3 = None
    
    # Modelo TrOCR (texto completo)
    try:
        from src.trocr_predictor import TrOCRPredictor
        model_trocr = TrOCRPredictor()
        logger.info("[OK] Modelo TrOCR cargado")
    except Exception as e:
        logger.error(f"Error cargando TrOCR: {e}")
        model_trocr = None
    
    # Mapa de etiquetas
    label_map = LabelMap(CUSTOM_LABELS)
    logger.info("Todos los modelos cargados correctamente")
    
    yield
    
    logger.info("Cerrando API...")


# ---------------------------------------------------------------------------
# Configuracion de la aplicacion FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Character Recognition API",
    description="API de reconocimiento de caracteres manuscritos con modelos CNN y TrOCR",
    version="3.0.0",
    lifespan=lifespan
)

# Habilitar CORS para desarrollo local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directorio de archivos estaticos (interfaz web)
PAGE_DIR = ROOT_DIR / "page"


# ---------------------------------------------------------------------------
# Endpoints de archivos estaticos
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """Sirve la pagina principal de seleccion de modelo."""
    index_path = PAGE_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return {"message": "Character Recognition API", "docs": "/docs"}


@app.get("/v2")
@app.get("/v2_finetuned.html")
async def v2_page():
    """Sirve la interfaz de canvas (modelo v2)."""
    return FileResponse(str(PAGE_DIR / "v2_finetuned.html"), media_type="text/html")


@app.get("/v3")
@app.get("/v3_super.html")
async def v3_page():
    """Sirve la interfaz de documentos (modelo v3/TrOCR)."""
    return FileResponse(str(PAGE_DIR / "v3_super.html"), media_type="text/html")


@app.get("/v2_finetuned.js")
async def v2_js():
    """Sirve el JavaScript de la interfaz v2."""
    return FileResponse(str(PAGE_DIR / "v2_finetuned.js"), media_type="application/javascript")


@app.get("/v3_super.js")
async def v3_js():
    """Sirve el JavaScript de la interfaz v3."""
    return FileResponse(str(PAGE_DIR / "v3_super.js"), media_type="application/javascript")


# ---------------------------------------------------------------------------
# Endpoint de salud
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """
    Verifica el estado de la API y sus modelos.
    
    Returns:
        dict: Estado de salud con informacion de cada modelo
    """
    return {
        "status": "healthy",
        "models": {
            "v2_finetuned": "loaded" if model_v2 else "not loaded",
            "v3_super": "loaded" if model_v3 else "not loaded",
            "trocr": "loaded" if model_trocr else "not loaded"
        }
    }


# ---------------------------------------------------------------------------
# Endpoints de prediccion
# ---------------------------------------------------------------------------
@app.post("/api/v2/predict", response_model=PredictionResponse)
async def predict_v2_canvas(request: PredictionRequest):
    """
    Predice un caracter desde una imagen de canvas.
    
    Este endpoint esta optimizado para caracteres dibujados a mano
    en un canvas web. Usa el modelo v2_finetuned.
    
    Args:
        request: Imagen normalizada como array de 784 floats (28x28)
    
    Returns:
        PredictionResponse: Caracter predicho, confianza y top 5 alternativas
    
    Raises:
        HTTPException 503: Si el modelo no esta cargado
        HTTPException 500: Si ocurre un error en la prediccion
    """
    if model_v2 is None:
        raise HTTPException(status_code=503, detail="Modelo v2_finetuned no disponible")
    
    try:
        # Convertir lista a array numpy 28x28
        img_array = np.array(request.image, dtype=np.float32).reshape(28, 28)
        
        # Ejecutar prediccion
        char, confidence, top5_tuples = model_v2.predict(img_array)
        
        # Formatear respuesta
        top5 = [{"character": c, "probability": p} for c, p in top5_tuples]
        result = PredictionResponse(character=char, confidence=confidence, top5=top5)
        
        logger.info(f"v2 Prediccion: {char} ({confidence:.2%})")
        return result
        
    except Exception as e:
        logger.error(f"Error en prediccion v2: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error de prediccion: {str(e)}")


@app.post("/api/v3/predict", response_model=PredictionResponse)
async def predict_v3_document(request: PredictionRequest):
    """
    Predice un caracter desde una imagen de documento.
    
    Este endpoint esta optimizado para caracteres extraidos de
    documentos escaneados. Usa el modelo v3_super.
    
    Args:
        request: Imagen normalizada como array de 784 floats (28x28)
    
    Returns:
        PredictionResponse: Caracter predicho, confianza y top 5 alternativas
    """
    if model_v3 is None:
        raise HTTPException(status_code=503, detail="Modelo v3_super no disponible")
    
    try:
        img_array = np.array(request.image, dtype=np.float32).reshape(28, 28)
        char, confidence, top5_tuples = model_v3.predict(img_array)
        
        top5 = [{"character": c, "probability": p} for c, p in top5_tuples]
        result = PredictionResponse(character=char, confidence=confidence, top5=top5)
        
        logger.info(f"v3 Prediccion: {char} ({confidence:.2%})")
        return result
        
    except Exception as e:
        logger.error(f"Error en prediccion v3: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error de prediccion: {str(e)}")


@app.post("/api/v4/predict_text", response_model=TrOCRTextResponse)
async def predict_text_trocr(request: ImageBase64Request):
    """
    Reconoce texto completo desde una imagen usando TrOCR.
    
    Este endpoint usa un modelo Transformer (TrOCR) para reconocer
    lineas completas de texto manuscrito, no solo caracteres individuales.
    
    Args:
        request: Imagen en formato base64 (puede incluir prefijo data:image/...)
    
    Returns:
        TrOCRTextResponse: Texto reconocido y nivel de confianza
    """
    if model_trocr is None:
        raise HTTPException(status_code=503, detail="Modelo TrOCR no disponible")
    
    try:
        # Extraer datos base64 (remover prefijo si existe)
        img_data = request.image_base64
        if ',' in img_data:
            img_data = img_data.split(',')[1]
        
        # Decodificar y abrir imagen
        img_bytes = base64.b64decode(img_data)
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Ejecutar prediccion
        text, confidence = model_trocr.predict_with_confidence(image)
        
        logger.info(f"TrOCR: '{text}' ({confidence:.2%})")
        return TrOCRTextResponse(text=text, confidence=confidence)
        
    except Exception as e:
        logger.error(f"Error en TrOCR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error TrOCR: {str(e)}")


# ---------------------------------------------------------------------------
# Punto de entrada para desarrollo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
