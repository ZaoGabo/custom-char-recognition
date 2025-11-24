import sys
import os
from pathlib import Path
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.api.schemas import PredictionRequest, PredictionResponse, PredictionTextResponse
from src.label_map import LabelMap
from src.config import CUSTOM_LABELS
from src.segmentation import segment_characters
from src.utils.logger import app_logger as logger
from src.core.constants import ModelVersion, MODELS_DIR

# Global variables
model_session = None
label_map = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for API startup and shutdown"""
    global model_session, label_map
    
    # Get model configuration from environment
    model_version = os.getenv("MODEL_VERSION", "v2_finetuned")
    model_path_env = os.getenv("MODEL_PATH")
    
    if model_path_env:
        model_path = ROOT_DIR / model_path_env
    else:
        model_path = ROOT_DIR / MODELS_DIR / f'cnn_modelo_{model_version}' / 'model.onnx'
    
    if not model_path.exists():
        error_msg = f"Model not found at {model_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Loading ONNX model from {model_path}...")
    model_session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    
    label_map = LabelMap(CUSTOM_LABELS)
    logger.success(f"Model {model_version} loaded successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down API...")

app = FastAPI(
    title="Character Recognition API",
    description="API for recognizing handwritten characters using CNN (ONNX)",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {"message": "Character Recognition API is running 🚀", "docs": "/docs"}

@app.get("/health")
async def health():
    if model_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "loaded"}

def _predict_single(img_data: np.ndarray) -> PredictionResponse:
    """Helper to predict a single 28x28 image."""
    # Reshape to (1, 1, 28, 28)
    input_tensor = img_data.reshape(1, 1, 28, 28)
    
    # Inference
    input_name = model_session.get_inputs()[0].name
    output_name = model_session.get_outputs()[0].name
    
    logits = model_session.run([output_name], {input_name: input_tensor})[0]
    
    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = (exp_logits / exp_logits.sum(axis=1))[0]
    
    # Get top prediction
    pred_idx = np.argmax(probs)
    confidence = float(probs[pred_idx])
    character = label_map.get_label(int(pred_idx))
    
    # Get top 5
    top5_indices = np.argsort(probs)[::-1][:5]
    top5 = [
        {"character": label_map.get_label(int(idx)), "probability": float(probs[idx])}
        for idx in top5_indices
    ]
    
    return PredictionResponse(
        character=character,
        confidence=confidence,
        top5=top5
    )

import sys
import os
from pathlib import Path
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.api.schemas import PredictionRequest, PredictionResponse, PredictionTextResponse
from src.label_map import LabelMap
from src.config import CUSTOM_LABELS
from src.segmentation import segment_characters
from src.utils.logger import app_logger as logger
from src.core.constants import ModelVersion, MODELS_DIR

# Global variables
model_session = None
label_map = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for API startup and shutdown"""
    global model_session, label_map
    
    # Get model configuration from environment
    model_version = os.getenv("MODEL_VERSION", "v2_finetuned")
    model_path_env = os.getenv("MODEL_PATH")
    
    if model_path_env:
        model_path = ROOT_DIR / model_path_env
    else:
        model_path = ROOT_DIR / MODELS_DIR / f'cnn_modelo_{model_version}' / 'model.onnx'
    
    if not model_path.exists():
        error_msg = f"Model not found at {model_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Loading ONNX model from {model_path}...")
    model_session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    
    label_map = LabelMap(CUSTOM_LABELS)
    logger.success(f"Model {model_version} loaded successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down API...")

app = FastAPI(
    title="Character Recognition API",
    description="API for recognizing handwritten characters using CNN (ONNX)",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {"message": "Character Recognition API is running 🚀", "docs": "/docs"}

@app.get("/health")
async def health():
    if model_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "loaded"}

def _predict_single(img_data: np.ndarray) -> PredictionResponse:
    """Helper to predict a single 28x28 image."""
    # Reshape to (1, 1, 28, 28)
    input_tensor = img_data.reshape(1, 1, 28, 28)
    
    # Inference
    input_name = model_session.get_inputs()[0].name
    output_name = model_session.get_outputs()[0].name
    
    logits = model_session.run([output_name], {input_name: input_tensor})[0]
    
    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = (exp_logits / exp_logits.sum(axis=1))[0]
    
    # Get top prediction
    pred_idx = np.argmax(probs)
    confidence = float(probs[pred_idx])
    character = label_map.get_label(int(pred_idx))
    
    # Get top 5
    top5_indices = np.argsort(probs)[::-1][:5]
    top5 = [
        {"character": label_map.get_label(int(idx)), "probability": float(probs[idx])}
        for idx in top5_indices
    ]
    
    return PredictionResponse(
        character=character,
        confidence=confidence,
        top5=top5
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_character(request: PredictionRequest):
    """Predict a single character from base64 image"""
    if model_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        logger.debug("Received prediction request")
        
        # Decode & normalize
        img_array = np.frombuffer(
            bytes.fromhex(request.image_data),
            dtype=np.float32
        ).reshape(28, 28)
        
        result = _predict_single(img_array)
        
        logger.info(f"Prediction successful: {result.character} ({result.confidence:.2%})")
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_text", response_model=PredictionTextResponse)
async def predict_text(request: PredictionRequest):
    """Predict full text from image with multiple characters"""
    if model_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        logger.debug("Received text prediction request")
        
        # Decode original image
        import base64
        img_bytes = base64.b64decode(request.image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        
        # Reshape to original image format
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(img_bytes))
        img_np = np.array(img.convert('L'))
        
        # Segment characters
        segments = segment_characters(img_np)
        logger.debug(f"Segmented {len(segments)} characters")
        
        if not segments:
            logger.warning("No characters found in image")
            return PredictionTextResponse(
                text="",
                characters=[],
                confidence=0.0
            )
        
        # Predict each segment
        predictions = []
        for seg_img in segments:
            pred = _predict_single(seg_img)
            predictions.append(pred)
        
        # Build text
        text = "".join([p.character for p in predictions])
        avg_confidence = sum(p.confidence for p in predictions) / len(predictions)
        
        logger.info(f"Text prediction: '{text}' ({avg_confidence:.2%})")
        
        return PredictionTextResponse(
            text=text,
            characters=predictions,
            confidence=avg_confidence
        )
        
    except Exception as e:
        logger.error(f"Text prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
