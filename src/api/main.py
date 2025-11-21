import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

# Add root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.api.schemas import PredictionRequest, PredictionResponse, PredictionTextResponse
from src.label_map import LabelMap
from src.config import CUSTOM_LABELS
from src.segmentation import segment_characters

# Global variables
model_session = None
label_map = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model_session, label_map
    
    model_path = ROOT_DIR / 'models' / 'cnn_modelo_v2_finetuned' / 'model.onnx'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"🚀 Loading ONNX model from {model_path}...")
    model_session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    
    label_map = LabelMap(CUSTOM_LABELS)
    print("✅ Model loaded successfully")
    
    yield
    
    # Cleanup
    print("🛑 Shutting down...")

app = FastAPI(
    title="Character Recognition API",
    description="API for recognizing handwritten characters using CNN v2 (ONNX)",
    version="1.1.0",
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
async def predict(request: PredictionRequest):
    if model_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess input
        img_data = np.array(request.image, dtype=np.float32)
        
        # Handle shapes
        if img_data.size != 784:
             raise HTTPException(status_code=400, detail=f"Invalid image size. Expected 784, got {img_data.size}")
             
        return _predict_single(img_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_text", response_model=PredictionTextResponse)
async def predict_text(request: PredictionRequest):
    """
    Predict text from an image containing multiple characters.
    Uses segmentation to split characters and predicts each one.
    """
    if model_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        # 1. Load image
        img_data = np.array(request.image, dtype=np.float32)
        
        shape = None
        if request.width and request.height:
            shape = (request.height, request.width)
        
        # 2. Segment
        segments = segment_characters(img_data, shape=shape)
        
        if not segments:
            return PredictionTextResponse(text="", segments=[])
            
        # 3. Predict each segment
        results = []
        text_parts = []
        
        for segment in segments:
            pred = _predict_single(segment)
            results.append(pred)
            text_parts.append(pred.character)
            
        return PredictionTextResponse(
            text="".join(text_parts),
            segments=results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
