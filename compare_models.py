import torch
import onnxruntime as ort
import numpy as np
from src.cnn_predictor_v2_finetuned import cargar_cnn_predictor_v2_finetuned
from pathlib import Path

def compare_models():
    print("⚖️ Comparing PyTorch vs ONNX models...")
    
    # 1. Load PyTorch Model (using robust loader)
    try:
        predictor = cargar_cnn_predictor_v2_finetuned()
        model = predictor.model
        device = predictor.device
    except Exception as e:
        print(f"❌ Failed to load PyTorch model: {e}")
        return

    # 2. Load ONNX Model
    model_dir = Path('models/cnn_modelo_v2_finetuned')
    onnx_path = model_dir / 'model.onnx'
    if not onnx_path.exists():
        print(f"❌ ONNX model not found at {onnx_path}")
        return
        
    ort_session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    
    # 3. Generate Random Input
    # Shape: (1, 1, 28, 28)
    dummy_input = np.random.rand(1, 1, 28, 28).astype(np.float32)
    
    # 4. PyTorch Inference
    tensor = torch.from_numpy(dummy_input).to(device)
    with torch.no_grad():
        torch_out = model(tensor)
        torch_probs = torch.softmax(torch_out, dim=1).cpu().numpy()
        
    # 5. ONNX Inference
    input_name = ort_session.get_inputs()[0].name
    onnx_out = ort_session.run(None, {input_name: dummy_input})[0]
    
    # Softmax on ONNX logits
    exp_logits = np.exp(onnx_out - np.max(onnx_out))
    onnx_probs = exp_logits / exp_logits.sum(axis=1)
    
    # 6. Compare
    diff = np.abs(torch_probs - onnx_probs)
    max_diff = np.max(diff)
    
    print(f"\n📊 Max difference in probabilities: {max_diff:.8f}")
    
    if max_diff < 1e-5:
        print("✅ Models are effectively identical.")
    else:
        print("⚠️ Significant difference detected!")
        print(f"PyTorch top: {np.argmax(torch_probs)} ({np.max(torch_probs):.4f})")
        print(f"ONNX top:    {np.argmax(onnx_probs)} ({np.max(onnx_probs):.4f})")

if __name__ == "__main__":
    compare_models()
