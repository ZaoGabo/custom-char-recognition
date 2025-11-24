"""
Export CNN v3 to ONNX format for API deployment
"""
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.cnn_model_v3 import CharCNN_v3
import json

def export_v3_to_onnx():
    print("🔄 Exporting CNN v3 to ONNX...")
    
    model_dir = Path('models/cnn_modelo_v3')
    
    # Load config
    config_path = model_dir / 'config_v3.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    num_classes = config['num_classes']
    dropout_rate = config.get('dropout_rate', 0.3)
    
    print(f"  Classes: {num_classes}")
    print(f"  Dropout: {dropout_rate}")
    
    # Create model
    model = CharCNN_v3(num_classes=num_classes, dropout_rate=dropout_rate)
    
    # Load weights
    model_path = model_dir / 'best_model_v3.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print("✅ Model loaded")
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # Export to ONNX
    output_path = model_dir / 'model.onnx'
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Model exported to: {output_path}")
    
    # Test the ONNX model
    import onnxruntime as ort
    import numpy as np
    
    session = ort.InferenceSession(str(output_path))
    test_input = np.random.randn(1, 1, 28, 28).astype(np.float32)
    outputs = session.run(None, {'input': test_input})
    
    print(f"✅ ONNX model tested successfully")
    print(f"   Output shape: {outputs[0].shape}")
    print(f"\n🎉 v3 ready for API deployment!")

if __name__ == "__main__":
    export_v3_to_onnx()
