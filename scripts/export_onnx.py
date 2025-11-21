"""
Script para exportar el modelo PyTorch entrenado a formato ONNX.
"""
import sys
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import json

# Añadir directorio raíz al path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.cnn_model_v2 import CharCNN_v2

def export_to_onnx():
    print("🚀 Iniciando exportación a ONNX...")
    
    # Rutas
    model_dir = ROOT_DIR / 'models' / 'cnn_modelo_v2_finetuned'
    weights_path = model_dir / 'best_model_finetuned.pth'
    output_path = model_dir / 'model.onnx'
    
    if not weights_path.exists():
        print(f"❌ Error: No se encontraron los pesos en {weights_path}")
        return
    
    # 1. Cargar checkpoint para inferir configuración
    print("⚖️ Cargando checkpoint...")
    device = torch.device('cpu')
    checkpoint = torch.load(weights_path, map_location=device)
    
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    
    # Inferir num_classes de la última capa (fc4.weight)
    if 'fc4.weight' in state_dict:
        num_classes = state_dict['fc4.weight'].shape[0]
        print(f"🧠 Configuración inferida: num_classes={num_classes}")
    else:
        print("⚠️ No se pudo inferir num_classes, usando valor por defecto 62")
        num_classes = 62
        
    dropout_rate = 0.5 # Valor seguro por defecto
    
    # 2. Inicializar modelo
    print("🔨 Inicializando modelo...")
    model = CharCNN_v2(num_classes=num_classes, dropout_rate=dropout_rate)
    
    # 3. Cargar pesos
    model.load_state_dict(state_dict)
    model.eval()
    
    # 4. Crear input dummy
    dummy_input = torch.randn(1, 1, 28, 28, requires_grad=True)
    
    # 5. Exportar
    print(f"📦 Exportando a {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("✅ Exportación completada.")
    
    # 6. Verificación
    verify_onnx(str(output_path), model, dummy_input)

def verify_onnx(onnx_path, torch_model, dummy_input):
    print("\n🔍 Verificando modelo ONNX...")
    
    # Verificar estructura
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✅ Estructura del modelo ONNX válida.")
    
    # Comparar salidas
    print("🧪 Comparando inferencia PyTorch vs ONNX Runtime...")
    
    # PyTorch output
    with torch.no_grad():
        torch_out = torch_model(dummy_input)
    
    # ONNX Runtime output
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # Comparar
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("✅ ¡Las salidas coinciden! El modelo exportado es fiel al original.")
    print(f"🎉 Modelo listo en: {onnx_path}")

if __name__ == '__main__':
    try:
        export_to_onnx()
    except Exception as e:
        print(f"\n❌ Error fatal durante la exportación: {e}")
        sys.exit(1)
