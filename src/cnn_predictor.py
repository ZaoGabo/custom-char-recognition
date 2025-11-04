"""
Predictor para modelo CNN
Carga el modelo entrenado y hace predicciones
"""
import torch
import numpy as np
from pathlib import Path
import json

from src.cnn_model import crear_modelo_cnn
from src.label_map import LabelMap
from src.config import CUSTOM_LABELS


class CNNPredictor:
    """Predictor para modelo CNN"""
    
    def __init__(self, model_dir='models/cnn_modelo'):
        """
        Inicializar predictor
        
        Args:
            model_dir: Directorio del modelo entrenado
        """
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cargar configuración
        config_path = self.model_dir / 'config.json'
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Crear label map
        self.label_map = LabelMap(CUSTOM_LABELS)
        
        # Crear modelo
        self.model = crear_modelo_cnn(
            num_classes=self.config['num_classes'],
            dropout_rate=self.config['dropout_rate']
        )
        
        # Cargar pesos
        checkpoint_path = self.model_dir / 'best_model.pth'
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ CNN cargada desde: {model_dir}")
        print(f"   Accuracy de validación: {self.config['best_val_acc']*100:.2f}%")
        print(f"   Device: {self.device}")
    
    def predecir(self, imagen):
        """
        Predecir carácter de una imagen
        
        Args:
            imagen: numpy array (28, 28) normalizado [0, 1]
            
        Returns:
            tuple: (carácter, probabilidad, top_5_predicciones)
        """
        # Asegurar dimensiones correctas
        if imagen.shape != (28, 28):
            raise ValueError(f"La imagen debe ser 28x28, recibido: {imagen.shape}")
        
        # Convertir a tensor: (28, 28) -> (1, 1, 28, 28)
        tensor = torch.from_numpy(imagen).float().unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        # Predicción
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
        
        # Convertir a numpy
        probs_np = probs.cpu().numpy()[0]
        
        # Top 5 predicciones
        top5_indices = np.argsort(probs_np)[::-1][:5]
        top5_probs = probs_np[top5_indices]
        top5_chars = [self.label_map.get_label(idx) for idx in top5_indices]
        
        top5 = list(zip(top5_chars, top5_probs))
        
        # Mejor predicción
        mejor_char = top5[0][0]
        mejor_prob = top5[0][1]
        
        return mejor_char, mejor_prob, top5
    
    def predecir_batch(self, imagenes):
        """
        Predecir lote de imágenes
        
        Args:
            imagenes: numpy array (N, 28, 28)
            
        Returns:
            list: Lista de tuplas (carácter, probabilidad, top_5)
        """
        resultados = []
        for imagen in imagenes:
            resultado = self.predecir(imagen)
            resultados.append(resultado)
        return resultados


def cargar_cnn_predictor(model_dir='models/cnn_modelo'):
    """Factory function para crear predictor"""
    return CNNPredictor(model_dir=model_dir)


if __name__ == '__main__':
    # Test del predictor
    print("=== TEST DEL PREDICTOR CNN ===\n")
    
    # Cargar predictor
    predictor = cargar_cnn_predictor()
    
    # Test con imagen de prueba
    from PIL import Image
    import sys
    
    # Cargar una imagen EMNIST de prueba
    test_image_path = "data/raw/A_upper/emnist_A_upper_00000.png"
    
    if Path(test_image_path).exists():
        imagen_pil = Image.open(test_image_path).convert('L')
        imagen_np = np.array(imagen_pil, dtype=np.float32) / 255.0
        
        print(f"\n🔍 Probando con: {test_image_path}")
        print(f"   Forma: {imagen_np.shape}")
        print(f"   Rango: [{imagen_np.min():.3f}, {imagen_np.max():.3f}]")
        
        # Predicción
        char, prob, top5 = predictor.predecir(imagen_np)
        
        print(f"\n📊 Resultados:")
        print(f"   Predicción: '{char}' con {prob*100:.2f}% de confianza")
        print(f"\n   Top 5:")
        for i, (c, p) in enumerate(top5, 1):
            print(f"      {i}. '{c}': {p*100:.2f}%")
        
        print("\n✅ Predictor funcionando correctamente!")
    else:
        print(f"❌ No se encontró imagen de prueba: {test_image_path}")
