"""
Red Neuronal Convolucional (CNN) para Reconocimiento de Caracteres
Usando PyTorch para mejor rendimiento y features automáticas
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    """
    CNN para reconocimiento de 62 caracteres (0-9, A-Z, a-z)
    
    Arquitectura optimizada:
    - 3 bloques convolucionales con BatchNorm y MaxPooling
    - 3 capas fully connected con Dropout
    - ReLU activations
    - Softmax en output
    
    Input: (batch, 1, 28, 28) - imágenes en escala de grises
    Output: (batch, 62) - probabilidades por clase
    """
    
    def __init__(self, num_classes=62, dropout_rate=0.5):
        super(CharCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # === BLOQUE CONVOLUCIONAL 1 ===
        # Input: 1x28x28 -> Output: 32x28x28 -> MaxPool: 32x14x14
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1  # Mantiene tamaño
        )
        self.bn1 = nn.BatchNorm2d(32)
        
        # === BLOQUE CONVOLUCIONAL 2 ===
        # Input: 32x14x14 -> Output: 64x14x14 -> MaxPool: 64x7x7
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        
        # === BLOQUE CONVOLUCIONAL 3 ===
        # Input: 64x7x7 -> Output: 128x7x7 -> MaxPool: 128x3x3
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)
        
        # === CAPAS FULLY CONNECTED ===
        # Después de 3 MaxPools de 2x2: 28 -> 14 -> 7 -> 3
        # Flatten: 128 * 3 * 3 = 1152
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout (regularización)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Tensor (batch, 1, 28, 28)
            
        Returns:
            Tensor (batch, num_classes) - logits (sin softmax para CrossEntropyLoss)
        """
        # Conv Block 1
        x = self.conv1(x)           # -> (batch, 32, 28, 28)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      # -> (batch, 32, 14, 14)
        
        # Conv Block 2
        x = self.conv2(x)           # -> (batch, 64, 14, 14)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      # -> (batch, 64, 7, 7)
        
        # Conv Block 3
        x = self.conv3(x)           # -> (batch, 128, 7, 7)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      # -> (batch, 128, 3, 3)
        
        # Flatten
        x = x.view(x.size(0), -1)   # -> (batch, 1152)
        
        # FC Block 1
        x = self.fc1(x)             # -> (batch, 256)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # FC Block 2
        x = self.fc2(x)             # -> (batch, 128)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output (sin softmax - PyTorch CrossEntropyLoss lo incluye)
        x = self.fc3(x)             # -> (batch, num_classes)
        
        return x
    
    def predict_proba(self, x):
        """
        Predecir probabilidades (con softmax)
        
        Args:
            x: Tensor (batch, 1, 28, 28) o numpy array
            
        Returns:
            Tensor (batch, num_classes) - probabilidades [0, 1]
        """
        self.eval()  # Modo evaluación (desactiva dropout)
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        
        return probs
    
    def predict(self, x):
        """
        Predecir clases
        
        Args:
            x: Tensor (batch, 1, 28, 28) o numpy array
            
        Returns:
            Tensor (batch,) - índices de clases predichas
        """
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)


# Para compatibilidad con NumPy
import numpy as np


def crear_modelo_cnn(num_classes=62, dropout_rate=0.5):
    """Factory function para crear el modelo"""
    model = CharCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    return model


if __name__ == '__main__':
    # Test del modelo
    print("=== TEST DE ARQUITECTURA CNN ===\n")
    
    model = crear_modelo_cnn()
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total de parámetros: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}\n")
    
    # Test forward pass
    batch_size = 4
    x_test = torch.randn(batch_size, 1, 28, 28)
    
    print(f"Input shape: {x_test.shape}")
    
    # Forward
    output = model(x_test)
    print(f"Output shape: {output.shape}")
    print(f"Output logits range: [{output.min():.2f}, {output.max():.2f}]")
    
    # Predict proba
    probs = model.predict_proba(x_test)
    print(f"\nProbabilities shape: {probs.shape}")
    print(f"Probabilities sum (should be ~1.0): {probs[0].sum():.6f}")
    print(f"Probabilities range: [{probs.min():.6f}, {probs.max():.6f}]")
    
    # Predict class
    preds = model.predict(x_test)
    print(f"\nPredicted classes: {preds}")
    
    print("\n✅ Arquitectura CNN funcionando correctamente!")
