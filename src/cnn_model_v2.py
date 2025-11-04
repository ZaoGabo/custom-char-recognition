"""
Convolutional Neural Network v2 for Character Recognition

Improved architecture: 4 conv blocks with batch normalization and dropout.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CharCNN_v2(nn.Module):
    """
    CNN v2 for character recognition.
    
    Architecture: Conv(32) -> Conv(64) -> Conv(128) -> Conv(256) -> FC(512) -> FC(256) -> FC(128) -> Output
    Input: (batch, 1, 28, 28)
    Output: (batch, num_classes)
    """
    
    def __init__(self, num_classes=62, dropout_rate=0.5):
        super(CharCNN_v2, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn_fc3 = nn.BatchNorm1d(128)
        
        self.fc4 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_light = nn.Dropout(dropout_rate * 0.5)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout_light(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout_light(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout_light(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn_fc3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        
        return x
    
    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        
        return probs
    
    def predict(self, x):
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)


def crear_modelo_cnn_v2(num_classes=62, dropout_rate=0.5):
    return CharCNN_v2(num_classes=num_classes, dropout_rate=dropout_rate)


if __name__ == '__main__':
    print("Testing CNN v2 architecture\n")
    
    model = crear_modelo_cnn_v2()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    batch_size = 4
    x_test = torch.randn(batch_size, 1, 28, 28)
    
    print(f"Input shape: {x_test.shape}")
    
    output = model(x_test)
    print(f"Output shape: {output.shape}")
    print(f"Output logits range: [{output.min():.2f}, {output.max():.2f}]")
    
    probs = model.predict_proba(x_test)
    print(f"\nProbabilities shape: {probs.shape}")
    print(f"Probabilities sum: {probs[0].sum():.6f}")
    
    preds = model.predict(x_test)
    print(f"\nPredicted classes: {preds}")
    
    print("\nCNN v2 architecture test passed.")

