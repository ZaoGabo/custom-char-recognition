import sys
import os
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.cnn_model_v3 import CharCNN_v3
from src.data_loader import DataLoader
from src.config import NETWORK_CONFIG, DATA_CONFIG, PATHS, LOGGING_CONFIG, CUSTOM_LABELS
from src.label_map import LabelMap

def train_v3():
    print("🚀 Starting Training for CNN v3 (Deep ResNet-like)")
    
    # Setup paths
    model_dir = Path(PATHS['modelos']) / 'cnn_modelo_v3'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("📂 Loading Data...")
    label_map = LabelMap(CUSTOM_LABELS)
    loader = DataLoader(PATHS['datos_crudos'], label_map)
    loader.cargar_desde_directorio()
    
    X_train_paths, X_val_paths, y_train, y_val = loader.dividir_datos(proporcion_entrenamiento=0.8)
    
    print(f"📊 Training samples: {len(X_train_paths)}")
    print(f"📊 Validation samples: {len(X_val_paths)}")
    
    # 2. Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️ Using device: {device}")
    
    model = CharCNN_v3(
        num_classes=len(CUSTOM_LABELS),
        dropout_rate=NETWORK_CONFIG.get('dropout_rate', 0.3)
    ).to(device)
    
    # 3. Setup Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=NETWORK_CONFIG['tasa_aprendizaje'],
        weight_decay=NETWORK_CONFIG.get('lambda_l2', 1e-5)
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Config
    BATCH_SIZE = NETWORK_CONFIG['tamano_lote']
    EPOCHS = 30 # Fixed for this run, or use config
    IMG_SIZE = tuple(DATA_CONFIG['tamano_imagen'])
    
    best_val_acc = 0.0
    patience_counter = 0
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Generator with Augmentation
        train_gen = loader.generar_lotes(
            X_train_paths, y_train, BATCH_SIZE, IMG_SIZE, augment=True
        )
        
        num_batches = int(np.ceil(len(X_train_paths) / BATCH_SIZE))
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for _ in pbar:
            X_batch, y_batch = next(train_gen)
            # Reshape for PyTorch (B, 1, 28, 28)
            X_tensor = torch.from_numpy(X_batch).view(-1, 1, IMG_SIZE[0], IMG_SIZE[1]).to(device)
            y_tensor = torch.from_numpy(y_batch).long().to(device)
            
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y_tensor.size(0)
            train_correct += predicted.eq(y_tensor).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*train_correct/train_total:.2f}%"})
            
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        val_gen = loader.generar_lotes(
            X_val_paths, y_val, BATCH_SIZE, IMG_SIZE, augment=False
        )
        val_batches = int(np.ceil(len(X_val_paths) / BATCH_SIZE))
        
        with torch.no_grad():
            for _ in range(val_batches):
                X_batch, y_batch = next(val_gen)
                if len(X_batch) == 0: break
                X_tensor = torch.from_numpy(X_batch).view(-1, 1, IMG_SIZE[0], IMG_SIZE[1]).to(device)
                y_tensor = torch.from_numpy(y_batch).long().to(device)
                
                outputs = model(X_tensor)
                _, predicted = outputs.max(1)
                val_total += y_tensor.size(0)
                val_correct += predicted.eq(y_tensor).sum().item()
                
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0.0
        print(f"Validation Accuracy: {val_acc:.2f}%")
        
        # Scheduler step
        scheduler.step(val_acc)
        
        # Save Best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_dir / 'best_model_v3.pth')
            print(f"✅ New best model saved! ({val_acc:.2f}%)")
        else:
            patience_counter += 1
            
        if patience_counter >= NETWORK_CONFIG.get('early_stopping_patience', 10):
            print("🛑 Early stopping triggered.")
            break
            
    # Save Config
    config_save = {
        'num_classes': len(CUSTOM_LABELS),
        'dropout_rate': NETWORK_CONFIG.get('dropout_rate', 0.3),
        'model_version': 'v3',
        'best_val_acc': best_val_acc
    }
    with open(model_dir / 'config_v3.json', 'w') as f:
        json.dump(config_save, f, indent=4)
        
    print(f"✨ Training complete. Best Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_v3()
