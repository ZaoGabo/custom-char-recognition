import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data_loader import DataLoader, _leer_imagen_gris
from src.cnn_predictor_v2_finetuned import cargar_cnn_predictor_v2_finetuned
from src.config import DATA_CONFIG, PATHS

def analyze_errors():
    print("🔍 Starting Error Analysis...")
    
    # 1. Load Model
    print("📦 Loading model...")
    predictor = cargar_cnn_predictor_v2_finetuned()
    
    # 2. Load Data
    print("📂 Loading data...")
    loader = DataLoader(PATHS['datos_crudos'], predictor.label_map)
    loader.cargar_desde_directorio()
    
    # Use validation set for analysis
    _, X_val_paths, _, y_val = loader.dividir_datos(proporcion_entrenamiento=0.8)
    
    if not X_val_paths:
        print("⚠️ No validation data found. Using all data for analysis.")
        X_val_paths = loader.rutas_imagenes
        y_val = loader.etiquetas

    print(f"📊 Analyzing {len(X_val_paths)} samples...")
    
    # 3. Run Predictions
    y_true = []
    y_pred = []
    
    print("🚀 Running inference...")
    for i, path in enumerate(tqdm(X_val_paths)):
        try:
            # Load and preprocess image
            img = _leer_imagen_gris(path, DATA_CONFIG['tamano_imagen'])
            img = img.astype(np.float32) / 255.0
            
            # Predict
            pred_char, _, _ = predictor.predict(img)
            
            # Get true label char
            true_char = predictor.label_map.get_label(y_val[i])
            
            y_true.append(true_char)
            y_pred.append(pred_char)
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    # 4. Compute Confusion Matrix
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 5. Find Top Confused Pairs
    confusions = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    'True': true_label,
                    'Predicted': pred_label,
                    'Count': int(cm[i, j])
                })
    
    # Sort by count descending
    confusions_df = pd.DataFrame(confusions)
    if not confusions_df.empty:
        confusions_df = confusions_df.sort_values('Count', ascending=False)
        
        print("\n🏆 TOP 20 CONFUSIONS:")
        print("-" * 40)
        print(confusions_df.head(20).to_string(index=False))
        print("-" * 40)
        
        # Save to CSV
        output_path = Path("confusion_report.csv")
        confusions_df.to_csv(output_path, index=False)
        print(f"\n💾 Full report saved to {output_path.absolute()}")
        
        # Specific check for I/l/1
        print("\n👀 Specific Check (I vs l vs 1 vs |):")
        tricky_chars = ['I', 'l', '1', '|']
        tricky_df = confusions_df[
            confusions_df['True'].isin(tricky_chars) & 
            confusions_df['Predicted'].isin(tricky_chars)
        ]
        if not tricky_df.empty:
            print(tricky_df.to_string(index=False))
        else:
            print("✅ No confusions found between these characters!")

    else:
        print("✅ Perfect accuracy! No confusions found.")

    # Calculate overall accuracy
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"\n📈 Overall Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    analyze_errors()
