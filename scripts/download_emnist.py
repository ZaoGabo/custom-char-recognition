import os
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torchvision
from torchvision.datasets import EMNIST
import torch

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'

def get_emnist_mapping():
    """
    EMNIST ByClass mapping:
    0-9: Digits
    10-35: A-Z
    36-61: a-z
    """
    mapping = {}
    
    # Digits 0-9
    for i in range(10):
        mapping[i] = f"{i}_digit"
        
    # Uppercase A-Z (10-35)
    for i in range(26):
        char = chr(ord('A') + i)
        mapping[10 + i] = f"{char}_upper"
        
    # Lowercase a-z (36-61)
    for i in range(26):
        char = chr(ord('a') + i)
        mapping[36 + i] = f"{char}_lower"
        
    return mapping

def download_and_organize():
    print(f"🚀 Downloading EMNIST data to {DATA_RAW}...")
    
    # Create temp dir for download
    temp_dir = PROJECT_ROOT / 'data' / 'temp_emnist'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Download EMNIST (Split: byclass = digits + upper + lower)
    print("📦 Downloading dataset (this may take a while)...")
    try:
        dataset = EMNIST(
            root=temp_dir, 
            split='byclass', 
            download=True, 
            train=True
        )
    except Exception as e:
        print(f"❌ Error downloading EMNIST: {e}")
        return

    print("✅ Download complete. Organizing files...")
    
    mapping = get_emnist_mapping()
    
    # Create directories
    if DATA_RAW.exists():
        print(f"⚠️ Cleaning existing {DATA_RAW}...")
        shutil.rmtree(DATA_RAW)
    DATA_RAW.mkdir(parents=True)
    
    for folder_name in mapping.values():
        (DATA_RAW / folder_name).mkdir(exist_ok=True)
        
    # Save images
    # We'll save a subset to avoid millions of files (e.g., 500 per class is enough for analysis/finetuning demo)
    SAMPLES_PER_CLASS = 500 
    class_counts = {k: 0 for k in mapping.keys()}
    
    print(f"💾 Saving {SAMPLES_PER_CLASS} images per class...")
    
    for idx in tqdm(range(len(dataset))):
        img, label = dataset[idx]
        label = int(label)
        
        if label not in mapping:
            continue
            
        if class_counts[label] >= SAMPLES_PER_CLASS:
            continue
            
        # EMNIST images are rotated 90 degrees and flipped. We need to fix them.
        # Convert to numpy, transpose, and flip
        img_np = np.array(img)
        img_np = np.transpose(img_np)
        img_fixed = Image.fromarray(img_np)
        
        folder_name = mapping[label]
        filename = f"{folder_name}_{class_counts[label]}.png"
        
        img_fixed.save(DATA_RAW / folder_name / filename)
        class_counts[label] += 1
        
        # Check if we're done
        if all(c >= SAMPLES_PER_CLASS for c in class_counts.values()):
            break
            
    # Cleanup
    print("🧹 Cleaning up temp files...")
    shutil.rmtree(temp_dir)
    
    print(f"✨ Done! Data restored in {DATA_RAW}")

if __name__ == "__main__":
    from PIL import Image
    download_and_organize()
