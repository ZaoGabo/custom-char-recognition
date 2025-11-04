# Custom Character Recognition

Handwritten character recognition system based on convolutional neural networks (CNN).

[Español](README.md)

## Description

Deep learning project for classification of 94 different characters using a CNN architecture trained on the EMNIST dataset and fine-tuned with synthetic canvas data. The model achieves 83.80% validation accuracy.

**Character set**: A-Z (uppercase), a-z (lowercase), 0-9 (digits), 32 special symbols

## Features

- **Architecture**: CNN v2 with 4 convolutional blocks and batch normalization
- **Accuracy**: 83.80% on validation (CNN v2 Finetuned)
- **Dataset**: EMNIST Extended + synthetic canvas data (30% in fine-tuning)
- **Framework**: PyTorch for CNN, NumPy for alternative MLP
- **Interface**: Streamlit web application with interactive canvas

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Streamlit
- NumPy, Pillow, pandas

See `requirements.txt` for complete list.

## Installation

```bash
git clone https://github.com/ZaoGabo/custom-char-recognition.git
cd custom-char-recognition
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## Quick Start

### Web Application

```bash
streamlit run demo/app.py
```

The application allows drawing characters on a canvas and getting real-time predictions.

### Programmatic Inference

```python
from src.cnn_predictor_v2_finetuned import CNNPredictor_v2_Finetuned
import numpy as np

predictor = CNNPredictor_v2_Finetuned()

image = np.random.rand(28, 28)  # 28x28 normalized image [0,1]
character, probability, top5 = predictor.predict(image)

print(f"Prediction: {character} ({probability*100:.1f}%)")
```

## Model Architecture

### CNN v2 (Production)

```
Input (1, 28, 28)
│
├─ Conv2d(32) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.25)
├─ Conv2d(64) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.25)
├─ Conv2d(128) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.25)
├─ Conv2d(256) + BatchNorm + ReLU + Dropout(0.5)
│
├─ Flatten (2304)
│
├─ Linear(512) + BatchNorm + ReLU + Dropout(0.5)
├─ Linear(256) + BatchNorm + ReLU + Dropout(0.5)
├─ Linear(128) + BatchNorm + ReLU + Dropout(0.5)
└─ Linear(94)
│
Output (94 classes)
```

**Total parameters**: ~2.3M  
**Architecture**: 4 conv blocks + 4 fully connected layers  
**Regularization**: Batch normalization + progressive Dropout  
**Optimizer**: Adam (lr=0.0001 in fine-tuning)

### MLP (Alternative)

Simpler architecture available in `src/network.py`:
- Layers: [784, 512, 256, 128, 94]
- Accuracy: 71.41%
- Use case: Resource-constrained environments

## Training

### CNN Model Fine-tuning

The base model was trained on EMNIST Extended and subsequently fine-tuned with canvas data:

```bash
python entrenar_finetune_robusto.py
```

**Fine-tuning configuration**:
- Epochs: 30
- Batch size: 64
- Learning rate: 0.0001
- Canvas weight: 30%
- Early stopping: patience 10

### Generate Synthetic Canvas Data

```bash
python src/generar_dataset_canvas.py
```

Generates 9,400 synthetic images simulating canvas strokes (100 per class).


## Results

| Model | Architecture | Accuracy | Parameters |
|-------|-------------|----------|------------|
| CNN v2 Finetuned | 4 Conv + 4 FC | 83.80% | 2.3M |
| CNN v2 Base | 4 Conv + 4 FC | 77.92% | 2.3M |
| CNN v1 | 3 Conv + 3 FC | 77.22% | 434K |
| MLP | 4 FC | 71.41% | 512K |

**Note**: CNNs were trained with aggressive augmentation (rotation, noise, blur, shift).

## Preprocessing

Canvas preprocessing pipeline:

1. RGB channel extraction
2. Grayscale conversion
3. Resize to 28x28 (LANCZOS)
4. Normalization to [0, 1] range

**Important**: No color inversion is applied. Canvas uses black background with white strokes, same as EMNIST.

## Configuration

Main configuration in `src/config.py`:

```python
CUSTOM_LABELS = [
    'A', 'B', ..., 'Z',  # 0-25
    'a', 'b', ..., 'z',  # 26-51
    '0', '1', ..., '9',  # 52-61
    '!', '@', '#', ...   # 62-93
]

NETWORK_CONFIG = {
    'capas': [784, 512, 256, 128, 94],
    'activaciones': ['relu', 'relu', 'relu', 'softmax'],
    'tasa_aprendizaje': 0.001,
    'dropout_rate': 0.1,
    # ...
}
```

## Tests

Run test suite:

```bash
pytest tests/
```

Available tests:
- `test_network.py`: MLP architecture
- `test_data_loader.py`: Data loading
- `test_label_map.py`: Label mapping
- `test_predictor.py`: Inference
- `test_utils.py`: Utilities

## Technical Notes

### Label Ordering

The system maintains a specific label order (A-Z, a-z, 0-9, symbols). The `LabelMap` and `DataLoader` modules preserve this order using `list()` instead of `sorted()` to avoid alphabetical reordering.

### Empty Canvas Detection

The application verifies that the canvas contains content before predicting:

```python
alpha_channel = canvas_data[:, :, 3]
drawn_pixels = np.sum(alpha_channel > 200)
is_empty = drawn_pixels < 50
```

### Available Models

- **CNN v2 Finetuned** (recommended): Best accuracy, optimized for canvas
- **CNN v2 Base**: Without fine-tuning, good general accuracy
- **MLP**: Lightweight alternative for resource-constrained environments

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## License

MIT License - See `LICENSE` for details.

## Author

ZaoGabo - [GitHub](https://github.com/ZaoGabo)

## Acknowledgments

- EMNIST Dataset: Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017)
- Framework: PyTorch
- UI: Streamlit and streamlit-drawable-canvas
