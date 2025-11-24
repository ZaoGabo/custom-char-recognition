# Custom Character Recognition System

Robust handwritten character recognition system based on deep convolutional neural networks, designed with modular architecture and software engineering best practices.

[Versión en Español](README.md)

## Project Overview

Deep learning project implemented with PyTorch for classification of 94 character categories (A-Z uppercase, a-z lowercase, 0-9 digits, and 32 special symbols). The main model achieves 83.80% validation accuracy on the EMNIST Extended dataset, with additional fine-tuning on synthetic canvas data.

### Key Features

- **Architecture**: CNN v2 with 4 convolutional blocks, batch normalization, and progressive dropout regularization
- **Dataset**: EMNIST Extended (697,932 images) with advanced augmentation using Albumentations
- **REST API**: FastAPI with ONNX support for optimized inference
- **Web Interface**: Interactive Streamlit canvas for real-time testing
- **CI/CD**: GitHub Actions configured for automated testing and Docker container builds

## System Requirements

- Python 3.10 or higher
- PyTorch 2.0+ (with optional CUDA support for GPU training)
- ONNX Runtime for production inference
- Minimum 4 GB RAM (16 GB recommended for training)

See `requirements.txt` for complete list of dependencies.

## Installation

### Environment Setup

```bash
git clone https://github.com/ZaoGabo/custom-char-recognition.git
cd custom-char-recognition

# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.venv\\Scripts\\activate

# Activate environment (Linux/macOS)
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables Configuration

```bash
# Copy configuration template
cp .env.example .env

# Edit .env according to your needs
# MODEL_VERSION=v2_finetuned
# LOG_LEVEL=INFO
# ENABLE_CUDA=true
```

## System Usage

### Interactive Web Application

```bash
streamlit run demo/app.py
```

The web application runs on `http://localhost:8501` and provides:
- Interactive drawing canvas
- Real-time predictions
- Top-5 alternatives visualization
- Confidence metrics

### REST API for Production

```bash
# Development mode
uvicorn src.api.main:app --reload

# Production mode
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API server exposes the following endpoints:
- `GET /`: Service information
- `GET /health`: Health check
- `POST /predict`: Single character prediction
- `POST /predict_text`: Full text prediction with segmentation

See interactive documentation at `http://localhost:8000/docs`

### Programmatic Usage

```python
from src.cnn_predictor_v2_finetuned import CNNPredictor_v2_Finetuned
import numpy as np

# Initialize predictor
predictor = CNNPredictor_v2_Finetuned()

# Normalized image in range [0, 1]
image = np.random.rand(28, 28).astype('float32')

# Make prediction
character, probability, top5 = predictor.predict(image)

print(f"Character: {character}")
print(f"Confidence: {probability:.2%}")
print(f"Top 5: {top5}")
```

## Model Architecture

### CNN v2 Finetuned (Production)

```
Input: (1, 28, 28)
│
├── Conv2D(32) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
├── Conv2D(64) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
├── Conv2D(128) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
├── Conv2D(256) → BatchNorm → ReLU → Dropout(0.5)
│
├── Flatten(2304)
│
├── Linear(512) → BatchNorm → ReLU → Dropout(0.5)
├── Linear(256) → BatchNorm → ReLU → Dropout(0.5)
├── Linear(128) → BatchNorm → ReLU → Dropout(0.5)
└── Linear(94)
│
Output: (94 classes)
```

**Technical Specifications:**
- Total parameters: 2.3M
- Optimizer: AdamW with weight decay 1e-5
- Learning rate: 0.0001 (with ReduceLROnPlateau scheduler)
- Regularization: Batch Normalization + Progressive Dropout
- Inference time (GPU): 10-15ms per image

### CNN v3 (Experimental)

ResNet-like architecture with residual blocks:
- 8 residual blocks with skip connections
- EMNIST accuracy: 85.44%
- Status: Trained on Google Colab, ONNX export pending

## Preprocessing Pipeline

### For Interactive Canvas

1. **Content detection**: Verification of drawn pixels (threshold: 50 pixels)
2. **Conversion**: RGB → Grayscale
3. **Bounding box**: Detection and cropping of region of interest with 2px padding
4. **Resizing**: Scaling to 20×20 preserving aspect ratio (LANCZOS)
5. **Centering**: Positioning on 28×28 canvas with black background
6. **Normalization**: Scaling to range [0, 1]

**Important Note**: The system maintains black background with white strokes, aligned with EMNIST format. No color inversion is applied.

## Training and Fine-tuning

### Base Training

```bash
python scripts/train_v3.py --device cuda --epochs 30 --batch-size 128
```

### Retraining with Synthetic Canvas

```bash
python entrenar_finetune_robusto.py \\
    --device cuda \\
    --epochs 30 \\
    --batch-size 64 \\
    --canvas-weight 0.3 \\
    --verbose
```

**Training Pipeline Features:**
- Early stopping with configurable patience
- Automatic checkpointing every 5 epochs
- Automatic retry on OutOfMemoryError
- Structured logging with file rotation
- Metrics history in JSON format

### Synthetic Data Generation

```bash
python src/generar_dataset_canvas.py
```

Generates 9,400 synthetic images simulating canvas strokes (100 per class).

## Performance Metrics

| Model | Architecture | Accuracy | Parameters | Size |
|-------|-------------|----------|------------|------|
| CNN v2 Finetuned | 4 Conv + 4 FC | 83.80% | 2.3M | 9.2 MB |
| CNN v3 ResNet | 8 ResBlocks | 85.44% | 1.8M | 7.1 MB |
| CNN v2 Base | 4 Conv + 4 FC | 77.92% | 2.3M | 9.2 MB |
| MLP Baseline | 4 FC | 71.41% | 512K | 2.0 MB |

All CNNs trained with intensive augmentation (rotation, noise, elastic distortion, grid distortion).

## Project Structure

```
custom-char-recognition/
├── .env.example              # Environment variables template
├── .gitignore                # Git rules (improved)
├── config.yml                # Centralized configuration
├── requirements.txt          # Production dependencies
├── requirements-test.txt     # Testing dependencies
│
├── demo/                     # Streamlit application
│   └── app.py
│
├── models/                   # Trained models
│   ├── cnn_modelo_v2_finetuned/
│   │   ├── best_model_finetuned.pth
│   │   └── model.onnx
│   └── cnn_modelo_v3/
│
├── src/                      # Source code
│   ├── core/                 # Core components
│   │   ├── __init__.py
│   │   └── constants.py     # Centralized constants
│   │
│   ├── utils/                # Utilities
│   │   ├── __init__.py
│   │   └── logger.py        # Structured logging
│   │
│   ├── api/                  # REST API
│   │   ├── main.py          # FastAPI endpoints
│   │   └── schemas.py       # Pydantic models
│   │
│   ├── analysis/             # Analysis and metrics
│   │   └── confusion_matrix.py
│   │
│   ├── models (Python)
│   │   ├── cnn_model_v2.py
│   │   └── cnn_model_v3.py
│   │
│   └── predictors
│       ├── cnn_predictor_v2_finetuned.py
│       └── cnn_predictor_v3.py
│
├── scripts/                  # Utility scripts
│   ├── debug/               # Debug scripts
│   ├── download_emnist.py
│   ├── train_v3.py
│   └── export_onnx.py
│
├── tests/                    # Test suite
│   ├── test_api_integration.py
│   ├── test_data_loader.py
│   ├── test_network.py
│   └── load_test.py
│
└── docs/                     # Documentation
    ├── api_guide.md
    └── deployment_guide.md
```

## Testing

### Running Unit Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific tests
pytest tests/test_api_integration.py -v
```

### Load Testing

```bash
python tests/load_test.py --requests 1000 --concurrency 50
```

Expected results:
- Throughput: 100-150 requests/second
- p95 Latency: < 100ms
- Success rate: > 99.5%

## Docker Deployment

### Building Images

```bash
# API
docker build -f Dockerfile.api -t char-recognition-api:latest .

# Streamlit
docker build -f Dockerfile.streamlit -t char-recognition-ui:latest .
```

### Running with Docker Compose

```bash
docker-compose up -d
```

Available services:
- API: `http://localhost:8000`
- UI: `http://localhost:8501`

## Implemented Best Practices

- **Centralized configuration**: Environment variables and YAML files
- **Structured logging**: Loguru with automatic rotation and configurable levels
- **Type hints**: Type annotations throughout codebase
- **Input validation**: Pydantic schemas in API
- **Automated testing**: GitHub Actions with linting and tests
- **Documentation**: Google-style docstrings in all public functions

## Troubleshooting

### Common Issues

**Error: Model not found**
```
Solution: Verify that MODEL_PATH in .env points to the correct file
```

**Error: CUDA out of memory**
```
Solution: Reduce batch_size in config.yml or train on CPU
```

**API not responding**
```
Solution: Verify that port 8000 is not in use
netstat -ano | findstr :8000  # Windows
lsof -i :8000  # Linux/macOS
```

**Incorrect canvas prediction**
```
Solution: Draw thicker, centered strokes, check top-5 for similar characters
```

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

### Style Guidelines

- **Python**: PEP 8 with flake8
- **Commits**: Conventional Commits
- **Documentation**: Google-style docstrings

## Roadmap

- [ ] Redis caching implementation
- [ ] Batch inference support
- [ ] Model v3 ONNX export
- [ ] Kubernetes deployment
- [ ] Performance metrics with Prometheus

## License

Distributed under MIT License. See `LICENSE` for more information.

