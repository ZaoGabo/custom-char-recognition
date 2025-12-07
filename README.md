# Custom Character Recognition System

Sistema de reconocimiento de caracteres manuscritos basado en redes neuronales convolucionales (CNN) y Transformers (TrOCR).



## Descripcion

Proyecto de aprendizaje profundo para clasificacion de 94 categorias de caracteres:
- A-Z mayusculas
- a-z minusculas  
- 0-9 digitos
- 32 simbolos especiales

### Modelos Disponibles

| Modelo | Uso | Precision |
|--------|-----|-----------|
| v2_finetuned | Canvas interactivo | 83.8% |
| v3_super | Documentos escaneados | 88.7% |
| TrOCR | Texto completo | Variable |

## Requisitos

- Python 3.10+
- PyTorch 2.0+
- 4 GB RAM minimo

```bash
pip install -r requirements.txt
```

## Inicio Rapido

### 1. Configurar Entorno

```bash
git clone https://github.com/ZaoGabo/custom-char-recognition.git
cd custom-char-recognition
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Ejecutar API

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 3. Acceder a Interfaces

- Selector: http://localhost:8000/
- Canvas (v2): http://localhost:8000/v2
- Documentos (v3): http://localhost:8000/v3
- API Docs: http://localhost:8000/docs

## Endpoints API

| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| `/` | GET | Interfaz web principal |
| `/v2` | GET | Canvas para dibujar |
| `/v3` | GET | Subir imagenes |
| `/health` | GET | Estado de la API |
| `/api/v2/predict` | POST | Prediccion desde canvas |
| `/api/v3/predict` | POST | Prediccion desde documento |
| `/api/v4/predict_text` | POST | Reconocimiento de texto (TrOCR) |

### Ejemplo de Uso

```python
import requests
import numpy as np

# Imagen 28x28 normalizada (0-1)
image = np.random.rand(784).tolist()

response = requests.post(
    "http://localhost:8000/api/v2/predict",
    json={"image": image}
)

result = response.json()
print(f"Caracter: {result['character']}")
print(f"Confianza: {result['confidence']:.2%}")
```

## Estructura del Proyecto

```
custom-char-recognition/
├── models/                    # Modelos entrenados
│   ├── cnn_modelo_v2_finetuned/
│   └── cnn_modelo_v3_super/
│
├── page/                      # Interfaz web
│   ├── index.html            # Selector de modelo
│   ├── v2_finetuned.html     # Canvas
│   └── v3_super.html         # Documentos
│
├── src/
│   ├── api/                   # FastAPI
│   │   ├── main.py           # Endpoints
│   │   └── schemas.py        # Modelos Pydantic
│   │
│   ├── core/                  # Constantes
│   │   └── constants.py
│   │
│   ├── utils/                 # Utilidades
│   │   └── logger.py
│   │
│   ├── cnn_model_v2.py       # Arquitectura CNN v2
│   ├── cnn_model_v3.py       # Arquitectura CNN v3
│   ├── cnn_predictor_v2_finetuned.py
│   ├── cnn_predictor_v3_super.py
│   ├── trocr_predictor.py    # TrOCR
│   ├── config.py
│   └── label_map.py
│
├── scripts/                   # Utilidades
│   ├── train_v3.py
│   ├── train_finetuned.py
│   ├── export_onnx.py
│   └── download_emnist.py
│
├── tests/                     # Tests
├── docs/                      # Documentacion
│   ├── api_guide.md          # Guia de uso API
│   ├── GUIA_ENTRENAMIENTO.md # Guia de entrenamiento
│   └── training_colab.ipynb  # Notebook de ejemplo
│
├── Dockerfile.api
├── docker-compose.yml
└── requirements.txt
```

## Docker

### Construccion

```bash
docker build -f Dockerfile.api -t char-recognition-api:latest .
```

### Ejecucion

```bash
docker run -p 8000:8000 char-recognition-api:latest
```

## Arquitectura de Modelos

### CNN v2 Finetuned

```
Input (1, 28, 28)
    │
    ├── Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    ├── Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    ├── Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    ├── Conv2D(256) → BatchNorm → ReLU → Dropout(0.5)
    │
    ├── Flatten
    │
    ├── Linear(512) → ReLU → Dropout(0.5)
    ├── Linear(256) → ReLU → Dropout(0.5)
    ├── Linear(128) → ReLU → Dropout(0.5)
    └── Linear(94)

Output: 94 clases
```

### TrOCR

Modelo Transformer de Microsoft para OCR de texto manuscrito:
- `microsoft/trocr-base-handwritten`
- Soporta lineas completas de texto
- Descarga automatica desde HuggingFace

## Preprocesamiento

El canvas web aplica el siguiente pipeline:

1. Deteccion de bounding box del trazo
2. Recorte con padding de 2px
3. Redimensionamiento a 20x20 (preservando aspect ratio)
4. Centrado en lienzo 28x28
5. Normalizacion a rango [0, 1]

## Testing

```bash
pytest tests/ -v
```

## Troubleshooting

### Modelo no encontrado
Verificar que existan los directorios en `models/`:
- `cnn_modelo_v2_finetuned/`
- `cnn_modelo_v3_super/`

### Puerto en uso
```bash
netstat -ano | findstr :8000
taskkill /PID <pid> /F
```

### Error CUDA
Reducir `batch_size` o usar CPU:
```bash
CUDA_VISIBLE_DEVICES="" python -m uvicorn src.api.main:app
```

## Licencia

MIT License - Ver archivo LICENSE
