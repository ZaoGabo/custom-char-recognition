# Custom Character Recognition System

Sistema robusto de reconocimiento de caracteres manuscritos basado en redes neuronales convolucionales profundas, diseñado con arquitectura modular y mejores prácticas de ingeniería de software.

[English Version](README.en.md)

## Descripción General

Proyecto de aprendizaje profundo implementado con PyTorch para la clasificación de 94 categorías de caracteres (A-Z mayúsculas, a-z minúsculas, 0-9 dígitos y 32 símbolos especiales). El modelo principal alcanza un 83.80% de precisión en validación sobre el dataset EMNIST Extended, con fine-tuning adicional sobre datos sintéticos de canvas.

### Características Principales

- **Arquitectura**: CNN v2 con 4 bloques convolucionales, normalización por lotes y regularización dropout progresiva
- **Dataset**: EMNIST Extended (697,932 imágenes) con augmentación avanzada usando Albumentations
- **API REST**: FastAPI con soporte ONNX para inferencia optimizada
- **Interfaz Web**: Streamlit con canvas interactivo para pruebas en tiempo real
- **CI/CD**: GitHub Actions configurado para testing automático y construcción de contenedores Docker

## Requisitos del Sistema

- Python 3.10 o superior
- PyTorch 2.0+ (con soporte CUDA opcional para entrenamiento en GPU)
- ONNX Runtime para inferencia en producción
- 4 GB RAM mínimo (16 GB recomendado para entrenamiento)

Consulte `requirements.txt` para la lista completa de dependencias.

## Instalación

### Configuración del Entorno

```bash
git clone https://github.com/ZaoGabo/custom-char-recognition.git
cd custom-char-recognition

# Crear entorno virtual
python -m venv .venv

# Activar entorno (Windows)
.venv\\Scripts\\activate

# Activar entorno (Linux/macOS)
# source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Configuración de Variables de Entorno

```bash
# Copiar plantilla de configuración
cp .env.example .env

# Editar .env según sus necesidades
# MODEL_VERSION=v2_finetuned
# LOG_LEVEL=INFO
# ENABLE_CUDA=true
```

## Uso del Sistema

### Aplicación Web Interactiva

```bash
streamlit run demo/app.py
```

La aplicación web se ejecutará en `http://localhost:8501` y proporciona:
- Canvas de dibujo interactivo
- Predicciones en tiempo real
- Visualización de top-5 alternativas
- Métricas de confianza

### API REST para Producción

```bash
# Modo desarrollo
uvicorn src.api.main:app --reload

# Modo producción
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

El servidor API expone los siguientes endpoints:
- `GET /`: Información del servicio
- `GET /health`: Health check
- `POST /predict`: Predicción de carácter único
- `POST /predict_text`: Predicción de texto completo con segmentación

Consulte la documentación interactiva en `http://localhost:8000/docs`

### Uso Programático

```python
from src.cnn_predictor_v2_finetuned import CNNPredictor_v2_Finetuned
import numpy as np

# Inicializar predictor
predictor = CNNPredictor_v2_Finetuned()

# Imagen normalizada en rango [0, 1]
imagen = np.random.rand(28, 28).astype('float32')

# Realizar predicción
caracter, probabilidad, top5 = predictor.predict(imagen)

print(f"Carácter: {caracter}")
print(f"Confianza: {probabilidad:.2%}")
print(f"Top 5: {top5}")
```

## Arquitectura del Modelo

### CNN v2 Finetuned (Producción)

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
Output: (94 clases)
```

**Especificaciones Técnicas:**
- Parámetros totales: 2.3M
- Optimizador: AdamW con weight decay 1e-5
- Learning rate: 0.0001 (con scheduler ReduceLROnPlateau)
- Regularización: Batch Normalization + Dropout progresivo
- Tiempo de inferencia (GPU): 10-15ms por imagen

### CNN v3 (Experimental)

Arquitectura ResNet-like con bloques residuales:
- 8 bloques residuales con skip connections
- Precisión en EMNIST: 85.44%
- Estado: Entrenada en Google Colab, exportación ONNX pendiente

## Pipeline de Preprocesamiento

### Para Canvas Interactivo

1. **Detección de contenido**: Verificación de píxeles dibujados (threshold: 50 píxeles)
2. **Conversión**: RGB → Escala de grises
3. **Bounding box**: Detección y recorte del área de interés con padding 2px
4. **Redimensionamiento**: Escalado a 20×20 preservando aspect ratio (LANCZOS)
5. **Centrado**: Posicionamiento en lienzo 28×28 con fondo negro
6. **Normalización**: Escalado a rango [0, 1]

**Nota Importante**: El sistema mantiene fondo negro con trazo blanco, alineado con el formato EMNIST. No se aplica inversión de colores.

## Entrenamiento y Fine-tuning

### Entrenamiento Base

```bash
python scripts/train_v3.py --device cuda --epochs 30 --batch-size 128
```

### Reentrenamiento con Canvas Sintético

```bash
python entrenar_finetune_robusto.py \\
    --device cuda \\
    --epochs 30 \\
    --batch-size 64 \\
    --canvas-weight 0.3 \\
    --verbose
```

**Características del Pipeline de Entrenamiento:**
- Early stopping con paciencia configurable
- Checkpointing automático cada 5 epochs
- Reintentos automáticos ante OutOfMemoryError
- Logging estructurado con rotación de archivos
- Historial de métricas en formato JSON

### Generación de Datos Sintéticos

```bash
python src/generar_dataset_canvas.py
```

Genera 9,400 imágenes sintéticas simulando trazos de canvas (100 por clase).

## Métricas de Rendimiento

| Modelo | Arquitectura | Precisión | Parámetros | Tamaño |
|--------|-------------|-----------|------------|--------|
| CNN v2 Finetuned | 4 Conv + 4 FC | 83.80% | 2.3M | 9.2 MB |
| CNN v3 ResNet | 8 ResBlocks | 85.44% | 1.8M | 7.1 MB |
| CNN v2 Base | 4 Conv + 4 FC | 77.92% | 2.3M | 9.2 MB |
| MLP Baseline | 4 FC | 71.41% | 512K | 2.0 MB |

Todas las CNN entrenadas con augmentación intensiva (rotación, ruido, distorsión elástica, grid distortion).

## Estructura del Proyecto

```
custom-char-recognition/
├── .env.example              # Plantilla de variables de entorno
├── .gitignore                # Reglas Git (mejoradas)
├── config.yml                # Configuración centralizada
├── requirements.txt          # Dependencias de producción
├── requirements-test.txt     # Dependencias de testing
│
├── demo/                     # Aplicación Streamlit
│   └── app.py
│
├── models/                   # Modelos entrenados
│   ├── cnn_modelo_v2_finetuned/
│   │   ├── best_model_finetuned.pth
│   │   └── model.onnx
│   └── cnn_modelo_v3/
│
├── src/                      # Código fuente
│   ├── core/                 # Componentes principales
│   │   ├── __init__.py
│   │   └── constants.py     # Constantes centralizadas
│   │
│   ├── utils/                # Utilidades
│   │   ├── __init__.py
│   │   └── logger.py        # Logging estructurado
│   │
│   ├── api/                  # API REST
│   │   ├── main.py          # Endpoints FastAPI
│   │   └── schemas.py       # Modelos Pydantic
│   │
│   ├── analysis/             # Análisis y métricas
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
├── scripts/                  # Scripts de utilidad
│   ├── debug/               # Scripts de debugging
│   ├── download_emnist.py
│   ├── train_v3.py
│   └── export_onnx.py
│
├── tests/                    # Suite de tests
│   ├── test_api_integration.py
│   ├── test_data_loader.py
│   ├── test_network.py
│   └── load_test.py
│
└── docs/                     # Documentación
    ├── api_guide.md
    └── deployment_guide.md
```

## Testing

### Ejecución de Tests Unitarios

```bash
# Todos los tests
pytest tests/ -v

# Con cobertura
pytest tests/ --cov=src --cov-report=html

# Tests específicos
pytest tests/test_api_integration.py -v
```

### Load Testing

```bash
python tests/load_test.py --requests 1000 --concurrency 50
```

Resultados esperados:
- Throughput: 100-150 requests/segundo
- Latencia p95: < 100ms
- Tasa de éxito: > 99.5%

## Deployment con Docker

### Construcción de Imágenes

```bash
# API
docker build -f Dockerfile.api -t char-recognition-api:latest .

# Streamlit
docker build -f Dockerfile.streamlit -t char-recognition-ui:latest .
```

### Ejecución con Docker Compose

```bash
docker-compose up -d
```

Servicios disponibles:
- API: `http://localhost:8000`
- UI: `http://localhost:8501`

## Mejores Prácticas Implementadas

- **Configuración centralizada**: Variables de entorno y archivos YAML
- **Logging estructurado**: Loguru con rotación automática y niveles configurables
- **Type hints**: Anotaciones de tipo en todo el código
- **Validación de inputs**: Pydantic schemas en la API
- **Testing automatizado**: GitHub Actions con linting y tests
- **Documentación**: Docstrings estilo Google en todas las funciones públicas

## Troubleshooting

### Problemas Comunes

**Error: Model not found**
```
Solución: Verificar que MODEL_PATH en .env apunte al archivo correcto
```

**Error: CUDA out of memory**
```
Solución: Reducir batch_size en config.yml o entrenar en CPU
```

**API no responde**
```
Solución: Verificar que el puerto 8000 no esté en uso
netstat -ano | findstr :8000  # Windows
lsof -i :8000  # Linux/macOS
```

**Predicción incorrecta en canvas**
```
Solución: Dibujar trazos más gruesos y centrados, verificar top-5 para caracteres similares
```

## Contribuciones

Las contribuciones son bienvenidas. Por favor, siga estos pasos:

1. Fork del repositorio
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

### Guías de Estilo

- **Python**: PEP 8 con flake8
- **Commits**: Conventional Commits
- **Documentación**: Docstrings estilo Google

## Roadmap

- [ ] Implementación de caching con Redis
- [ ] Soporte para batch inference
- [ ] Export del modelo v3 a ONNX
- [ ] Deployment en Kubernetes
- [ ] Métricas de performance con Prometheus

## Licencia

Distribuido bajo licencia MIT. Consulte `LICENSE` para más información.
