# Custom Character Recognition

Sistema de reconocimiento de caracteres manuscritos basado en redes neuronales convolucionales (CNN).

[English](README.en.md)

## Descripción

Proyecto de deep learning para clasificación de 94 caracteres diferentes utilizando una arquitectura CNN entrenada sobre el dataset EMNIST y fine-tuned con datos sintéticos de canvas. El modelo alcanza un 83.80% de precisión en validación.

**Conjunto de caracteres**: A-Z (mayúsculas), a-z (minúsculas), 0-9 (dígitos), 32 símbolos especiales

## Características

- **Arquitectura**: CNN v2 con 4 bloques convolucionales y batch normalization
- **Precisión**: 83.80% en validación (CNN v2 Finetuned)
- **Dataset**: EMNIST Extended + datos sintéticos de canvas (30% en fine-tuning)
- **Framework**: PyTorch para CNN, NumPy para MLP alternativo
- **Interface**: Aplicación web Streamlit con canvas interactivo

## Requisitos

- Python 3.8+
- PyTorch 2.0+
- Streamlit
- NumPy, Pillow, pandas

Ver `requirements.txt` para lista completa.

## Instalación

```bash
git clone https://github.com/ZaoGabo/custom-char-recognition.git
cd custom-char-recognition
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## Uso Rápido

### Aplicación Web

```bash
streamlit run demo/app.py
```

La aplicación permite dibujar caracteres en un canvas y obtener predicciones en tiempo real.

### Inferencia Programática

```python
from src.cnn_predictor_v2_finetuned import CNNPredictor_v2_Finetuned
import numpy as np

predictor = CNNPredictor_v2_Finetuned()

imagen = np.random.rand(28, 28)  # Imagen 28x28 normalizada [0,1]
caracter, probabilidad, top5 = predictor.predict(imagen)

print(f"Predicción: {caracter} ({probabilidad*100:.1f}%)")
```

## Arquitectura del Modelo

### CNN v2 (Producción)

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

**Parámetros totales**: ~2.3M  
**Arquitectura**: 4 bloques conv + 4 capas fully connected  
**Regularización**: Batch normalization + Dropout progresivo  
**Optimizador**: Adam (lr=0.0001 en fine-tuning)

### MLP (Alternativa)

Arquitectura más simple disponible en `src/network.py`:
- Capas: [784, 512, 256, 128, 94]
- Precisión: 71.41%
- Uso: Entornos con recursos limitados

## Entrenamiento

### Fine-tuning del modelo CNN

El modelo base fue entrenado en EMNIST Extended y posteriormente fine-tuned con datos de canvas:

```bash
python entrenar_finetune_robusto.py
```

**Configuración de fine-tuning**:
- Epochs: 30
- Batch size: 64
- Learning rate: 0.0001
- Canvas weight: 30%
- Early stopping: patience 10

### Generar datos sintéticos de canvas

```bash
python src/generar_dataset_canvas.py
```

Genera 9,400 imágenes sintéticas simulando trazos de canvas (100 por clase).


## Resultados

| Modelo | Arquitectura | Precisión | Parámetros |
|--------|-------------|-----------|------------|
| CNN v2 Finetuned | 4 Conv + 4 FC | 83.80% | 2.3M |
| CNN v2 Base | 4 Conv + 4 FC | 77.92% | 2.3M |
| CNN v1 | 3 Conv + 3 FC | 77.22% | 434K |
| MLP | 4 FC | 71.41% | 512K |

**Nota**: Las CNN fueron entrenadas con augmentation agresivo (rotación, ruido, blur, shift).

## Preprocesamiento

El pipeline de preprocesamiento para canvas:

1. Extracción de canal RGB
2. Conversión a escala de grises
3. Redimensionamiento a 28x28 (LANCZOS)
4. Normalización a rango [0, 1]

**Importante**: No se aplica inversión de colores. El canvas usa fondo negro y trazos blancos, igual que EMNIST.

## Configuración

Configuración principal en `src/config.py`:

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

Ejecutar suite de tests:

```bash
pytest tests/
```

Tests disponibles:
- `test_network.py`: Arquitectura MLP
- `test_data_loader.py`: Carga de datos
- `test_label_map.py`: Mapeo de etiquetas
- `test_predictor.py`: Inferencia
- `test_utils.py`: Utilidades

## Notas Técnicas

### Orden de etiquetas

El sistema mantiene un orden específico de etiquetas (A-Z, a-z, 0-9, símbolos). Los módulos `LabelMap` y `DataLoader` preservan este orden utilizando `list()` en lugar de `sorted()` para evitar reordenamiento alfabético.

### Detección de canvas vacío

La aplicación verifica que el canvas contiene contenido antes de predecir:

```python
alpha_channel = canvas_data[:, :, 3]
drawn_pixels = np.sum(alpha_channel > 200)
is_empty = drawn_pixels < 50
```

### Modelos disponibles

- **CNN v2 Finetuned** (recomendado): Mejor precisión, optimizado para canvas
- **CNN v2 Base**: Sin fine-tuning, buena precisión general
- **MLP**: Alternativa ligera para entornos con recursos limitados

## Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -am 'Agrega nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

## Licencia

MIT License - Ver `LICENSE` para más detalles.

## Autor

ZaoGabo - [GitHub](https://github.com/ZaoGabo)

## Reconocimientos

- Dataset EMNIST: Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017)
- Framework: PyTorch
- UI: Streamlit y streamlit-drawable-canvas
