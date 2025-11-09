# Custom Character Recognition

Sistema de reconocimiento de caracteres manuscritos basado en una CNN v2 fine-tuned sobre EMNIST y datos sinteticos de canvas.

## Resumen
- 94 clases: letras mayusculas y minusculas, digitos y simbolos.
- Modelo principal: `models/cnn_modelo_v2_finetuned/best_model_finetuned.pth` (83.80 % val acc, ~92 % en pruebas sinteticas).
- Aplicacion web en Streamlit (`demo/app.py`) con preprocesamiento alineado a EMNIST.
- Pipeline de entrenamiento robusto (`entrenar_finetune_robusto.py`) con AdamW, early stopping y reintentos.

## Requisitos
- Python 3.10+ (el repo usa 3.11.9).
- PyTorch 2.x con soporte CUDA opcional.
- Streamlit y streamlit-drawable-canvas para la UI.
- Dependencias listadas en `requirements.txt` y `requirements-test.txt`.

## Instalacion rapida
```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows
pip install -r requirements.txt
```

## Uso
### Aplicacion web
```bash
streamlit run demo/app.py
```
La app abre http://localhost:8501 y permite dibujar en el canvas. El preprocesamiento mantiene fondo negro y trazo blanco, por lo que no es necesario invertir colores.

### Inferencia programatica
```python
import numpy as np
from src.cnn_predictor_v2_finetuned import CNNPredictor_v2_Finetuned

predictor = CNNPredictor_v2_Finetuned()
imagen = np.random.rand(28, 28).astype("float32")  # Normalizada [0,1]
caracter, prob, top5 = predictor.predict(imagen)
print(caracter, prob)
```

## Arquitectura y metricas
- 4 bloques convolucionales (32-64-128-256 filtros) con batch norm y dropout.
- Capas totalmente conectadas (512-256-128-94) con dropout progresivo.
- Entrenada con AdamW, grad clipping opcional y scheduler step decay.
- Rendimiento observado:
  - Validacion: 83.80 %
  - Pruebas sinteticas (canvas): ~92 %
  - Inferencia en GPU: 10-15 ms por caracter

## Preprocesamiento del canvas
1. Se toma la imagen RGB del canvas y se convierte a escala de grises.
2. Se detecta el bounding box del trazo blanco y se recorta con padding.
3. Se redimensiona a 20x20 manteniendo la relacion de aspecto.
4. Se centra en un lienzo 28x28 con fondo negro y se normaliza a [0,1].

## Entrenamiento
El script `entrenar_finetune_robusto.py` expone CLI para reentrenar el modelo:
```bash
python entrenar_finetune_robusto.py --device cuda --epochs 5 --verbose
```
Caracteristicas clave:
- Reintentos automaticos ante `RecoverableTrainingError` (por ejemplo OOM).
- Checkpoints rotativos en `models/<destino>/checkpoints/`.
- Historico en `history.json` con metricas por epoca.
- Demo de OOM real (requiere GPU):
  ```bash
  python scripts/oom_recovery_demo.py --epochs 1 --batch-size 512 --verbose
  ```
  El primer intento provoca `torch.cuda.OutOfMemoryError` y el segundo completa la época.

## Tests
```bash
pytest tests/
```
Actualmente se incluyen pruebas para loader de datos, label map y utilidades.

## Estructura
```text
custom-char-recognition/
├── demo/app.py
├── models/cnn_modelo_v2_finetuned/
├── src/
│   ├── cnn_model_v2.py
│   ├── cnn_predictor_v2_finetuned.py
│   └── training/
└── tests/
```

## Troubleshooting rapido
- **La app no abre**: verifica instalacion de Streamlit y que el puerto 8501 este libre.
- **Prediccion incorrecta**: dibuja trazos mas gruesos y centrados; revisa el top-5 mostrado.
- **CUDA no disponible**: confirma instalacion de PyTorch con `torch.cuda.is_available()`.

## Licencia
Distribuido bajo la licencia [MIT](LICENSE).
