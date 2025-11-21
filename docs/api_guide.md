# Guía de API y Optimización ONNX

Esta guía documenta cómo utilizar la API REST de reconocimiento de caracteres y el proceso de exportación a ONNX.

## 🚀 Resumen
El proyecto incluye una fase de optimización que consiste en:
1.  **Optimización**: Exportación del modelo PyTorch a **ONNX** (`models/cnn_modelo_v2_finetuned/model.onnx`).
2.  **API**: Servicio REST con **FastAPI** (`src/api/main.py`) para inferencia eficiente.

## 🛠️ Cómo ejecutar la API

### 1. Iniciar el servidor
```bash
uvicorn src.api.main:app --reload
```
La API estará disponible en `http://localhost:8000`.

### 2. Documentación Interactiva
Abre tu navegador en `http://localhost:8000/docs` para ver la interfaz Swagger UI, donde puedes probar los endpoints directamente.

### 3. Probar Predicción (Ejemplo con cURL)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"image": [0.0, 0.0, ...]}' # Array de 784 floats (imagen 28x28 aplanada)
```

## 📦 Exportación a ONNX

Si reentrenas el modelo y necesitas actualizar la versión ONNX, utiliza el script de exportación:

```bash
python scripts/export_onnx.py
```

Este script:
1. Carga el último checkpoint (`best_model_finetuned.pth`).
2. Infiere la configuración del modelo.
3. Exporta a `models/cnn_modelo_v2_finetuned/model.onnx`.
4. Verifica numéricamente que la salida coincida con PyTorch.

## 🧪 Tests de Integración

Para verificar que la API funciona correctamente (Health check + Predicción):

```bash
python tests/test_api_integration.py
```

## 📂 Archivos Clave
- `scripts/export_onnx.py`: Script de conversión a ONNX.
- `src/api/main.py`: Aplicación FastAPI.
- `src/api/schemas.py`: Modelos de datos Pydantic.
