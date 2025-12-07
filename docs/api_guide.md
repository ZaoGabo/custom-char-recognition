# Guía de API REST

Esta documentación detalla los endpoints disponibles en la API de reconocimiento de caracteres.

##  Inicio Rápido

1.  **Iniciar Servidor**:
    ```bash
    python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    ```
2.  **Documentación Interactiva (Swagger)**:
    -   Acceder a [http://localhost:8000/docs](http://localhost:8000/docs)

---

##  Endpoints Principales

### 1. Predicción desde Canvas (v2)
Optimizado para trazos dibujados a mano en la interfaz web.

-   **URL**: `/api/v2/predict`
-   **Método**: `POST`
-   **Entrada**:
    ```json
    {
      "image": [0.0, ..., 1.0],  // Array de 784 floats (28x28 normalizado)
      "width": 28,
      "height": 28
    }
    ```
-   **Respuesta**:
    ```json
    {
      "character": "A",
      "confidence": 0.98,
      "top5": [
        {"character": "A", "probability": 0.98},
        {"character": "a", "probability": 0.01},
        ...
      ]
    }
    ```

### 2. Predicción de Documentos (v3)
Optimizado para caracteres extraídos de documentos escaneados.

-   **URL**: `/api/v3/predict`
-   **Método**: `POST`
-   **Entrada**: Mismo formato que v2.
-   **Respuesta**: Mismo formato que v2.

### 3. Reconocimiento de Texto TrOCR (v4)
Reconocimiento de líneas completas de texto usando Transformers.

-   **URL**: `/api/v4/predict_text`
-   **Método**: `POST`
-   **Entrada**:
    ```json
    {
      "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
    }
    ```
-   **Respuesta**:
    ```json
    {
      "text": "Hola Mundo",
      "confidence": 0.95
    }
    ```

---

##  Estado del Sistema

### Health Check
Verifica que los modelos estén cargados en memoria.

-   **URL**: `/health`
-   **Método**: `GET`
-   **Respuesta**:
    ```json
    {
      "status": "healthy",
      "models": {
        "v2_finetuned": "loaded",
        "v3_super": "loaded",
        "trocr": "loaded"
      }
    }
    ```
