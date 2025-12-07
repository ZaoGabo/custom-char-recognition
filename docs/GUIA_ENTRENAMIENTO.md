# Guía de Entrenamiento Robusto

Esta guía describe el proceso de entrenamiento del modelo CNN v2 utilizando el pipeline robusto (`RobustTrainer`).

## Descripción General

El sistema implementa un entrenador resiliente que maneja automáticamente:
-   **Interrupciones**: Detención segura ante `Ctrl+C` o señales del sistema.
-   **Errores de Memoria (OOM)**: Recuperación automática ante `CUDA OutOfMemory`.
-   **Checkpoints**: Guardado rotativo de los mejores modelos.
-   **Early Stopping**: Detención automática si no hay mejora.

## Script de Entrenamiento

El script principal es `scripts/train_finetuned.py`.

### Uso Básico

```bash
python scripts/train_finetuned.py --epochs 30 --device cuda
```

### Argumentos Principales

| Argumento | Descripción | Default |
|-----------|-------------|---------|
| `--epochs` | Número de épocas de entrenamiento | (Desde config) |
| `--device` | Dispositivo (`cuda`, `cpu`) | `auto` |
| `--no-resume` | Iniciar entrenamiento desde cero (ignorar checkpoints previos) | `False` |
| `--max-retries` | Número de intentos de recuperación ante fallos | 5 |
| `--patience` | Épocas sin mejora antes de detener (Early Stopping) | 8 |
| `--verbose` | Mostrar barra de progreso detallada por paso | `False` |

## Pipeline de Datos

El entrenamiento utiliza un pipeline de datos optimizado para manejar clases desbalanceadas (si aplica en futuras versiones) y aumentación de datos.

1.  **Carga**: `src/data_loader.py` lee imágenes RAW y procesadas.
2.  **Preprocesamiento**: Normalización y redimensionamiento (28x28).
3.  **Aumentación**: Rotaciones, traslaciones y ruido aleatorio (configurado en `RobustTrainer`).

## Recuperación de Fallos

Si el entrenamiento se interrumpe (ej. error de red, memoria), el script:
1.  Captura la excepción.
2.  Espera un tiempo de enfriamiento (5-10s).
3.  Reinicia el proceso cargando el último checkpoint válido (`checkpoint_last.pth`).
4.  Continúa desde la época exacta donde falló.

## Resultados

Los artefactos se guardan en `models/cnn_modelo_v2_finetuned/`:
-   `best_model_finetuned.pth`: Pesos del modelo con mejor precisión de validación.
-   `checkpoint_last.pth`: Último estado completo (para reanudar).
-   `history.json`: Métricas de entrenamiento y validación por época.
-   `config_finetuned.json`: Hiperparámetros utilizados.
