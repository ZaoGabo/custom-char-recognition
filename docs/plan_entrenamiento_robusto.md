# Plan de Trabajo: Entrenamiento Robusto y Despliegue

> Documento interno de planificación. No subir a control de versiones sin revisión.

## Visión General

Construir un pipeline de entrenamiento altamente robusto (basado en `RobustTrainer`) y preparar el camino para un despliegue de inferencia en producción con infraestructura de soporte. Este plan asume una duración aproximada de **4 semanas** y se divide en fases con entregables claros.

## Línea de Tiempo (Resumen)

| Semana | Enfoque principal | Entregables clave |
|--------|-------------------|-------------------|
| 0 (Día 0-2) | Preparación y diagnóstico | Auditoría de entorno, mapa de riesgos, definición de KPIs |
| 1 | Entrenamiento robusto | Implementación `RobustTrainer` integrada al pipeline actual |
| 2 | Data balancing + Sintéticos | Pipeline `BalancedAugmentation` + validación de dataset |
| 3 | API de producción + CI/CD | FastAPI + Docker + Redis/Postgres + pipeline GitHub Actions |
| 4 | Infraestructura extendida | Auto GPU setup, monitoreo (Prometheus/Grafana), hardening |

## Fase 0 (Día 0-2): Preparación

**Objetivo**: Tener claridad del estado actual del proyecto y preparar el entorno de trabajo.

- **Estado actual** (08/11/2025):
  - Python 3.11.9; PyTorch 2.6.0+cu124 (CUDA 12.4 habilitado, RTX 2060 disponible según `nvidia-smi`).
  - Suite `pytest`: 40 tests ✅ (1 warning de Pillow por `mode`, corregir antes de Pillow 13).
  - Dataset `data/raw/`: 94 clases, 62 000 imágenes (prom. 660, min 620, max 700).
  - Métrica base `val_acc`: 83.80 % (modelo `cnn_modelo_v2_finetuned`, epoch 27 / 30).

- **KPIs iniciales propuestos**:
  | KPI | Línea base | Objetivo | Responsable |
  |-----|-----------|----------|-------------|
  | `val_acc` CNN v2 fine-tuned | 83.8 % | ≥ 84.5 % sostenido | Equipo de entrenamiento |
  | Tiempo por epoch (batch 64) | ~120 s (CPU) | ≤ 60 s con GPU | Equipo MLOps |
  | Convergencia sin fallos | 100 % (runs cortos) | ≥ 95 % runs largos con `RobustTrainer` | Equipo de entrenamiento |
  | Disponibilidad API | N/A | ≥ 99 % ideal | Equipo backend |

- **Riesgos prioritarios**:
  1. **PyTorch sin CUDA** → instalar binarios con GPU para reducir tiempos de entrenamiento.
  2. **Clases borderline (<650 muestras)** → requerirán `BalancedAugmentation` para evitar sobreajuste.
  3. **Pillow Deprecation** → refactor de `Image.fromarray(..., mode='L')` antes de 2026.
  4. **Falta de monitoreo/CI** → se cubre en Fases 3-4.

- **Día 0**
  - Revisar dependencias (`requirements.txt`) y versiones de Python/PyTorch.
  - Ejecutar `pytest` completo para verificar línea base de tests.
  - Documentar métricas actuales: `val_acc`, tiempos de epoch, uso de memoria.
- **Día 1**
  - Analizar capacidad de GPU: ejecutar `nvidia-smi`, benchmark ligero (ver script `auto_gpu_setup`).
  - Identificar directorios de datos (`data/raw`, `data/processed`) y disponibilidad real (62k imágenes confirmadas).
- **Día 2**
  - Definir KPIs: precisión objetivo (>84%), tiempo máximo por epoch, tasa de fallos tolerable.
  - Elaborar checklist de riesgos (falta de GPU, OOM, desbalance, latencia API).

**Entregables**: Informe corto con estado inicial; issues creados para tareas clave.

## Fase 1 (Semana 1): Entrenamiento Robusto

**Objetivo**: Integrar el `RobustTrainer` para entrenamientos resilientes.

- **Snippet base**: Clase `RobustTrainer` (manejo de interrupciones, OOM, gradient clipping, checkpoints rotativos, early stopping). Este código servirá como plantilla para `src/training/robust_trainer.py` y para extender `entrenar_finetune_robusto.py`.

- **Día 3-4**
  - Crear módulo `src/training/robust_trainer.py` implementando clase `RobustTrainer` (tomando como base el snippet validado).
  - Adaptar `entrenar_finetune_robusto.py` para usar `RobustTrainer.train()` con `AdamW` y scheduler (Cosine Annealing o actual step-decay).
- **Día 5**
  - Añadir lógica de checkpoints rotativos (`models/checkpoints/`) y best-model.
  - Integrar reporting de métricas (JSON + consola) compatible con historial existente.
- **Día 6**
  - Pruebas controladas: 3 epochs en subconjunto. Verificar manejo de `KeyboardInterrupt`, OOM simulado.
- **Día 7**
  - Documentar flujo en README (sección “Entrenamiento robusto”).
  - Crear scripts auxiliares: `scripts/resume_training.py`, `scripts/analyze_history.py` (opcional).

**Entregables**: Código en master con tests actualizados; tutorial interno de uso.

### Progreso al 08/11/2025

- ✅ Días 3-5 completados: módulo `RobustTrainer`, script CLI actualizado y documentación básica en README.
- ✅ Entrenamiento corto de validación (`--epochs 1`, CPU): `acc_val`=0.6897, historial en `models/cnn_modelo_v2_finetuned/history.json`, checkpoints rotativos activos.
- ✅ Prueba de reintentos (Día 6): `RecoverableTrainingError` simulado; el segundo intento completó el entrenamiento con éxito.
- ✅ Simulación `KeyboardInterrupt` (Día 6): reintento automático completó el entrenamiento tras la interrupción manual.
- ✅ Simulación OOM (Día 6): `torch.cuda.OutOfMemoryError` forzada en GPU; el reintento completó el entrenamiento.
- ✅ OOM real reproducido (Día 6): `scripts/oom_recovery_demo.py` provoca un OOM genuino en el primer intento (patch de `_run_epoch` + `resume=False`), y el reintento automático completa la época.
- ✅ Entrenamiento en GPU (Día 6): corrida de 3 épocas con `--device cuda`, mejor `acc_val`=0.7211 y artefactos actualizados.

## Fase 2 (Semana 2): BalancedAugmentation & Canvas Pipeline

**Objetivo**: Construir dataset balanceado combinando EMNIST y datos sintéticos.

- **Snippet base**: `BalancedAugmentation`, `CanvasStyleAugmentation` y `create_balanced_dataset` (estrategias light/medium/aggressive, pipeline de mixup y combinación 70/30). Se integrará en `src/data/balanced_augmentation.py` y `scripts/create_balanced_dataset.py`.

- **Día 8-9**
  - Implementar módulos:
    - `src/data/balanced_augmentation.py` con `BalancedAugmentation` y `CanvasStyleAugmentation`.
    - Añadir tests unitarios básicos (comprobar aumento de muestras, preservación de etiquetas).
- **Día 10**
  - Script `scripts/create_balanced_dataset.py` con CLI:
    ```bash
    python scripts/create_balanced_dataset.py --target-per-class 800 --mixup
    ```
  - Guardar datasets generados en `data/processed/balanced_v1/`.
- **Día 11**
  - Evaluar impacto en entrenamiento (correr 5 epochs). Comparar métricas con dataset original.
  - Ajustar parámetros de augmentación según resultados (clip de ruido, thresholds).
- **Día 12-13**
  - Documentar pipeline, añadir sección en README y en `docs/data_pipeline.md`.
  - Preparar job opcional para generarse vía CI (nightly data augmentation).

**Entregables**: Dataset balanceado reproducible + guías de uso.

## Fase 3 (Semana 3): API de Producción y CI/CD

**Objetivo**: Exponer el modelo con FastAPI y preparar infraestructura de despliegue.

- **Snippets base**:
  - API FastAPI (`ModelManager`, `CacheManager`, endpoints `/predict`, `/predict/batch`, `/stats`, Redis cache, rate limiting).
  - Infraestructura Docker (`docker-compose`, `Dockerfile`, `nginx.conf`, `init.sql`, `prometheus.yml`, `Makefile`, workflow GitHub Actions).
  Estos artefactos guiarán la construcción de `api/`, `docker/`, `.github/workflows/` y documentación asociada.

- **Día 14-15**
  - Crear paquete `api/` con FastAPI (basado en snippet validado).
  - Implementar `ModelManager`, `CacheManager` y endpoints `/predict`, `/predict/batch`, `/stats`.
- **Día 16**
  - Dockerizar la API (`Dockerfile`) y preparar `docker-compose.yml` con Redis, Postgres, Nginx.
  - Configurar `init.sql`, `nginx.conf`, `prometheus.yml`, `grafana` dashboards iniciales.
- **Día 17**
  - Pipeline GitHub Actions (`.github/workflows/deploy.yml`) con jobs: tests → build → deploy.
  - Probar despliegue en entorno staging (VM local o cloud).
- **Día 18**
  - Tests de carga ligera (`locust`, `k6`) enfocados en `/predict` (latencia <150ms con GPU).
  - Configurar logging centralizado (`logs/` + dashboards Prometheus/Grafana).

**Entregables**: Stack docker-compose operativo; CI/CD funcional; documentación de endpoints.

## Fase 4 (Semana 4): Infraestructura Extendida

**Objetivo**: Mejorar DX (developer experience) y monitoreo continuo.

- **Snippet base**: Script `auto_gpu_setup.py` (detección Colab/Kaggle/local, setup CUDA, creación de `test_gpu.py`). Se agregará en `scripts/auto_gpu_setup.py` y documentará en `docs/operaciones.md`.

- **Día 19-20**
  - Integrar script `auto_gpu_setup.py` para colaboradores/notebooks.
  - Añadir utilidades de diagnóstico (`test_gpu.py`) al repositorio.
- **Día 21**
  - Refuerzo de seguridad: limitar CORS, agregar rate limiting extra en Nginx, revisar políticas de Redis/Postgres.
- **Día 22-23**
  - Documentar guías operativas (`docs/operaciones.md`): cómo reanudar entrenamientos, cómo monitorear, checklist de despliegue.
  - Crear playbooks de incidentes (OOM, downtime API, degradación de accuracy).
- **Día 24-25**
  - Retrospectiva interna: comparar métricas antes/después, identificar mejoras pendientes.
  - Plan de mantenimiento (retraining trimestral, limpieza de checkpoints, rotación de claves).

**Entregables**: Herramientas de soporte, documentación operativa, planes de mantenimiento.

## Referencias y Artefactos Relevantes

- **Entrenamiento actual**: `src/training/pipeline.py`, `entrenar_finetune_robusto.py`
- **Dataset**: `data/raw/`, `data/processed/`, `data/canvas_synthetic/`
- **Modelos**: `models/cnn_modelo_v2_finetuned/`, `history_finetuned.json`
- **Tests**: `tests/test_trainer.py`, `tests/test_data_loader.py`
- **Infraestructura planificada**:
  - API: snippets FastAPI + Redis + Postgres + Nginx (ver documentación adjunta)
  - CI/CD: `.github/workflows/deploy.yml` (por implementar)

## Próximos Pasos Inmediatos

1. Validar disponibilidad de GPU y recursos (Fase 0 – Día 0).
2. Crear rama `feature/robust-training` para iniciar Fase 1.
3. Agendar checkpoints semanales con resultados intermedios.

---

**Nota**: Ajustar calendario según disponibilidad del equipo y recursos de cómputo. Este documento sirve como guía; los hitos pueden adaptarse si surgen prioridades urgentes (bugs críticos, cambios de infraestructura, etc.).
