[English](README.en.md)
# Sistema de Reconocimiento de Caracteres Personalizados

> **Importante:** si quieres reconocer letras con texturas, estilos o degradados diferentes a los datos sintéticos incluidos, debes reentrenar el modelo con ejemplos que representen esos estilos. Usa python scripts/run_pipeline.py --force para generar un modelo actualizado y luego pruébalo rápidamente con la interfaz Streamlit (streamlit run demo/app.py).

## Resumen

Este proyecto implementa una red neuronal multicapa (784 ➜ 512 ➜ 256 ➜ 128 ➜ 52) escrita en NumPy para clasificar 52 caracteres (A–Z y a–z). Incluye:
- src/network.py: red con ReLU, softmax, Adam, dropout y regularización L2.
- src/data_loader.py: carga de carpetas data/raw/, augmentación opcional, particiones estratificadas aunque falte sklearn o cv2.
- src/trainer.py: flujo de entrenamiento/evaluación; genera métricas para train/val/test en un solo paso.
- scripts/run_pipeline.py: CLI unificado para entrenar y analizar confusiones (--force, --skip-train, --confusion-report).
- demo/app.py: demo Streamlit compatible con modelos antiguos y nuevos.

## Requisitos

- Python 3.8+
- Dependencias listadas en 
equirements.txt (NumPy obligatorio; cv2/sklearn opcionales).

## Instalación Rápida

```
git clone https://github.com/ZaoGabo/custom-char-recognition.git
cd custom-char-recognition
python -m venv venv
venv\Scripts\activate  # o source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Entrenamiento y Evaluación

1. Organiza tus imágenes en data/raw/<letra>_<upper|lower>/. Cada archivo puede ser PNG/JPG/JPEG/BMP y será redimensionado a 28×28.
2. Ejecuta el pipeline:
   `
   python scripts/run_pipeline.py --force --confusion-report --limit 5
   `
   - Genera datos sintéticos si la carpeta está vacía.
   - Aplica augmentación (ruido, shifts, escalado) si DATA_CONFIG['usar_augmentacion'] está activo.
   - Muestra las métricas finales y las clases con menor precisión.
3. El modelo resultante se guarda en models/modelo_entrenado.pkl.

### Entrenamiento incremental / datos reales

Si tienes muestras reales adicionales puedes reutilizar 	rain_con_imagenes_reales.py, que ahora reconoce la ruta data/raw/. Recuerda volver a correr el pipeline para actualizar métricas.

## Uso con Streamlit

`
streamlit run demo/app.py
`
La app realiza preprocesamiento (escala de grises + 28×28), muestra top-5 predicciones y soporta modelos guardados en models/.

## Scripts útiles

- scripts/probar_modelo.py: inferencia sobre archivos existentes.
- scripts/verificar_sistema.py: valida estructura de data/raw/ y carga básica.
- src/predictor.py: normalización y evaluación rápida desde CSV.

## Configuración

src/config.py expone:
- Arquitectura (NETWORK_CONFIG['capas'], activaciones, tasa de aprendizaje, dropout, L2, hiperparámetros de Adam).
- Parámetros de datos (DATA_CONFIG): tamaño, augmentación, splits, semilla reproducible.
- Etiquetas personalizadas (CUSTOM_LABELS).

## Resultados de Referencia

Con las 520 imágenes sintéticas generadas automáticamente:
- 	train accuracy ≈ 0.998
-  val accuracy ≈ 0.63 (dataset sintético pequeño, notarás varianza)
- 	test accuracy ≈ 0.75
Estas cifras mejoran al añadir ejemplos reales representativos.

## Próximos pasos recomendados

- Recolectar muestras reales con diferentes fuentes/texturas para las clases con peor desempeño (scripts/run_pipeline.py --confusion-report).
- Ajustar hiperparámetros (dropout_rate, lambda_l2, capas) según la complejidad de tus datos.
- Si lo deseas, integra una CNN o normalización por lotes partiendo del esqueleto de src/network.py.

## Licencia

MIT © ZaoGabo