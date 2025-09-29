# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
y este proyecto sigue [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased - v1.0.0] - Próxima versión estable

### 🔧 Por Corregir
- Optimización de la precisión del modelo con caracteres similares
- Mejoras en el preprocesamiento de imágenes  
- Correcciones menores en la interfaz web
- Validación adicional de entrada de datos
- Optimización de rendimiento en entrenamiento
- Eliminación de warnings de deprecación
- Tests automatizados

## [0.9.0] - 2025-09-28 - **VERSIÓN EXPERIMENTAL**

### ⚠️ Estado
**PROYECTO EN FASE DE DESARROLLO Y EXPERIMENTACIÓN**
- Funcionalidades principales implementadas pero con correcciones pendientes
- Recomendado solo para propósitos educativos y de prueba
- Se están realizando correcciones continuas

### ✨ Agregado
- Implementación completa de red neuronal desde cero con NumPy
- Reconocimiento de 52 caracteres (A-Z mayúsculas y a-z minúsculas)
- Arquitectura de 3 capas: 784 → 256 → 52 neuronas
- Aplicación web interactiva con Streamlit
- Múltiples opciones de entrenamiento:
  - Entrenamiento básico con datos sintéticos
  - Entrenamiento mejorado con 5200 imágenes
  - Entrenamiento incremental
  - Entrenamiento con imágenes reales
- Estructura de datos organizada para 52 clases
- Preprocesamiento automático de imágenes
- Visualización de confianza y top 5 predicciones
- Generación sintética de imágenes de entrenamiento
- Augmentación de datos (rotación, ruido, blur)
- Múltiples fuentes y estilos de texto

### 🎯 Rendimiento (En Pruebas)
- Precisión del 100% en datos sintéticos (métricas experimentales)
- Precisión >95% en datos reales (puede variar)
- Tiempo de predicción <1ms
- Modelo compacto <1MB

### 🐛 Problemas Conocidos
- Confusión ocasional entre caracteres similares ('o', 'O', '0')
- Warnings de deprecación en la interfaz web (no afectan funcionalidad)
- Compatibilidad limitada con algunas versiones de NumPy
- Resultados imprecisos con imágenes muy pequeñas o borrosas

### 📁 Estructura del Proyecto
- `src/` - Código fuente principal
- `demo/` - Aplicación web (experimental)
- `data/` - 52 carpetas organizadas para entrenamiento
- `models/` - Modelos entrenados
- Scripts de entrenamiento especializados

### 🔧 Configuración
- Archivo `.gitignore` completo para proyectos de ML
- `requirements.txt` con dependencias optimizadas
- `README.md` profesional con avisos experimentales
- Licencia MIT

### 🌐 Interfaz Web (Beta)
- Subida de imágenes drag & drop
- Visualización en tiempo real del preprocesamiento
- Métricas de confianza
- Top 5 predicciones con barras de progreso
- Información del modelo
- Consejos para el usuario
- ⚠️ Algunos warnings de deprecación pendientes de corrección

### 🧠 Algoritmo (En Desarrollo)
- Backpropagation implementado desde cero
- Función de activación sigmoid
- Inicialización de pesos optimizada
- Tasa de aprendizaje adaptativa
- Entrenamiento por lotes
- 🔧 Optimizaciones pendientes para caracteres similares

---

## Hoja de ruta de desarrollo

### [1.0.0] - Próxima versión estable
- [x] ~~Implementación básica completa~~
- [ ] Corrección de problemas conocidos
- [ ] Validación exhaustiva del sistema
- [ ] Tests automatizados
- [ ] Documentación finalizada
- [ ] Optimización de precisión
- [ ] Eliminación de warnings

### [1.1.0] - Funcionalidades adicionales
- [ ] Soporte para números (0-9)
- [ ] API REST para predicciones
- [ ] Modo de entrenamiento online
- [ ] Optimizaciones de rendimiento

### [1.2.0] - Expansión
- [ ] Soporte para símbolos especiales
- [ ] Modelos pre-entrenados
- [ ] Soporte para GPU con CuPy
- [ ] Interfaz de línea de comandos

### [2.0.0] - Arquitectura avanzada
- [ ] Arquitecturas de red más avanzadas
- [ ] Redes convolucionales
- [ ] Transfer learning
- [ ] Deployment en la nube