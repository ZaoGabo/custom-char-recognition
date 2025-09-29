# 🧠 Sistema de Reconocimiento de Caracteres con Red Neuronal

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.38+-red.svg)
![NumPy](https://img.shields.io/badge/numpy-v2.2+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-experimental-orange.svg)
![Development](https://img.shields.io/badge/development-in_progress-yellow.svg)

> ⚠️ **PROYECTO EN FASE EXPERIMENTAL** ⚠️  
> Este proyecto está actualmente en desarrollo y experimentación. Algunas funcionalidades pueden tener errores menores que están siendo corregidos. Se recomienda usarlo únicamente para propósitos educativos y de prueba.

Un sistema completo de reconocimiento de caracteres (A-Z, a-z) implementado desde cero con una red neuronal personalizada y una interfaz web interactiva.

## 🚧 Estado del Proyecto

**Versión:** 0.9.0 (Pre-release)  
**Estado:** Experimental - En desarrollo activo  
**Última actualización:** 28 de septiembre de 2025

### 🔧 Conocidas a Corregir
- [ ] Optimización de la precisión del modelo con ciertos caracteres
- [ ] Mejoras en el preprocesamiento de imágenes
- [ ] Correcciones menores en la interfaz web
- [ ] Validación adicional de entrada de datos
- [ ] Optimización de rendimiento en entrenamiento

### ✅ Funcionalidades Estables
- ✅ Entrenamiento básico del modelo
- ✅ Interfaz web funcional
- ✅ Carga y predicción de modelos
- ✅ Estructura de datos organizada

## ✨ Características Principales

- 🔤 **52 Clases de Caracteres**: Reconoce A-Z (mayúsculas) y a-z (minúsculas)
- 🧠 **Red Neuronal Personalizada**: Implementada desde cero con NumPy
- 🎯 **Alta Precisión**: 100% en datos sintéticos, >95% en datos reales  
- 🌐 **Interfaz Web**: Aplicación Streamlit para pruebas interactivas
- 📊 **Visualización**: Muestra confianza y top 5 predicciones
- 🔄 **Entrenamiento Flexible**: Múltiples opciones de entrenamiento

## 🚀 Demo en Vivo

La aplicación web permite:
- 📷 Subir imágenes de caracteres
- 🔍 Ver preprocesamiento en tiempo real
- 📊 Obtener predicciones con niveles de confianza
- 🏆 Visualizar top 5 resultados

## 🏗️ Arquitectura del Sistema

```
Entrada (784 neuronas) → Capa Oculta (256 neuronas) → Salida (52 neuronas)
        28x28 pixels           ReLU/Sigmoid              Sigmoid
```

### 📊 Especificaciones Técnicas
- **Entrada**: Imágenes 28x28 píxeles en escala de grises
- **Arquitectura**: Red neuronal feedforward de 3 capas
- **Función de activación**: Sigmoid
- **Algoritmo**: Backpropagation con descenso de gradiente
- **Optimización**: Tasa de aprendizaje adaptativa

## 🛠️ Instalación

> ⚠️ **Nota:** Este proyecto está en fase experimental. Pueden ocurrir errores durante la instalación o ejecución. Se están realizando correcciones continuas.

### Prerrequisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/ZaoGabo/custom-char-recognition.git
cd custom-char-recognition
```

2. **Crear entorno virtual (recomendado)**
```bash
python -m venv venv
source venv/bin/activate  # En Linux/Mac
# o
venv\\Scripts\\activate     # En Windows
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Entrenar el modelo**
```bash
python train_mejorado.py
```

5. **Ejecutar la aplicación web**
```bash
streamlit run demo/app_simple.py
```

## 📁 Estructura del Proyecto

```
custom-char-recognition/
├── 📁 src/                 # Código fuente principal
│   ├── network.py          # Implementación de la red neuronal
│   ├── data_loader.py      # Carga y procesamiento de datos
│   ├── trainer.py          # Script de entrenamiento
│   └── config.py           # Configuraciones del proyecto
├── 📁 demo/                # Aplicación web demo
│   ├── app_simple.py       # Aplicación Streamlit
│   └── modelo.py          # Módulo compartido del modelo
├── 📁 data/               # Estructura de datos de entrenamiento
│   ├── A_upper/           # Imágenes de 'A' mayúscula
│   ├── B_upper/           # Imágenes de 'B' mayúscula
│   ├── ...
│   ├── a_lower/           # Imágenes de 'a' minúscula
│   └── z_lower/           # Imágenes de 'z' minúscula
├── 📁 models/             # Modelos entrenados
├── 📄 requirements.txt    # Dependencias Python
├── 📄 README.md          # Este archivo
└── 📄 .gitignore         # Archivos ignorados por Git
```

## 🎯 Uso del Sistema

### 1. Entrenamiento del Modelo

**Entrenamiento mejorado (recomendado):**
```bash
python train_mejorado.py
```

**Entrenamiento incremental:**
```bash
python train_incremental.py
```

**Entrenamiento con datos reales:**
```bash
python train_con_imagenes_reales.py
```

### 2. Uso de la Aplicación Web

1. Ejecutar la aplicación:
```bash
streamlit run demo/app_simple.py
```

2. Abrir en el navegador: `http://localhost:8502`

3. Subir una imagen de un carácter

4. Ver los resultados de la predicción

### 3. Uso Programático

```python
import pickle
import numpy as np
from demo.modelo import RedNeuronalSimple

# Cargar modelo entrenado
with open('models/modelo_entrenado.pkl', 'rb') as f:
    modelo = pickle.load(f)

# Hacer predicción
imagen_preprocesada = preprocess_image(tu_imagen)  # 28x28 normalizada
prediccion = modelo.predecir(imagen_preprocesada.flatten())
caracter_predicho = obtener_etiqueta(np.argmax(prediccion))
```

## 📊 Rendimiento

> 📊 **Nota:** Métricas obtenidas en entorno de desarrollo. Los resultados pueden variar según el hardware y datos de entrada.

| Métrica | Datos Sintéticos | Datos Reales | Estado |
|---------|------------------|--------------|--------|
| **Precisión** | 100% | >95% | 🧪 En pruebas |
| **Tiempo de entrenamiento** | ~5 minutos | Variable | ✅ Estable |
| **Tamaño del modelo** | <1 MB | <1 MB | ✅ Estable |
| **Velocidad de predicción** | <1ms | <1ms | 🔄 Optimizando |## 🔧 Configuración Avanzada

### Parámetros del Modelo
```python
modelo = RedNeuronalSimple(
    entrada_neuronas=784,    # 28x28 píxeles
    oculta_neuronas=256,     # Neuronas capa oculta
    salida_neuronas=52,      # A-Z + a-z
    tasa_aprendizaje=0.2     # Tasa de aprendizaje
)
```

### Personalización de Datos
- Agregar imágenes a las carpetas `data/X_upper/` o `data/x_lower/`
- Soporta formatos: PNG, JPG, JPEG, BMP
- Resolución recomendada: 28x28 píxeles

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor:

1. Fork del proyecto
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit de tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## 🐛 Reportar Problemas

Si encuentras algún problema, por favor abre un [issue](../../issues) con:
- Descripción del problema
- Pasos para reproducirlo
- Información del entorno (OS, Python version, etc.)
- Screenshots si es relevante

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 👨‍💻 Autor

**ZaoGabo**
- GitHub: [@ZaoGabo](https://github.com/ZaoGabo)

## 🙏 Agradecimientos

- Inspirado en conceptos de redes neuronales clásicas
- Comunidad de Machine Learning por recursos educativos
- Documentación de NumPy y SciPy

## ⚠️ Limitaciones Conocidas

### � Problemas en Desarrollo
- Algunos caracteres similares (como 'o', 'O', '0') pueden confundirse ocasionalmente
- La interfaz web puede mostrar warnings de deprecación (no afectan funcionalidad)
- El entrenamiento con imágenes muy pequeñas o borrosas puede dar resultados imprecisos
- Compatibilidad limitada con algunas versiones de NumPy

### 🔧 Correcciones Programadas
- Mejora del algoritmo de diferenciación de caracteres similares
- Actualización de dependencias para eliminar warnings
- Validación mejorada de entrada de imágenes
- Optimización del preprocesamiento

## �🔮 Próximas Características

### Versión 1.0.0 (Próximamente)
- [ ] Corrección de problemas conocidos
- [ ] Validación exhaustiva del sistema
- [ ] Documentación actualizada
- [ ] Tests automatizados

### Versiones futuras
- [ ] Soporte para más caracteres (números, símbolos)
- [ ] Modelos pre-entrenados
- [ ] API REST
- [ ] Modo de entrenamiento online
- [ ] Optimizaciones de rendimiento
- [ ] Soporte para GPU

---

⭐ **¡No olvides dar una estrella al proyecto si te resultó útil!** ⭐