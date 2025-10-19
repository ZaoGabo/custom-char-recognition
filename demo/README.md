# Aplicaciones Demo - Reconocimiento de Caracteres

Este proyecto incluye 2 aplicaciones Streamlit con diferentes características:

## 📱 Aplicaciones disponibles:

### 1. `app_sin_pyarrow.py` ⭐ (RECOMENDADA - Actualmente en uso)
**Aplicación sin dependencias de PyArrow**

**Características:**
- ✅ Subida de imágenes desde archivos
- ✅ Canvas HTML básico para dibujar
- ✅ NO requiere PyArrow ni componentes adicionales
- ✅ Funciona en cualquier entorno

**Cómo ejecutar:**
```cmd
.venv\Scripts\python.exe -m streamlit run demo\app_sin_pyarrow.py
```

**Uso:**
1. Ve a http://localhost:8501
2. Sube una imagen de un carácter, o
3. Dibuja en el canvas HTML, descarga y sube la imagen

---

### 2. `app_canvas.py` (Requiere instalación adicional)
**Aplicación con canvas interactivo avanzado**

**Características:**
- ✅ Canvas interactivo nativo (dibujas directo)
- ✅ Predicción inmediata sin subir archivos
- ✅ Interfaz más intuitiva
- ❌ **REQUIERE PyArrow instalado**
- ❌ **REQUIERE streamlit-drawable-canvas**

**Cómo ejecutar:**
```cmd
# Primero instalar PyArrow (con conda recomendado):
conda install -c conda-forge pyarrow

# Luego ejecutar:
.venv\Scripts\python.exe -m streamlit run demo\app_canvas.py
```

**Uso:**
1. Ve a http://localhost:8501
2. Dibuja directamente en el canvas negro
3. Presiona "Predecir"

---

## 🎯 ¿Cuál usar?

- **Para uso general y desarrollo:** `app_sin_pyarrow.py` (actual)
- **Para demos profesionales:** `app_canvas.py` (si instalas PyArrow con conda)

---

## 📊 Precisión actual del modelo:
- **91.41%** de precisión en validación
- Reconoce A-Z mayúsculas y a-z minúsculas (52 clases)

---

## 🔧 Mejoras implementadas:
1. ✅ BatchNormalization
2. ✅ Early Stopping
3. ✅ Learning Rate Scheduling
4. ✅ Data Augmentation optimizado

