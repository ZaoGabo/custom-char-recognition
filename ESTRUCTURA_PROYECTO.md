# 📁 Estructura del Proyecto - Custom Character Recognition

## ✅ Archivos y Carpetas en Uso

### 🎯 **Core del Proyecto**
```
src/
├── __init__.py              # Inicialización del paquete
├── config.py                # Configuración principal
├── data_loader.py           # ✅ Carga imágenes de data/raw/
├── label_map.py             # Mapeo de etiquetas
├── network.py               # ✅ Red neuronal (arquitectura)
├── predictor.py             # ✅ Predicciones del modelo
├── trainer.py               # ✅ Entrenamiento del modelo
└── utils.py                 # Utilidades generales
```

### 📊 **Datos**
```
data/
├── raw/                     # ✅ 62,000 imágenes EMNIST (1,000 por clase × 62)
│   ├── 0-9_digit/           # Números (0-9)
│   ├── A-Z_upper/           # Mayúsculas (A-Z)
│   └── a-z_lower/           # Minúsculas (a-z)
└── processed/               # Datos procesados (vacío)
```

**Nombres de archivo:** `emnist_A_upper_00000.png`, `emnist_0_digit_00001.png`, etc.

### 🤖 **Modelo Entrenado**
```
models/
└── modelo_entrenado/        # ✅ Modelo 78.09% precisión
    ├── arquitectura.json    # Estructura de la red
    ├── pesos_0-4.npy        # Pesos de cada capa
    ├── sesgos_0-4.npy       # Sesgos de cada capa
    └── batch_norm_*.npy     # Parámetros BatchNorm
```

### 🎨 **Demo Streamlit**
```
demo/
├── app_canvas.py            # ✅ App principal (dibuja y predice)
├── app.py                   # App alternativa (texto)
└── modelo.py                # Cargador del modelo
```

**Ejecutar:** `streamlit run demo\app_canvas.py`

### 🧪 **Scripts de Prueba**
```
scripts/
├── probar_modelo.py         # ✅ Evaluar modelo (78% accuracy)
├── run_pipeline.py          # Pipeline completo
└── verificar_sistema.py     # Verificar dependencias
```

### 📝 **Configuración**
```
config.yml                   # ✅ Configuración YAML (93 clases)
requirements.txt             # ✅ Dependencias Python
.gitignore                   # ✅ Exclusiones de Git
```

---

## ❌ Archivos Eliminados (Obsoletos)

```
❌ src/scripts/generar_imagenes_sinteticas.py  # Generaba Arial (fallaba)
❌ src/scripts/descargar_kaggle_az.py         # Script Kaggle (no usado)
❌ src/scripts/descargar_emnist.py            # Ya descargado
❌ INSTRUCCIONES_KAGGLE.md                    # Instrucciones Kaggle
❌ train_con_imagenes_reales.py               # Duplicado de src.trainer
```

---

## 🚀 Comandos Principales

### **1. Entrenar Modelo**
```powershell
$env:PYTHONPATH="$PWD"
python -m src.trainer --force --verbose
```

### **2. Probar Modelo**
```powershell
python scripts\probar_modelo.py
```

### **3. Ejecutar Demo**
```powershell
streamlit run demo\app_canvas.py
```

### **4. Validación Cruzada**
```powershell
python scripts\cross_validation.py --k-folds 5 --verbose
```

---

## 📈 Resultados Actuales

- **Dataset:** 62,000 imágenes EMNIST (escritura a mano real)
- **Clases:** 62 (0-9, A-Z, a-z)
- **Precisión de Test:** 78.09%
- **Precisión de Validación:** 74.98%
- **Modelo:** Red neuronal [784, 512, 384, 256, 128, 62] con BatchNorm

---

## 🔧 Configuración de .gitignore

Los siguientes archivos/carpetas NO se suben a GitHub:

```
✅ models/modelo_entrenado/      # Modelo entrenado (muy pesado)
✅ data/raw/**/*.png              # 62,000 imágenes (muy pesado)
✅ src/scripts/descargar_*.py     # Scripts de descarga (locales)
✅ __pycache__/                   # Caché de Python
✅ .venv/                         # Entorno virtual
✅ .kaggle/                       # Credenciales Kaggle
```

---

## 📦 Dependencias Principales

```
numpy==1.26.4
Pillow==10.4.0
streamlit==1.50.0
streamlit-drawable-canvas==0.9.3
PyYAML==6.0.1
torch==2.9.0+cpu              # Solo para descargar datasets
torchvision==0.24.0+cpu       # Solo para descargar datasets
```

---

## 📚 Documentación

- `README.md` - Documentación principal del proyecto
- `README.en.md` - Documentación en inglés
- `docs/EXTENSIONES.md` - Guía de extensiones

---

**Última actualización:** 19 de octubre de 2025
**Modelo actual:** EMNIST 78.09% (escritura a mano real)
