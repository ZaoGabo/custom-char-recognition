# Guía de Organización de Datos

## Estructura de Carpetas Creadas

Se han creado automáticamente **52 carpetas** en `data/raw/` para organizar las imágenes de entrenamiento:

### 📁 Letras Mayúsculas (26 carpetas)
- `A_upper/`, `B_upper/`, `C_upper/`, ..., `Z_upper/`

### 📁 Letras Minúsculas (26 carpetas)  
- `a_lower/`, `b_lower/`, `c_lower/`, ..., `z_lower/`

## 🎯 Cómo Usar

### 1. **Organizar las Imágenes**
Coloque las imágenes de cada carácter en su carpeta correspondiente:

```
data/raw/
├── A_upper/          # Imágenes de la letra A mayúscula
│   ├── imagen1.png
│   ├── imagen2.jpg
│   └── imagen3.jpeg
├── B_upper/          # Imágenes de la letra B mayúscula
│   └── ...
├── a_lower/          # Imágenes de la letra a minúscula
│   ├── imagen1.png
│   └── imagen2.jpg
└── ...
```

### 2. **Formatos Soportados**
- ✅ PNG (.png)
- ✅ JPEG (.jpg, .jpeg)
- ✅ BMP (.bmp)

### 3. **Recomendaciones de Imágenes**
- **Calidad**: Imágenes claras y nítidas
- **Contraste**: Buen contraste entre el carácter y el fondo
- **Tamaño**: Mínimo 28x28 píxeles (se redimensionarán automáticamente)
- **Cantidad**: Al menos 10-20 imágenes por clase para mejores resultados
- **Variedad**: Incluir diferentes estilos, fuentes, y condiciones

### 4. **Ejemplo de Nombres de Archivo**
```
A_upper/
├── A_arial_1.png
├── A_times_2.jpg
├── A_handwritten_3.png
└── A_bold_4.jpeg
```

## 🔧 Configuración Automática

El sistema está configurado para:
- **52 clases**: A-Z (mayúsculas) + a-z (minúsculas)
- **Mapeo automático**: Las carpetas con sufijos se mapean correctamente
- **Preprocesamiento**: Redimensionado automático a 28x28 píxeles
- **Normalización**: Valores de píxeles normalizados entre 0 y 1

## 🚀 Siguiente Paso

Una vez que tenga las imágenes organizadas, ejecute:

```bash
python src/trainer.py
```

El sistema cargará automáticamente todas las imágenes, las procesará y entrenará el modelo de reconocimiento de caracteres.

## 📊 Estadísticas Esperadas

Con la estructura actual, el sistema puede:
- Distinguir entre **52 tipos diferentes** de caracteres
- Reconocer tanto **mayúsculas** como **minúsculas**
- Procesar **múltiples formatos** de imagen
- Aplicar **augmentación de datos** automáticamente