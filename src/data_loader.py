"""
Carga y preprocesamiento de datos.
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from .config import DATA_CONFIG
from .label_map import LabelMap, DEFAULT_LABEL_MAP
from .utils import normalize_image, apply_augmentation

class DataLoader:
    def __init__(self, ruta_datos, mapa_etiquetas=None):
        """
        Inicializar el cargador de datos.
        
        Args:
            ruta_datos (str): Ruta a los datos
            mapa_etiquetas (LabelMap): Mapeo de etiquetas
        """
        self.ruta_datos = ruta_datos
        self.mapa_etiquetas = mapa_etiquetas or DEFAULT_LABEL_MAP
        self.imagenes = []
        self.etiquetas = []
    
    def _mapear_carpeta_a_etiqueta(self, nombre_carpeta):
        """
        Mapear nombre de carpeta con sufijo a etiqueta real.
        
        Args:
            nombre_carpeta (str): Nombre de la carpeta (ej: 'A_upper', 'a_lower')
            
        Returns:
            str: Etiqueta real (ej: 'A', 'a') o None si no se puede mapear
        """
        if nombre_carpeta.endswith('_upper'):
            return nombre_carpeta[0].upper()
        elif nombre_carpeta.endswith('_lower'):
            return nombre_carpeta[0].lower()
        else:
            # Si no tiene sufijo, asumir que es el nombre directo
            return nombre_carpeta
        
    def cargar_desde_directorio(self, tamano_imagen=None):
        """
        Cargar imágenes desde estructura de directorios.
        Estructura esperada: ruta_datos/clase/imagen.png
        
        Args:
            tamano_imagen (tuple): Tamaño de imagen (altura, ancho)
        """
        if tamano_imagen is None:
            tamano_imagen = DATA_CONFIG['tamano_imagen']
            
        imagenes = []
        etiquetas = []
        
        for nombre_clase in os.listdir(self.ruta_datos):
            ruta_clase = os.path.join(self.ruta_datos, nombre_clase)
            if not os.path.isdir(ruta_clase):
                continue
            
            # Mapear nombre de carpeta a etiqueta real
            etiqueta_real = self._mapear_carpeta_a_etiqueta(nombre_clase)
            if etiqueta_real is None:
                print(f"Advertencia: No se pudo mapear la carpeta '{nombre_clase}' a una etiqueta")
                continue
                
            indice_clase = self.mapa_etiquetas.get_index(etiqueta_real)
            if indice_clase == -1:
                print(f"Advertencia: Etiqueta '{etiqueta_real}' no encontrada en mapa_etiquetas")
                continue
                
            for archivo_imagen in os.listdir(ruta_clase):
                if not archivo_imagen.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                ruta_imagen = os.path.join(ruta_clase, archivo_imagen)
                try:
                    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, tamano_imagen)
                    imagenes.append(img)
                    etiquetas.append(indice_clase)
                except Exception as e:
                    print(f"Error cargando {ruta_imagen}: {e}")
                    
        self.imagenes = np.array(imagenes)
        self.etiquetas = np.array(etiquetas)
        
        print(f"Cargadas {len(self.imagenes)} imágenes de {len(np.unique(self.etiquetas))} clases")
        
    def cargar_desde_csv(self, ruta_csv, columna_imagen='image_path', columna_etiqueta='label'):
        """
        Cargar datos desde archivo CSV.
        
        Args:
            ruta_csv (str): Ruta al archivo CSV
            columna_imagen (str): Nombre de la columna con rutas de imágenes
            columna_etiqueta (str): Nombre de la columna con etiquetas
        """
        df = pd.read_csv(ruta_csv)
        imagenes = []
        etiquetas = []
        
        for _, fila in df.iterrows():
            ruta_imagen = fila[columna_imagen]
            etiqueta = fila[columna_etiqueta]
            
            if not os.path.isabs(ruta_imagen):
                ruta_imagen = os.path.join(os.path.dirname(ruta_csv), ruta_imagen)
                
            try:
                img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, DATA_CONFIG['tamano_imagen'])
                imagenes.append(img)
                
                indice = self.mapa_etiquetas.get_index(etiqueta)
                if indice == -1:
                    print(f"Advertencia: Etiqueta '{etiqueta}' no encontrada")
                    continue
                etiquetas.append(indice)
                
            except Exception as e:
                print(f"Error cargando {ruta_imagen}: {e}")
                
        self.imagenes = np.array(imagenes)
        self.etiquetas = np.array(etiquetas)
        
    def preprocesar_imagenes(self):
        """Preprocesar las imágenes cargadas."""
        if DATA_CONFIG['normalizar']:
            self.imagenes = normalize_image(self.imagenes)
            
        self.imagenes = self.imagenes.reshape(self.imagenes.shape[0], -1)
        
    def dividir_datos(self, proporcion_entrenamiento=None, proporcion_validacion=None, semilla=42):
        """
        Dividir datos en entrenamiento, validación y prueba.
        
        Args:
            proporcion_entrenamiento (float): Proporción para entrenamiento
            proporcion_validacion (float): Proporción para validación
            semilla (int): Semilla para reproducibilidad
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if proporcion_entrenamiento is None:
            proporcion_entrenamiento = DATA_CONFIG['division_entrenamiento']
        if proporcion_validacion is None:
            proporcion_validacion = DATA_CONFIG['division_validacion']
            
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.imagenes, self.etiquetas, 
            train_size=proporcion_entrenamiento, 
            random_state=semilla,
            stratify=self.etiquetas
        )
        
        proporcion_val = proporcion_validacion / (proporcion_validacion + DATA_CONFIG['division_prueba'])
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            train_size=proporcion_val,
            random_state=semilla,
            stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def aplicar_augmentacion(self, X_train, y_train, factor=2):
        """
        Aplicar augmentación de datos.
        
        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento  
            factor: Cuántas veces duplicar cada imagen
            
        Returns:
            tuple: (X_augmented, y_augmented)
        """
        if not DATA_CONFIG['usar_augmentacion']:
            return X_train, y_train
        
        print(f"🔄 Aplicando augmentación con factor {factor}...")
        
        imagenes_aug = []
        etiquetas_aug = []
        
        for i in range(len(X_train)):
            # Imagen original (ya aplanada)
            imagen_original = X_train[i]
            etiqueta = y_train[i]
            
            # Agregar imagen original
            imagenes_aug.append(imagen_original)
            etiquetas_aug.append(etiqueta)
            
            # Generar versiones augmentadas
            for _ in range(factor - 1):
                # Reshape para augmentación (convertir de 1D a 2D)
                img_2d = imagen_original.reshape(DATA_CONFIG['tamano_imagen'])
                
                # Aplicar augmentación simple
                img_aug = self._augmentacion_simple(img_2d)
                
                # Volver a aplanar y asegurar que tiene el mismo tamaño
                img_flat = img_aug.flatten()
                if len(img_flat) == len(imagen_original):
                    imagenes_aug.append(img_flat)
                    etiquetas_aug.append(etiqueta)
        
        total_original = len(X_train)
        total_augmentado = len(imagenes_aug)
        print(f"✅ Augmentación completa: {total_original} → {total_augmentado} imágenes")
        
        return np.array(imagenes_aug), np.array(etiquetas_aug)
    
    def _augmentacion_simple(self, imagen_2d):
        """
        Aplicar augmentación simple a una imagen 2D.
        
        Args:
            imagen_2d: Imagen en formato 2D (altura, ancho)
            
        Returns:
            np.array: Imagen augmentada del mismo tamaño
        """
        img = imagen_2d.copy().astype(np.float32)
        
        # Agregar ruido gaussiano suave
        if np.random.random() > 0.5:
            ruido = np.random.normal(0, 0.05, img.shape)
            img = np.clip(img + ruido, 0, 1)
        
        # Desplazamiento pequeño (circular)
        if np.random.random() > 0.5:
            shift_x = np.random.randint(-2, 3)
            shift_y = np.random.randint(-2, 3)
            img = np.roll(img, shift_x, axis=1)
            img = np.roll(img, shift_y, axis=0)
        
        # Multiplicar por un factor aleatorio pequeño
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.9, 1.1)
            img = np.clip(img * factor, 0, 1)
        
        return img
        
    def obtener_pesos_clase(self):
        """Calcular pesos de clases para datos desbalanceados."""
        from sklearn.utils.class_weight import compute_class_weight
        
        clases = np.unique(self.etiquetas)
        pesos = compute_class_weight('balanced', classes=clases, y=self.etiquetas)
        return dict(zip(clases, pesos))
