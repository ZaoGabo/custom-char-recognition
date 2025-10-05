"""
Carga y preprocesamiento de datos.
"""

import os
try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None
from .config import DATA_CONFIG
from .label_map import LabelMap, DEFAULT_LABEL_MAP
from .utils import normalize_image, apply_augmentation

def _leer_imagen_gris(ruta_imagen: str, tamano: Tuple[int, int]) -> np.ndarray:
    alto, ancho = tamano
    if cv2 is not None:
        img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen {ruta_imagen}")
        return cv2.resize(img, (ancho, alto))
    img = Image.open(ruta_imagen).convert('L')
    img = img.resize((ancho, alto))
    return np.array(img, dtype=np.uint8)


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
        Cargar imÃƒÆ’Ã‚Â¡genes desde estructura de directorios.
        Estructura esperada: ruta_datos/clase/imagen.png
        
        Args:
            tamano_imagen (tuple): TamaÃƒÆ’Ã‚Â±o de imagen (altura, ancho)
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
                    img = _leer_imagen_gris(ruta_imagen, tamano_imagen)
                    imagenes.append(img)
                    etiquetas.append(indice_clase)
                except Exception as e:
                    print(f"Error cargando {ruta_imagen}: {e}")
                    
        self.imagenes = np.array(imagenes)
        self.etiquetas = np.array(etiquetas)
        
        print(f"Cargadas {len(self.imagenes)} imÃƒÆ’Ã‚Â¡genes de {len(np.unique(self.etiquetas))} clases")
        
    def cargar_desde_csv(self, ruta_csv, columna_imagen='image_path', columna_etiqueta='label'):
        """
        Cargar datos desde archivo CSV.
        
        Args:
            ruta_csv (str): Ruta al archivo CSV
            columna_imagen (str): Nombre de la columna con rutas de imÃƒÆ’Ã‚Â¡genes
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
                img = _leer_imagen_gris(ruta_imagen, DATA_CONFIG['tamano_imagen'])
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
        """Preprocesar las imÃƒÆ’Ã‚Â¡genes cargadas."""
        if DATA_CONFIG['normalizar']:
            self.imagenes = normalize_image(self.imagenes)
            
        self.imagenes = self.imagenes.reshape(self.imagenes.shape[0], -1)
        

    def _dividir_manual(self, proporcion_entrenamiento, proporcion_validacion, semilla):
        rng = np.random.default_rng(semilla)
        val_ratio = proporcion_validacion / (proporcion_validacion + DATA_CONFIG['division_prueba'])
        train_idx = []
        val_idx = []
        test_idx = []

        for clase in np.unique(self.etiquetas):
            indices = np.where(self.etiquetas == clase)[0]
            rng.shuffle(indices)
            n = len(indices)
            n_train = int(np.floor(proporcion_entrenamiento * n))
            n_rest = n - n_train
            n_val = int(np.floor(val_ratio * n_rest))
            n_test = n - n_train - n_val

            train_idx.extend(indices[:n_train])
            val_idx.extend(indices[n_train:n_train + n_val])
            test_idx.extend(indices[n_train + n_val:n_train + n_val + n_test])

        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)

        train_idx = np.array(train_idx, dtype=int)
        val_idx = np.array(val_idx, dtype=int)
        test_idx = np.array(test_idx, dtype=int)

        return (
            self.imagenes[train_idx],
            self.imagenes[val_idx],
            self.imagenes[test_idx],
            self.etiquetas[train_idx],
            self.etiquetas[val_idx],
            self.etiquetas[test_idx],
        )

    def dividir_datos(self, proporcion_entrenamiento=None, proporcion_validacion=None, semilla=42):
        """
        Dividir datos en entrenamiento, validacion y prueba.

        Args:
            proporcion_entrenamiento (float): Proporcion para entrenamiento
            proporcion_validacion (float): Proporcion para validacion
            semilla (int): Semilla para reproducibilidad

        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if proporcion_entrenamiento is None:
            proporcion_entrenamiento = DATA_CONFIG['division_entrenamiento']
        if proporcion_validacion is None:
            proporcion_validacion = DATA_CONFIG['division_validacion']

        if train_test_split is None:
            return self._dividir_manual(proporcion_entrenamiento, proporcion_validacion, semilla)

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
        """Aplicar augmentacion de datos."""
        if not DATA_CONFIG['usar_augmentacion']:
            return X_train, y_train

        imagenes_aug = []
        etiquetas_aug = []

        for i in range(len(X_train)):
            imagen_original = X_train[i]
            etiqueta = y_train[i]

            imagenes_aug.append(imagen_original)
            etiquetas_aug.append(etiqueta)

            for _ in range(factor - 1):
                img_2d = imagen_original.reshape(DATA_CONFIG['tamano_imagen'])
                img_aug = self._augmentacion_simple(img_2d)
                img_flat = img_aug.flatten()
                if len(img_flat) == len(imagen_original):
                    imagenes_aug.append(img_flat)
                    etiquetas_aug.append(etiqueta)

        total_original = len(X_train)
        total_augmentado = len(imagenes_aug)
        print(f"Augmentacion completa: {total_original} -> {total_augmentado} imagenes")

        return np.array(imagenes_aug), np.array(etiquetas_aug)

    def _augmentacion_simple(self, imagen_2d):
        """
        Aplicar augmentaciÃƒÆ’Ã‚Â³n simple a una imagen 2D.
        
        Args:
            imagen_2d: Imagen en formato 2D (altura, ancho)
            
        Returns:
            np.array: Imagen augmentada del mismo tamaÃƒÆ’Ã‚Â±o
        """
        img = imagen_2d.copy().astype(np.float32)
        
        # Agregar ruido gaussiano suave
        if np.random.random() > 0.5:
            ruido = np.random.normal(0, 0.05, img.shape)
            img = np.clip(img + ruido, 0, 1)
        
        # Desplazamiento pequeÃƒÆ’Ã‚Â±o (circular)
        if np.random.random() > 0.5:
            shift_x = np.random.randint(-2, 3)
            shift_y = np.random.randint(-2, 3)
            img = np.roll(img, shift_x, axis=1)
            img = np.roll(img, shift_y, axis=0)
        
        # Multiplicar por un factor aleatorio pequeÃƒÆ’Ã‚Â±o
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
