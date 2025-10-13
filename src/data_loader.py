"""Data loading and preprocessing utilities for large datasets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Iterator

import numpy as np
import pandas as pd
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None

from .config import DATA_CONFIG
from .label_map import LabelMap, DEFAULT_LABEL_MAP
from .utils import apply_augmentation, normalize_image

CV2_IMREAD_GRAYSCALE = getattr(cv2, "IMREAD_GRAYSCALE", 0) if cv2 is not None else 0


def _leer_imagen_gris(ruta_imagen: str, tamano: Tuple[int, int]) -> np.ndarray:
    """Return a grayscale image resized to ``tamano``."""
    alto, ancho = tamano
    if cv2 is not None:
        imagen = cv2.imread(ruta_imagen, CV2_IMREAD_GRAYSCALE)
        if imagen is None:
            raise ValueError(f"No se pudo cargar la imagen {ruta_imagen}")
        return cv2.resize(imagen, (ancho, alto))

    with Image.open(ruta_imagen) as img_pil:
        imagen = img_pil.convert("L").resize((ancho, alto))
    return np.array(imagen, dtype=np.uint8)


class DataLoader:
    """Carga y preprocesamiento de datos para la red neuronal."""

    def __init__(self, ruta_datos: str, mapa_etiquetas: Optional[LabelMap] = None) -> None:
        self.ruta_datos = ruta_datos
        self.mapa_etiquetas = mapa_etiquetas or DEFAULT_LABEL_MAP
        self.rutas_imagenes: List[str] = []
        self.etiquetas: List[int] = []

    def _mapear_carpeta_a_etiqueta(self, nombre_carpeta: str) -> str:
        """Convertir un nombre de carpeta a la etiqueta real."""
        if nombre_carpeta.endswith("_upper"):
            return nombre_carpeta[0].upper()
        if nombre_carpeta.endswith("_lower"):
            return nombre_carpeta[0].lower()
        return nombre_carpeta

    def _iterar_archivos(self) -> Iterator[Tuple[str, int]]:
        for nombre_clase in sorted(os.listdir(self.ruta_datos)):
            ruta_clase = os.path.join(self.ruta_datos, nombre_clase)
            if not os.path.isdir(ruta_clase):
                continue

            etiqueta_real = self._mapear_carpeta_a_etiqueta(nombre_clase)
            indice_clase = self.mapa_etiquetas.get_index(etiqueta_real)
            if indice_clase == -1:
                print(f"Advertencia: etiqueta '{etiqueta_real}' no encontrada.")
                continue

            for archivo in sorted(os.listdir(ruta_clase)):
                if archivo.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    yield os.path.join(ruta_clase, archivo), indice_clase

    def cargar_desde_directorio(self) -> None:
        """Cargar rutas de archivo y etiquetas desde ``ruta_datos``."""
        rutas: List[str] = []
        etiquetas: List[int] = []
        for ruta_imagen, indice in self._iterar_archivos():
            rutas.append(ruta_imagen)
            etiquetas.append(indice)

        self.rutas_imagenes = rutas
        self.etiquetas = etiquetas
        clases = len(set(self.etiquetas))
        print(f"Encontradas {len(self.rutas_imagenes)} imagenes de {clases} clases.")

    def dividir_datos(
        self,
        proporcion_entrenamiento: float = 0.8,
        semilla: int = 42
    ) -> Tuple[List[str], List[str], List[int], List[int]]:
        """Dividir las rutas de archivo en conjuntos de entrenamiento y validación."""
        if train_test_split is None:
            raise ImportError("scikit-learn no está instalado.")
        
        X_train, X_val, y_train, y_val = train_test_split(
            self.rutas_imagenes,
            self.etiquetas,
            train_size=proporcion_entrenamiento,
            random_state=semilla,
            stratify=self.etiquetas,
        )
        return X_train, X_val, y_train, y_val

    def generar_lotes(
        self,
        rutas_imagenes: List[str],
        etiquetas: List[int],
        tamano_lote: int,
        tamano_imagen: Tuple[int, int],
        augment: bool = False
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generador que produce lotes de imágenes y etiquetas preprocesadas."""
        num_muestras = len(rutas_imagenes)
        indices = np.arange(num_muestras)
        rng = np.random.default_rng(DATA_CONFIG.get('semilla'))

        while True:
            rng.shuffle(indices)
            for inicio in range(0, num_muestras, tamano_lote):
                fin = inicio + tamano_lote
                if fin > num_muestras:
                    continue
                
                indices_lote = indices[inicio:fin]
                
                lote_X: List[np.ndarray] = []
                lote_y: List[int] = []

                for i in indices_lote:
                    try:
                        imagen = _leer_imagen_gris(rutas_imagenes[i], tamano_imagen)
                        if augment and DATA_CONFIG.get('usar_augmentacion'):
                            imagen = apply_augmentation(imagen)
                        
                        lote_X.append(imagen)
                        lote_y.append(etiquetas[i])
                    except (OSError, ValueError) as e:
                        print(f"Error cargando {rutas_imagenes[i]}: {e}")

                if not lote_X:
                    continue

                
                lote_X_np = np.array(lote_X, dtype=np.float32)
                lote_X_np = normalize_image(lote_X_np)
                lote_X_np = lote_X_np.reshape(lote_X_np.shape[0], -1)
                
                yield lote_X_np, np.array(lote_y)
