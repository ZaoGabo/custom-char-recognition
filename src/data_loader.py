"""Data loading and preprocessing utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover - optional dependency
    train_test_split = None

from .config import DATA_CONFIG
from .label_map import LabelMap, DEFAULT_LABEL_MAP
from .utils import apply_augmentation, normalize_image

CV2_IMREAD_GRAYSCALE = getattr(cv2, "IMREAD_GRAYSCALE", 0) if cv2 is not None else 0


def _leer_imagen_gris(ruta_imagen: str, tamano: Tuple[int, int]) -> np.ndarray:
    """Return a grayscale image resized to ``tamano``."""
    alto, ancho = tamano
    if cv2 is not None:
        imread = getattr(cv2, "imread", None)
        resize = getattr(cv2, "resize", None)
        if callable(imread) and callable(resize):
            imagen = imread(ruta_imagen, CV2_IMREAD_GRAYSCALE)
            if imagen is None:
                raise ValueError(f"No se pudo cargar la imagen {ruta_imagen}")
            return resize(imagen, (ancho, alto))

    with Image.open(ruta_imagen) as img_pil:
        imagen = img_pil.convert("L").resize((ancho, alto))
    return np.array(imagen, dtype=np.uint8)


class DataLoader:
    """Carga y preprocesamiento de datos para la red neuronal."""

    def __init__(self, ruta_datos: str, mapa_etiquetas: Optional[LabelMap] = None) -> None:
        self.ruta_datos = ruta_datos
        self.mapa_etiquetas = mapa_etiquetas or DEFAULT_LABEL_MAP
        self.imagenes: np.ndarray = np.array([])
        self.etiquetas: np.ndarray = np.array([])

    def _mapear_carpeta_a_etiqueta(self, nombre_carpeta: str) -> str:
        """Convertir un nombre de carpeta (p. ej. ``A_upper``) a la etiqueta real."""
        if nombre_carpeta.endswith("_upper"):
            return nombre_carpeta[0].upper()
        if nombre_carpeta.endswith("_lower"):
            return nombre_carpeta[0].lower()
        return nombre_carpeta

    def _iterar_archivos(self) -> Iterable[Tuple[str, int]]:
        for nombre_clase in sorted(os.listdir(self.ruta_datos)):
            ruta_clase = os.path.join(self.ruta_datos, nombre_clase)
            if not os.path.isdir(ruta_clase):
                continue

            etiqueta_real = self._mapear_carpeta_a_etiqueta(nombre_clase)
            indice_clase = self.mapa_etiquetas.get_index(etiqueta_real)
            if indice_clase == -1:
                print(f"Advertencia: etiqueta '{etiqueta_real}' no encontrada en el mapa de etiquetas")
                continue

            for archivo in sorted(os.listdir(ruta_clase)):
                if archivo.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    yield os.path.join(ruta_clase, archivo), indice_clase

    def cargar_desde_directorio(self, tamano_imagen: Optional[Tuple[int, int]] = None) -> None:
        """Cargar todas las imágenes almacenadas en ``ruta_datos``."""
        tamano = tamano_imagen or DATA_CONFIG["tamano_imagen"]
        imagenes: List[np.ndarray] = []
        etiquetas: List[int] = []

        for ruta_imagen, indice in self._iterar_archivos():
            try:
                imagen = _leer_imagen_gris(ruta_imagen, tamano)
            except (OSError, ValueError) as error:
                print(f"Error cargando {ruta_imagen}: {error}")
                continue
            imagenes.append(imagen)
            etiquetas.append(indice)

        self.imagenes = np.array(imagenes)
        self.etiquetas = np.array(etiquetas)
        clases = len(np.unique(self.etiquetas)) if self.etiquetas.size else 0
        print(f"Cargadas {len(self.imagenes)} imagenes de {clases} clases")

    def cargar_desde_csv(
        self,
        ruta_csv: str,
        columna_imagen: str = 'image_path',
        columna_etiqueta: str = 'label',
    ) -> None:
        """Leer imágenes y etiquetas desde un CSV."""
        dataframe = pd.read_csv(ruta_csv)
        imagenes: List[np.ndarray] = []
        etiquetas: List[int] = []

        base_path = Path(ruta_csv).parent
        for _, fila in dataframe.iterrows():
            ruta_imagen = Path(fila[columna_imagen])
            if not ruta_imagen.is_absolute():
                ruta_imagen = base_path / ruta_imagen
            etiqueta = fila[columna_etiqueta]

            try:
                imagen = _leer_imagen_gris(str(ruta_imagen), DATA_CONFIG['tamano_imagen'])
            except (OSError, ValueError) as error:
                print(f"Error cargando {ruta_imagen}: {error}")
                continue

            indice = self.mapa_etiquetas.get_index(etiqueta)
            if indice == -1:
                print(f"Advertencia: etiqueta '{etiqueta}' no encontrada")
                continue

            imagenes.append(imagen)
            etiquetas.append(indice)

        self.imagenes = np.array(imagenes)
        self.etiquetas = np.array(etiquetas)

    def preprocesar_imagenes(self) -> None:
        """Normalizar y aplanar las imágenes cargadas."""
        if DATA_CONFIG.get('normalizar', True):
            self.imagenes = normalize_image(self.imagenes)
        self.imagenes = self.imagenes.reshape(self.imagenes.shape[0], -1)

    def _dividir_manual(
        self,
        proporcion_entrenamiento: float,
        proporcion_validacion: float,
        semilla: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(semilla)
        val_ratio = proporcion_validacion / (proporcion_validacion + DATA_CONFIG['division_prueba'])

        train_idx: List[int] = []
        val_idx: List[int] = []
        test_idx: List[int] = []

        for clase in np.unique(self.etiquetas):
            indices = np.where(self.etiquetas == clase)[0]
            rng.shuffle(indices)
            n_total = len(indices)
            n_train = int(np.floor(proporcion_entrenamiento * n_total))
            n_remanente = n_total - n_train
            n_val = int(np.floor(val_ratio * n_remanente))
            n_test = n_total - n_train - n_val

            train_idx.extend(indices[:n_train])
            val_idx.extend(indices[n_train:n_train + n_val])
            test_idx.extend(indices[n_train + n_val:n_train + n_val + n_test])

        for arreglo in (train_idx, val_idx, test_idx):
            rng.shuffle(arreglo)

        train_idx_arr = np.array(train_idx, dtype=int)
        val_idx_arr = np.array(val_idx, dtype=int)
        test_idx_arr = np.array(test_idx, dtype=int)

        return (
            self.imagenes[train_idx_arr],
            self.imagenes[val_idx_arr],
            self.imagenes[test_idx_arr],
            self.etiquetas[train_idx_arr],
            self.etiquetas[val_idx_arr],
            self.etiquetas[test_idx_arr],
        )

    def dividir_datos(
        self,
        proporcion_entrenamiento: Optional[float] = None,
        proporcion_validacion: Optional[float] = None,
        semilla: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Dividir el conjunto cargado en train / validation / test."""
        proporcion_entrenamiento = (
            proporcion_entrenamiento or DATA_CONFIG['division_entrenamiento']
        )
        proporcion_validacion = (
            proporcion_validacion or DATA_CONFIG['division_validacion']
        )

        if train_test_split is None:
            return self._dividir_manual(proporcion_entrenamiento, proporcion_validacion, semilla)

        X_train, X_temp, y_train, y_temp = train_test_split(
            self.imagenes,
            self.etiquetas,
            train_size=proporcion_entrenamiento,
            random_state=semilla,
            stratify=self.etiquetas,
        )

        proporcion_val = proporcion_validacion / (
            proporcion_validacion + DATA_CONFIG['division_prueba']
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            train_size=proporcion_val,
            random_state=semilla,
            stratify=y_temp,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def aplicar_augmentacion(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        factor: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generar copias augmentadas de ``X_train`` y ``y_train``."""
        if not DATA_CONFIG.get('usar_augmentacion', False):
            return X_train, y_train

        imagenes_aug: List[np.ndarray] = []
        etiquetas_aug: List[int] = []

        for imagen_original, etiqueta in zip(X_train, y_train):
            imagenes_aug.append(imagen_original)
            etiquetas_aug.append(int(etiqueta))
            for _ in range(max(factor - 1, 0)):
                imagen_aug = self._augmentacion_simple(
                    imagen_original.reshape(DATA_CONFIG['tamano_imagen'])
                )
                imagenes_aug.append(imagen_aug.flatten())
                etiquetas_aug.append(int(etiqueta))

        print(
            "Augmentacion completa: "
            f"{len(X_train)} -> {len(imagenes_aug)} imagenes"
        )

        return np.array(imagenes_aug), np.array(etiquetas_aug)

    def _augmentacion_simple(self, imagen_2d: np.ndarray) -> np.ndarray:
        """Crear una variante sencilla de ``imagen_2d``."""
        imagen = imagen_2d.astype(np.float32).copy()

        if np.random.random() > 0.5:
            ruido = np.random.normal(0, 0.05, imagen.shape)
            imagen = np.clip(imagen + ruido, 0, 1)

        if np.random.random() > 0.5:
            shift_x = np.random.randint(-2, 3)
            shift_y = np.random.randint(-2, 3)
            imagen = np.roll(imagen, shift_x, axis=1)
            imagen = np.roll(imagen, shift_y, axis=0)

        if np.random.random() > 0.5:
            factor = np.random.uniform(0.9, 1.1)
            imagen = np.clip(imagen * factor, 0, 1)

        return imagen

    def obtener_pesos_clase(self) -> dict:
        """Calcular pesos balanceados para cada clase."""
        from sklearn.utils.class_weight import compute_class_weight  # type: ignore

        clases = np.unique(self.etiquetas)
        pesos = compute_class_weight('balanced', classes=clases, y=self.etiquetas)
        return dict(zip(clases, pesos))