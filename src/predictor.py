"""Evaluacion rapida del modelo a partir de archivos CSV."""

from __future__ import annotations

import numpy as np

from src.label_map import DEFAULT_LABEL_MAP


def normalizar_entrada(valores: np.ndarray) -> np.ndarray:
    """Normalizar un vector de pixeles al rango [0, 1]."""
    arr = np.asarray(valores, dtype=np.float32)
    if arr.size and arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def evaluar_red(red, datos_prueba) -> float:
    """Calcular accuracy de ``red`` sobre iterables de lineas CSV."""
    aciertos: list[int] = []
    for registro in datos_prueba:
        valores = registro.strip().split(',')
        etiqueta_correcta = valores[0]
        entrada = normalizar_entrada(np.array(valores[1:], dtype=np.float32))
        salida = red.predecir_probabilidades(entrada)
        indice_predicho = int(np.argmax(salida))
        etiqueta_predicha = DEFAULT_LABEL_MAP.get_label(indice_predicho)
        aciertos.append(int(etiqueta_predicha == etiqueta_correcta))

    rendimiento = float(np.mean(aciertos))
    print('Rendimiento =', rendimiento)
    return rendimiento