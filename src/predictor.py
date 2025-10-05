import numpy as np

from src.label_map import DEFAULT_LABEL_MAP


def normalizar_entrada(valores):
    arr = np.asarray(valores, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def evaluar_red(red, datos_prueba):
    aciertos = []

    for registro in datos_prueba:
        valores = registro.strip().split(',')
        etiqueta_correcta = valores[0]
        entrada = normalizar_entrada([float(v) for v in valores[1:]])
        salida = red.predecir_probabilidades(entrada)
        indice_predicho = int(np.argmax(salida))
        etiqueta_predicha = DEFAULT_LABEL_MAP.get_label(indice_predicho)

        aciertos.append(int(etiqueta_predicha == etiqueta_correcta))

    rendimiento = float(np.mean(aciertos))
    print("Rendimiento =", rendimiento)
    return rendimiento
