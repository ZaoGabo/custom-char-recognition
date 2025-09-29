import numpy
from label_map import index_to_label

def normalizar_entrada(valores):
    return (numpy.asarray(valores, dtype=float) / 255.0 * 0.99) + 0.01

def evaluar_red(red, datos_prueba):
    aciertos = []

    for registro in datos_prueba:
        valores = registro.strip().split(',')
        etiqueta_correcta = valores[0]
        entrada = normalizar_entrada(valores[1:])
        salida = red.predecir(entrada)
        indice_predicho = numpy.argmax(salida)
        etiqueta_predicha = index_to_label(indice_predicho)

        aciertos.append(int(etiqueta_predicha == etiqueta_correcta))

    rendimiento = numpy.asarray(aciertos).mean()
    print("Rendimiento =", rendimiento)
    return rendimiento
