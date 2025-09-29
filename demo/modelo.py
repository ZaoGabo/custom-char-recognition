"""
Módulo con la clase de red neuronal simple.
"""

import numpy as np

def sigmoid(x):
    """Función sigmoid simple."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class RedNeuronalSimple:
    """Red neuronal simple para reconocimiento de caracteres."""
    
    def __init__(self, entrada_neuronas, oculta_neuronas, salida_neuronas, tasa_aprendizaje=0.1):
        self.entrada_neuronas = entrada_neuronas
        self.oculta_neuronas = oculta_neuronas
        self.salida_neuronas = salida_neuronas
        self.tasa_aprendizaje = tasa_aprendizaje
        
        # Inicializar pesos aleatoriamente
        self.pesos_entrada_oculta = np.random.normal(0.0, pow(self.oculta_neuronas, -0.5), 
                                                    (self.oculta_neuronas, self.entrada_neuronas))
        self.pesos_oculta_salida = np.random.normal(0.0, pow(self.salida_neuronas, -0.5), 
                                                   (self.salida_neuronas, self.oculta_neuronas))
        
        # Inicializar sesgos
        self.sesgo_oculta = np.zeros((self.oculta_neuronas, 1))
        self.sesgo_salida = np.zeros((self.salida_neuronas, 1))
    
    def predecir(self, entrada_lista):
        """Hacer predicción."""
        # Convertir lista de entrada a array 2D
        entradas = np.array(entrada_lista, ndmin=2).T
        
        # Calcular señales hacia la capa oculta
        entradas_oculta = np.dot(self.pesos_entrada_oculta, entradas) + self.sesgo_oculta
        salidas_oculta = sigmoid(entradas_oculta)
        
        # Calcular señales hacia la capa de salida
        entradas_salida = np.dot(self.pesos_oculta_salida, salidas_oculta) + self.sesgo_salida
        salidas_finales = sigmoid(entradas_salida)
        
        return salidas_finales
    
    def entrenar(self, entrada_lista, objetivo_lista):
        """Entrenar la red neuronal."""
        # Convertir entradas a arrays 2D
        entradas = np.array(entrada_lista, ndmin=2).T
        objetivos = np.array(objetivo_lista, ndmin=2).T
        
        # Forward pass
        entradas_oculta = np.dot(self.pesos_entrada_oculta, entradas) + self.sesgo_oculta
        salidas_oculta = sigmoid(entradas_oculta)
        
        entradas_salida = np.dot(self.pesos_oculta_salida, salidas_oculta) + self.sesgo_salida
        salidas_finales = sigmoid(entradas_salida)
        
        # Calcular errores
        errores_salida = objetivos - salidas_finales
        errores_oculta = np.dot(self.pesos_oculta_salida.T, errores_salida)
        
        # Actualizar pesos y sesgos
        self.pesos_oculta_salida += self.tasa_aprendizaje * np.dot(
            (errores_salida * salidas_finales * (1.0 - salidas_finales)), 
            salidas_oculta.T
        )
        
        self.sesgo_salida += self.tasa_aprendizaje * errores_salida * salidas_finales * (1.0 - salidas_finales)
        
        self.pesos_entrada_oculta += self.tasa_aprendizaje * np.dot(
            (errores_oculta * salidas_oculta * (1.0 - salidas_oculta)), 
            entradas.T
        )
        
        self.sesgo_oculta += self.tasa_aprendizaje * errores_oculta * salidas_oculta * (1.0 - salidas_oculta)