import numpy
import scipy.special

class NeuralNetwork:
    def __init__(self, entrada_neuronas, oculta_neuronas, salida_neuronas, tasa_aprendizaje):
        self.entrada_neuronas = entrada_neuronas
        self.oculta_neuronas = oculta_neuronas
        self.salida_neuronas = salida_neuronas
        self.tasa_aprendizaje = tasa_aprendizaje
        
        # Matrices de pesos: entrada→oculta y oculta→salida
        self.pesos_entrada_oculta = numpy.random.rand(self.oculta_neuronas, self.entrada_neuronas) - 0.5
        self.pesos_oculta_salida = numpy.random.rand(self.salida_neuronas, self.oculta_neuronas) - 0.5
        
        # Función de activación sigmoide (método en lugar de lambda para serialización)
        
    def funcion_activacion(self, x):
        """Función de activación sigmoide"""
        return scipy.special.expit(x)
    
    def entrenar(self, datos_entrada, valores_objetivo):
        #convertir listas a arrays  y transponer
        entradas = numpy.array(datos_entrada, ndmin=2).T
        objetivos = numpy.array(valores_objetivo, ndmin=2).T
        #propagación hacia adelante
        señales_capa_oculta = numpy.dot(self.pesos_entrada_oculta, entradas)
        activaciones_capa_oculta = self.funcion_activacion(señales_capa_oculta)
        #calculo de errores y retropropagación
        señales_capa_salida = numpy.dot(self.pesos_oculta_salida, activaciones_capa_oculta)
        activaciones_capa_salida = self.funcion_activacion(señales_capa_salida)
        #actualización de pesos
        errores_salida = objetivos - activaciones_capa_salida
        errores_oculta = numpy.dot(self.pesos_oculta_salida.T, errores_salida)
        
        self.pesos_oculta_salida += self.tasa_aprendizaje * numpy.dot(
            errores_salida * activaciones_capa_salida * (1.0 - activaciones_capa_salida), 
            activaciones_capa_oculta.T
        )
        
        self.pesos_entrada_oculta += self.tasa_aprendizaje * numpy.dot(
            errores_oculta * activaciones_capa_oculta * (1.0 - activaciones_capa_oculta), 
            entradas.T
        )
    
    def predecir(self, datos_entrada):
        #convertir entrada a array y transponer
        entradas = numpy.array(datos_entrada, ndmin=2).T
        #propagación hacia adelante
        señales_capa_oculta = numpy.dot(self.pesos_entrada_oculta, entradas)
        activaciones_capa_oculta = self.funcion_activacion(señales_capa_oculta)
        
        señales_capa_salida = numpy.dot(self.pesos_oculta_salida, activaciones_capa_oculta)
        activaciones_capa_salida = self.funcion_activacion(señales_capa_salida)
        
        return activaciones_capa_salida
