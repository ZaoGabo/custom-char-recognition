"""Implementacion de una red neuronal totalmente conectada con optimizador Adam."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float32)


def _leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0.0, x, alpha * x)


def _leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0.0, 1.0, alpha).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def _sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    sig = _sigmoid(x)
    return sig * (1.0 - sig)


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


_ACTIVACIONES = {
    'relu': (_relu, _relu_derivative),
    'leaky_relu': (_leaky_relu, _leaky_relu_derivative),
    'sigmoid': (_sigmoid, _sigmoid_derivative),
}


class NeuralNetwork:
    """Red neuronal multicapa con soporte para dropout, L2, Adam y BatchNorm."""

    def __init__(
        self,
        capas: List[int],
        activaciones: Optional[List[str]] = None,
        tasa_aprendizaje: float = 0.001,
        lambda_l2: float = 0.0,
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        semilla: Optional[int] = None,
    ) -> None:
        if len(capas) < 2:
            raise ValueError('Se requiere al menos una capa de entrada y una de salida')

        self.capas = capas
        self.num_capas = len(capas)
        self.activaciones = activaciones or ['relu'] * (self.num_capas - 2) + ['softmax']
        if len(self.activaciones) != self.num_capas - 1:
            raise ValueError('El numero de activaciones debe ser num_capas - 1')

        self.tasa_aprendizaje = tasa_aprendizaje
        self.lambda_l2 = lambda_l2
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.bn_momentum = 0.9
        self.rng = np.random.default_rng(semilla)

        self.pesos: List[np.ndarray] = []
        self.sesgos: List[np.ndarray] = []
        self.gammas: List[np.ndarray] = []
        self.betas: List[np.ndarray] = []
        self.running_means: List[np.ndarray] = []
        self.running_vars: List[np.ndarray] = []

        self.momentos_m: List[np.ndarray] = []
        self.momentos_v: List[np.ndarray] = []
        self.momentos_m_gamma: List[np.ndarray] = []
        self.momentos_v_gamma: List[np.ndarray] = []
        self.momentos_m_beta: List[np.ndarray] = []
        self.momentos_v_beta: List[np.ndarray] = []
        self.t = 0

        self._inicializar_parametros()

    def _inicializar_parametros(self) -> None:
        """Inicializar pesos, sesgos y momentos."""
        self.pesos.clear()
        self.sesgos.clear()
        self.momentos_m.clear()
        self.momentos_v.clear()

        if self.use_batch_norm:
            self.gammas.clear()
            self.betas.clear()
            self.running_means.clear()
            self.running_vars.clear()
            self.momentos_m_gamma.clear()
            self.momentos_v_gamma.clear()
            self.momentos_m_beta.clear()
            self.momentos_v_beta.clear()

        for i in range(self.num_capas - 1):
            fan_in = self.capas[i]
            fan_out = self.capas[i + 1]
            activacion = self.activaciones[i]

            if activacion in ('relu', 'leaky_relu'):
                limite = math.sqrt(2.0 / fan_in)
            else:
                limite = math.sqrt(1.0 / fan_in)

            pesos = self.rng.normal(0.0, limite, size=(fan_out, fan_in)).astype(np.float32)
            sesgos = np.zeros((fan_out, 1), dtype=np.float32)

            self.pesos.append(pesos)
            self.sesgos.append(sesgos)
            self.momentos_m.append(np.zeros_like(pesos))
            self.momentos_v.append(np.zeros_like(pesos))

            if self.use_batch_norm and i < self.num_capas - 2:
                gamma = np.ones((fan_out, 1), dtype=np.float32)
                beta = np.zeros((fan_out, 1), dtype=np.float32)
                running_mean = np.zeros((fan_out, 1), dtype=np.float32)
                running_var = np.ones((fan_out, 1), dtype=np.float32)

                self.gammas.append(gamma)
                self.betas.append(beta)
                self.running_means.append(running_mean)
                self.running_vars.append(running_var)
                self.momentos_m_gamma.append(np.zeros_like(gamma))
                self.momentos_v_gamma.append(np.zeros_like(gamma))
                self.momentos_m_beta.append(np.zeros_like(beta))
                self.momentos_v_beta.append(np.zeros_like(beta))

    def _aplicar_dropout(self, activaciones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Aplicar máscara de dropout y devolver logits y máscara."""
        if self.dropout_rate <= 0.0:
            return activaciones, np.ones_like(activaciones)
        mascara = (self.rng.random(activaciones.shape) > self.dropout_rate).astype(np.float32)
        activaciones_dropout = (activaciones * mascara) / (1.0 - self.dropout_rate)
        return activaciones_dropout, mascara

    def _forward(
        self,
        X: np.ndarray,
        training: bool = True,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[Tuple]]:
        """Realizar el paso forward y devolver caches intermedios."""
        caches: Dict[str, np.ndarray] = {}
        dropout_masks: Dict[str, np.ndarray] = {}
        bn_caches: List[Tuple] = []
        activacion_actual = X.T
        caches['A0'] = activacion_actual

        for i in range(self.num_capas - 1):
            pesos = self.pesos[i]
            sesgos = self.sesgos[i]
            logits = pesos @ activacion_actual + sesgos
            caches[f'Z{i+1}'] = logits

            if self.use_batch_norm and i < self.num_capas - 2:
                gamma = self.gammas[i]
                beta = self.betas[i]
                if training:
                    mean = np.mean(logits, axis=1, keepdims=True)
                    var = np.var(logits, axis=1, keepdims=True)
                    self.running_means[i] = self.bn_momentum * self.running_means[i] + (1 - self.bn_momentum) * mean
                    self.running_vars[i] = self.bn_momentum * self.running_vars[i] + (1 - self.bn_momentum) * var
                else:
                    mean = self.running_means[i]
                    var = self.running_vars[i]
                
                logits_norm = (logits - mean) / np.sqrt(var + self.epsilon)
                logits = gamma * logits_norm + beta
                bn_caches.append((logits_norm, var))

            caches[f'Z_bn{i+1}'] = logits

            nombre_activacion = self.activaciones[i]
            if nombre_activacion == 'softmax':
                activacion_actual = _softmax(logits.T).T
            elif nombre_activacion == 'sigmoid':
                activacion_actual = _sigmoid(logits)
            elif nombre_activacion == 'relu':
                activacion_actual = _relu(logits)
            elif nombre_activacion == 'leaky_relu':
                activacion_actual = _leaky_relu(logits)
            else:
                raise ValueError(f'Activacion no soportada: {nombre_activacion}')

            if training and i < self.num_capas - 2 and self.dropout_rate > 0.0:
                activacion_actual, mascara = self._aplicar_dropout(activacion_actual)
                dropout_masks[f'dropout{i+1}'] = mascara

            caches[f'A{i+1}'] = activacion_actual

        return caches, dropout_masks, bn_caches

    def _calcular_perdida(self, Y: np.ndarray, Y_pred: np.ndarray) -> float:
        """Calcular la funcion de perdida cross-entropy (con regularizacion L2)."""
        muestras = Y.shape[0]
        eps = 1e-12
        log_probs = np.log(np.clip(Y_pred, eps, 1.0))
        perdida = -np.sum(Y * log_probs) / muestras
        if self.lambda_l2 > 0.0:
            suma = sum(np.sum(np.square(peso)) for peso in self.pesos)
            perdida += (self.lambda_l2 / (2.0 * muestras)) * suma
        return float(perdida)

    def calcular_perdida(self, Y_true_one_hot: np.ndarray, Y_pred_prob: np.ndarray) -> float:
        """Interfaz publica para obtener la perdida del modelo."""
        return self._calcular_perdida(Y_true_one_hot, Y_pred_prob)

    def _backward(
        self,
        caches: Dict[str, np.ndarray],
        dropout_masks: Dict[str, np.ndarray],
        bn_caches: List[Tuple],
        X: np.ndarray,
        Y: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Calcular gradientes para todos los parametros."""
        muestras = X.shape[0]
        activacion_final = caches[f'A{self.num_capas - 1}']
        grad_actual = (activacion_final.T - Y) / muestras
        grad_actual = grad_actual.T

        grads_W = [np.zeros_like(p) for p in self.pesos]
        grads_b = [np.zeros_like(s) for s in self.sesgos]
        grads_gamma = [np.zeros_like(g) for g in self.gammas] if self.use_batch_norm else []
        grads_beta = [np.zeros_like(b) for b in self.betas] if self.use_batch_norm else []

        for i in reversed(range(self.num_capas - 1)):
            activacion_prev = caches[f'A{i}']
            
            if self.use_batch_norm and i < self.num_capas - 2:
                logits_norm, var = bn_caches[i]
                dZ_bn = grad_actual
                grads_gamma[i] = np.sum(dZ_bn * logits_norm, axis=1, keepdims=True)
                grads_beta[i] = np.sum(dZ_bn, axis=1, keepdims=True)

                dZ_norm = dZ_bn * self.gammas[i]
                dZ = (1. / (muestras * np.sqrt(var + self.epsilon))) * \
                    (muestras * dZ_norm - np.sum(dZ_norm, axis=1, keepdims=True) -
                    logits_norm * np.sum(dZ_norm * logits_norm, axis=1, keepdims=True))
            else:
                dZ = grad_actual

            grads_W[i] = dZ @ activacion_prev.T
            grads_b[i] = np.sum(dZ, axis=1, keepdims=True)

            if self.lambda_l2 > 0.0:
                grads_W[i] += (self.lambda_l2 / muestras) * self.pesos[i]

            if i > 0:
                pesos = self.pesos[i]
                grad_prev = pesos.T @ dZ
                logits_prev = caches[f'Z_bn{i}'] if self.use_batch_norm and i-1 < self.num_capas - 2 else caches[f'Z{i}']

                nombre_activacion = self.activaciones[i - 1]
                if nombre_activacion == 'relu':
                    grad_prev *= _relu_derivative(logits_prev)
                elif nombre_activacion == 'leaky_relu':
                    grad_prev *= _leaky_relu_derivative(logits_prev)
                elif nombre_activacion == 'sigmoid':
                    grad_prev *= _sigmoid_derivative(logits_prev)

                if self.dropout_rate > 0.0 and f'dropout{i}' in dropout_masks:
                    mascara = dropout_masks[f'dropout{i}']
                    grad_prev *= mascara
                    grad_prev /= (1.0 - self.dropout_rate)

                grad_actual = grad_prev

        return grads_W, grads_b, grads_gamma, grads_beta

    def _actualizar_parametros(
        self,
        grads_W: List[np.ndarray],
        grads_b: List[np.ndarray],
        grads_gamma: List[np.ndarray],
        grads_beta: List[np.ndarray],
    ) -> None:
        """Actualizar todos los parametros usando Adam."""
        self.t += 1
        for i in range(self.num_capas - 1):
            # Actualizar pesos y sesgos
            self.momentos_m[i] = self.beta1 * self.momentos_m[i] + (1.0 - self.beta1) * grads_W[i]
            self.momentos_v[i] = self.beta2 * self.momentos_v[i] + (1.0 - self.beta2) * np.square(grads_W[i])
            m_hat = self.momentos_m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.momentos_v[i] / (1.0 - self.beta2 ** self.t)
            self.pesos[i] -= self.tasa_aprendizaje * m_hat / (np.sqrt(v_hat) + self.epsilon)
            self.sesgos[i] -= self.tasa_aprendizaje * grads_b[i]

            # Actualizar gamma y beta si se usa BatchNorm
            if self.use_batch_norm and i < self.num_capas - 2:
                self.momentos_m_gamma[i] = self.beta1 * self.momentos_m_gamma[i] + (1.0 - self.beta1) * grads_gamma[i]
                self.momentos_v_gamma[i] = self.beta2 * self.momentos_v_gamma[i] + (1.0 - self.beta2) * np.square(grads_gamma[i])
                m_hat_gamma = self.momentos_m_gamma[i] / (1.0 - self.beta1 ** self.t)
                v_hat_gamma = self.momentos_v_gamma[i] / (1.0 - self.beta2 ** self.t)
                self.gammas[i] -= self.tasa_aprendizaje * m_hat_gamma / (np.sqrt(v_hat_gamma) + self.epsilon)

                self.momentos_m_beta[i] = self.beta1 * self.momentos_m_beta[i] + (1.0 - self.beta1) * grads_beta[i]
                self.momentos_v_beta[i] = self.beta2 * self.momentos_v_beta[i] + (1.0 - self.beta2) * np.square(grads_beta[i])
                m_hat_beta = self.momentos_m_beta[i] / (1.0 - self.beta1 ** self.t)
                v_hat_beta = self.momentos_v_beta[i] / (1.0 - self.beta2 ** self.t)
                self.betas[i] -= self.tasa_aprendizaje * m_hat_beta / (np.sqrt(v_hat_beta) + self.epsilon)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epocas: int = 100,
        tamano_lote: int = 32,
        barajar: bool = True,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> List[Dict[str, float]]:
        """Entrenar la red neuronal mediante mini-batch gradient descent."""
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError('X e Y deben ser matrices 2D (muestras x caracteristicas/clases)')
        if X.shape[0] != Y.shape[0]:
            raise ValueError('X e Y deben tener la misma cantidad de muestras')

        historia: List[Dict[str, float]] = []
        muestras = X.shape[0]

        for epoca in range(epocas):
            indices = np.arange(muestras)
            if barajar:
                self.rng.shuffle(indices)
            X_barajado = X[indices]
            Y_barajado = Y[indices]

            for inicio in range(0, muestras, tamano_lote):
                fin = inicio + tamano_lote
                X_lote = X_barajado[inicio:fin]
                Y_lote = Y_barajado[inicio:fin]
                if not X_lote.size:
                    continue

                caches, masks, bn_caches = self._forward(X_lote, training=True)
                grads_W, grads_b, grads_gamma, grads_beta = self._backward(caches, masks, bn_caches, X_lote, Y_lote)
                self._actualizar_parametros(grads_W, grads_b, grads_gamma, grads_beta)

            caches_train, _, _ = self._forward(X, training=False)
            Y_pred = caches_train[f'A{self.num_capas - 1}'].T
            perdida_train = self._calcular_perdida(Y, Y_pred)
            registro = {'epoch': epoca + 1, 'loss_train': perdida_train}

            if X_val is not None and Y_val is not None:
                caches_val, _, _ = self._forward(X_val, training=False)
                Y_val_pred = caches_val[f'A{self.num_capas - 1}'].T
                registro['loss_val'] = self._calcular_perdida(Y_val, Y_val_pred)
                registro['acc_val'] = self.calcular_precision(Y_val, Y_val_pred)

            historia.append(registro)

            if verbose and ((epoca + 1) % max(1, epocas // 10) == 0 or epoca == epocas - 1):
                mensaje = f"Epoca {epoca + 1}/{epocas} - loss: {perdida_train:.4f}"
                if 'loss_val' in registro:
                    mensaje += f" - val_loss: {registro['loss_val']:.4f}"
                    mensaje += f" - val_acc: {registro['acc_val']:.4f}"
                print(mensaje)

        return historia

    def predecir_probabilidades(self, X: np.ndarray) -> np.ndarray:
        """Obtener probabilidades de salida para ``X``."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        caches, _, _ = self._forward(X, training=False)
        return caches[f'A{self.num_capas - 1}'].T

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """Devolver los indices de las clases con mayor probabilidad."""
        probabilidades = self.predecir_probabilidades(X)
        return np.argmax(probabilidades, axis=1)

    @staticmethod
    def calcular_precision(Y_true_one_hot: np.ndarray, Y_pred_prob: np.ndarray) -> float:
        """Calcular la precision basada en codificacion one-hot."""
        y_true = np.argmax(Y_true_one_hot, axis=1)
        y_pred = np.argmax(Y_pred_prob, axis=1)
        return float(np.mean(y_true == y_pred))

    def reset(self) -> None:
        """Reinicializar pesos y contadores del optimizador."""
        self.t = 0
        self._inicializar_parametros()

    def set_tasa_aprendizaje(self, nueva_tasa: float) -> None:
        self.tasa_aprendizaje = nueva_tasa
        
    def guardar_modelo(self, ruta_base: str) -> None:
        """Guardar la arquitectura y parametros del modelo."""
        p = Path(ruta_base)
        p.mkdir(parents=True, exist_ok=True)

        arquitectura = {
            'capas': self.capas,
            'activaciones': self.activaciones,
            'dropout_rate': self.dropout_rate,
            'lambda_l2': self.lambda_l2,
            'use_batch_norm': self.use_batch_norm,
        }
        with open(p / "arquitectura.json", "w") as f:
            json.dump(arquitectura, f)

        for i, (pesos, sesgos) in enumerate(zip(self.pesos, self.sesgos)):
            np.save(p / f"pesos_{i}.npy", pesos)
            np.save(p / f"sesgos_{i}.npy", sesgos)

        if self.use_batch_norm:
            for i in range(self.num_capas - 2):
                np.save(p / f"gamma_{i}.npy", self.gammas[i])
                np.save(p / f"beta_{i}.npy", self.betas[i])
                np.save(p / f"running_mean_{i}.npy", self.running_means[i])
                np.save(p / f"running_var_{i}.npy", self.running_vars[i])

    @classmethod
    def cargar_modelo(cls, ruta_base: str) -> NeuralNetwork:
        """Cargar un modelo desde su arquitectura y pesos."""
        p = Path(ruta_base)
        with open(p / "arquitectura.json", "r") as f:
            arquitectura = json.load(f)
        
        modelo = cls(**arquitectura)
        
        for i in range(modelo.num_capas - 1):
            modelo.pesos[i] = np.load(p / f"pesos_{i}.npy")
            modelo.sesgos[i] = np.load(p / f"sesgos_{i}.npy")
            
        if modelo.use_batch_norm:
            for i in range(modelo.num_capas - 2):
                modelo.gammas[i] = np.load(p / f"gamma_{i}.npy")
                modelo.betas[i] = np.load(p / f"beta_{i}.npy")
                modelo.running_means[i] = np.load(p / f"running_mean_{i}.npy")
                modelo.running_vars[i] = np.load(p / f"running_var_{i}.npy")

        return modelo
