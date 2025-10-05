import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float32)


def _leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0.0, x, alpha * x)


def _leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    grad = np.ones_like(x)
    grad[x < 0.0] = alpha
    return grad


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


_ACTIVATIONS: Dict[str, Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]] = {
    "relu": (_relu, _relu_derivative),
    "leaky_relu": (_leaky_relu, _leaky_relu_derivative),
    "sigmoid": (_sigmoid, _sigmoid_derivative),
}


class NeuralNetwork:
    """Red neuronal multicapa con soporte para Adam, L2 y dropout."""

    def __init__(
        self,
        capas: List[int],
        activaciones: Optional[List[str]] = None,
        tasa_aprendizaje: float = 0.001,
        lambda_l2: float = 0.0,
        dropout_rate: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        semilla: Optional[int] = None,
    ) -> None:
        if len(capas) < 2:
            raise ValueError("Se requiere al menos una capa de entrada y una de salida")

        self.capas = capas
        self.num_capas = len(capas)

        if activaciones is None:
            activaciones = ["relu"] * (self.num_capas - 2) + ["softmax"]
        if len(activaciones) != self.num_capas - 1:
            raise ValueError("La cantidad de activaciones debe ser num_capas - 1")

        self.activaciones = activaciones
        self.tasa_aprendizaje = tasa_aprendizaje
        self.lambda_l2 = lambda_l2
        self.dropout_rate = dropout_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.rng = np.random.default_rng(semilla)

        self.pesos: List[np.ndarray] = []
        self.sesgos: List[np.ndarray] = []
        self.momentos_m: List[np.ndarray] = []
        self.momentos_v: List[np.ndarray] = []
        self.t = 0

        self._inicializar_parametros()

    def _inicializar_parametros(self) -> None:
        self.pesos.clear()
        self.sesgos.clear()
        self.momentos_m.clear()
        self.momentos_v.clear()

        for i in range(self.num_capas - 1):
            fan_in = self.capas[i]
            fan_out = self.capas[i + 1]
            activacion = self.activaciones[i]

            if activacion in ("relu", "leaky_relu"):
                limite = math.sqrt(2.0 / fan_in)
            else:
                limite = math.sqrt(1.0 / fan_in)

            pesos = self.rng.normal(0.0, limite, size=(fan_out, fan_in)).astype(np.float32)
            sesgos = np.zeros((fan_out, 1), dtype=np.float32)

            self.pesos.append(pesos)
            self.sesgos.append(sesgos)
            self.momentos_m.append(np.zeros_like(pesos))
            self.momentos_v.append(np.zeros_like(pesos))

    def _aplicar_dropout(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.dropout_rate <= 0.0:
            return A, np.ones_like(A)
        mascara = (self.rng.random(A.shape) > self.dropout_rate).astype(np.float32)
        A_dropout = (A * mascara) / (1.0 - self.dropout_rate)
        return A_dropout, mascara

    def _forward(
        self, X: np.ndarray, training: bool = True
    ):
        caches = {}
        A = X.T
        caches["A0"] = A

        dropout_masks = {}

        for i in range(self.num_capas - 1):
            W = self.pesos[i]
            b = self.sesgos[i]
            Z = W @ A + b
            caches[f"Z{i+1}"] = Z

            activacion = self.activaciones[i]
            if activacion == "softmax":
                A = _softmax(Z.T).T
            elif activacion == "sigmoid":
                A = _sigmoid(Z)
            elif activacion == "relu":
                A = _relu(Z)
            elif activacion == "leaky_relu":
                A = _leaky_relu(Z)
            else:
                raise ValueError(f"Activacion no soportada: {activacion}")

            if training and i < self.num_capas - 2 and self.dropout_rate > 0.0:
                A, mask = self._aplicar_dropout(A)
                dropout_masks[f"dropout{i+1}"] = mask

            caches[f"A{i+1}"] = A

        return caches, dropout_masks

    def _calcular_perdida(self, Y: np.ndarray, Y_pred: np.ndarray) -> float:
        m = Y.shape[0]
        eps = 1e-12
        log_probs = np.log(np.clip(Y_pred, eps, 1.0))
        perdida = -np.sum(Y * log_probs) / m

        if self.lambda_l2 > 0.0:
            suma = sum(np.sum(np.square(W)) for W in self.pesos)
            perdida += (self.lambda_l2 / (2.0 * m)) * suma
        return float(perdida)

    def _backward(
        self,
        caches: Dict[str, np.ndarray],
        dropout_masks: Dict[str, np.ndarray],
        X: np.ndarray,
        Y: np.ndarray,
    ):
        m = X.shape[0]
        A_final = caches[f"A{self.num_capas - 1}"]
        dZ = (A_final.T - Y) / m
        dZ = dZ.T

        grads_W: List[np.ndarray] = [np.zeros_like(W) for W in self.pesos]
        grads_b: List[np.ndarray] = [np.zeros_like(b) for b in self.sesgos]

        for i in reversed(range(self.num_capas - 1)):
            A_prev = caches[f"A{i}"]
            grads_W[i] = dZ @ A_prev.T
            grads_b[i] = np.sum(dZ, axis=1, keepdims=True)

            if self.lambda_l2 > 0.0:
                grads_W[i] += (self.lambda_l2 / m) * self.pesos[i]

            if i > 0:
                W = self.pesos[i]
                dA_prev = W.T @ dZ
                Z_prev = caches[f"Z{i}"]

                activacion = self.activaciones[i - 1]
                if activacion == "relu":
                    dZ_prev = dA_prev * _relu_derivative(Z_prev)
                elif activacion == "leaky_relu":
                    dZ_prev = dA_prev * _leaky_relu_derivative(Z_prev)
                elif activacion == "sigmoid":
                    dZ_prev = dA_prev * _sigmoid_derivative(Z_prev)
                else:
                    dZ_prev = dA_prev

                if self.dropout_rate > 0.0 and f"dropout{i}" in dropout_masks:
                    mask = dropout_masks[f"dropout{i}"]
                    dZ_prev *= mask
                    dZ_prev /= (1.0 - self.dropout_rate)

                dZ = dZ_prev

        return grads_W, grads_b

    def _actualizar_parametros(self, grads_W: List[np.ndarray], grads_b: List[np.ndarray]) -> None:
        self.t += 1
        for i in range(self.num_capas - 1):
            self.momentos_m[i] = self.beta1 * self.momentos_m[i] + (1.0 - self.beta1) * grads_W[i]
            self.momentos_v[i] = self.beta2 * self.momentos_v[i] + (1.0 - self.beta2) * np.square(grads_W[i])

            m_hat = self.momentos_m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.momentos_v[i] / (1.0 - self.beta2 ** self.t)

            self.pesos[i] -= self.tasa_aprendizaje * m_hat / (np.sqrt(v_hat) + self.epsilon)
            self.sesgos[i] -= self.tasa_aprendizaje * grads_b[i]

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
        if X.ndim != 2:
            raise ValueError("X debe ser un arreglo 2D (muestras, caracteristicas)")
        if Y.ndim != 2:
            raise ValueError("Y debe ser un arreglo 2D (muestras, clases)")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X y Y deben tener la misma cantidad de muestras")

        historia: List[Dict[str, float]] = []
        n = X.shape[0]

        for epoca in range(epocas):
            indices = np.arange(n)
            if barajar:
                self.rng.shuffle(indices)
            X_barajado = X[indices]
            Y_barajado = Y[indices]

            for inicio in range(0, n, tamano_lote):
                fin = inicio + tamano_lote
                X_lote = X_barajado[inicio:fin]
                Y_lote = Y_barajado[inicio:fin]
                if X_lote.size == 0:
                    continue

                caches, masks = self._forward(X_lote, training=True)
                grads_W, grads_b = self._backward(caches, masks, X_lote, Y_lote)
                self._actualizar_parametros(grads_W, grads_b)

            caches_train, _ = self._forward(X, training=False)
            Y_pred = caches_train[f"A{self.num_capas - 1}"].T
            perdida_train = self._calcular_perdida(Y, Y_pred)
            registro = {"epoch": epoca + 1, "loss_train": perdida_train}

            if X_val is not None and Y_val is not None:
                caches_val, _ = self._forward(X_val, training=False)
                Y_val_pred = caches_val[f"A{self.num_capas - 1}"].T
                perdida_val = self._calcular_perdida(Y_val, Y_val_pred)
                registro["loss_val"] = perdida_val
                registro["acc_val"] = self.calcular_precision(Y_val, Y_val_pred)

            historia.append(registro)

            if verbose and ((epoca + 1) % max(1, epocas // 10) == 0 or epoca == epocas - 1):
                mensaje = f"Epoca {epoca + 1}/{epocas} - loss: {perdida_train:.4f}"
                if "loss_val" in registro:
                    mensaje += f" - val_loss: {registro['loss_val']:.4f}"
                    mensaje += f" - val_acc: {registro['acc_val']:.4f}"
                print(mensaje)

        return historia

    def predecir_probabilidades(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        caches, _ = self._forward(X, training=False)
        return caches[f"A{self.num_capas - 1}"].T

    def predecir(self, X: np.ndarray) -> np.ndarray:
        probabilidades = self.predecir_probabilidades(X)
        return np.argmax(probabilidades, axis=1)

    @staticmethod
    def calcular_precision(Y_true_one_hot: np.ndarray, Y_pred_prob: np.ndarray) -> float:
        y_true = np.argmax(Y_true_one_hot, axis=1)
        y_pred = np.argmax(Y_pred_prob, axis=1)
        return float(np.mean(y_true == y_pred))

    def reset(self) -> None:
        self.t = 0
        self._inicializar_parametros()
