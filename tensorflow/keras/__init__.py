from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlretrieve

import numpy as np


def set_random_seed(seed):
    np.random.seed(seed)


class History:
    def __init__(self):
        self.history = {}


class Dense:
    def __init__(self, units, activation=None, input_shape=None):
        self.units = units
        self.activation = activation
        self.input_shape = input_shape
        self.built = False

    def build(self, input_dim):
        std = np.sqrt(2.0 / input_dim)
        self.W = (np.random.randn(input_dim, self.units) * std).astype(np.float32)
        self.b = np.zeros(self.units, dtype=np.float32)
        self.built = True

    def forward(self, x):
        self.last_input = x
        self.z = x @ self.W + self.b
        if self.activation == 'relu':
            self.a = np.maximum(0.0, self.z)
        elif self.activation == 'softmax':
            shifted = self.z - np.max(self.z, axis=1, keepdims=True)
            exp = np.exp(shifted)
            self.a = exp / np.sum(exp, axis=1, keepdims=True)
        else:
            self.a = self.z
        return self.a

    def backward(self, grad_output):
        if self.activation == 'relu':
            grad_z = grad_output * (self.z > 0)
        elif self.activation == 'softmax':
            grad_z = grad_output
        else:
            grad_z = grad_output

        self.grad_W = self.last_input.T @ grad_z
        self.grad_b = np.sum(grad_z, axis=0)
        return grad_z @ self.W.T


class AdamOptimizer:
    def __init__(self, layers, learning_rate=0.001):
        self.layers = layers
        self.learning_rate = learning_rate
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
        self.m = []
        self.v = []
        for layer in layers:
            self.m.append({'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)})
            self.v.append({'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)})

    def step(self):
        self.t += 1
        for index, layer in enumerate(self.layers):
            self.m[index]['W'] = self.beta1 * self.m[index]['W'] + (1 - self.beta1) * layer.grad_W
            self.m[index]['b'] = self.beta1 * self.m[index]['b'] + (1 - self.beta1) * layer.grad_b
            self.v[index]['W'] = self.beta2 * self.v[index]['W'] + (1 - self.beta2) * (layer.grad_W ** 2)
            self.v[index]['b'] = self.beta2 * self.v[index]['b'] + (1 - self.beta2) * (layer.grad_b ** 2)

            m_hat_w = self.m[index]['W'] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m[index]['b'] / (1 - self.beta1 ** self.t)
            v_hat_w = self.v[index]['W'] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v[index]['b'] / (1 - self.beta2 ** self.t)

            layer.W -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            layer.b -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)


class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.compiled = False

    def _build(self, input_dim):
        current_dim = input_dim
        for layer in self.layers:
            if not layer.built:
                layer.build(current_dim)
            current_dim = layer.units

    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=None):
        self.loss_name = loss
        self.metrics = metrics or []
        self.optimizer_name = optimizer
        self.compiled = True

    def summary(self):
        total_params = 0
        print("Model: Sequential")
        print("=" * 60)
        current_dim = None
        for index, layer in enumerate(self.layers, start=1):
            if current_dim is None and layer.input_shape is not None:
                current_dim = layer.input_shape[0]
            if not layer.built and isinstance(current_dim, int):
                layer.build(current_dim)
            if layer.built:
                params = layer.W.size + layer.b.size
                total_params += params
                print(f"Layer {index}: Dense({layer.units}, activation={layer.activation}) | params={params}")
                current_dim = layer.units
            else:
                print(f"Layer {index}: Dense({layer.units}, activation={layer.activation})")
        print("=" * 60)
        print(f"Total params: {total_params}")

    def _forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def _loss_and_grad(self, y_true, probs):
        y_true = y_true.astype(int)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(y_true)), y_true] = 1.0
        loss = -np.mean(np.sum(one_hot * np.log(probs), axis=1))
        grad = probs - one_hot
        return loss, grad

    def fit(self, X, y, epochs=1, batch_size=32, validation_split=0.0, verbose=1):
        history = History()
        history.history = {'loss': [], 'accuracy': []}
        if validation_split:
            history.history['val_loss'] = []
            history.history['val_accuracy'] = []

        if epochs == 0:
            return history

        X = X.astype(np.float32)
        y = y.astype(np.int64)
        n_samples = len(X)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        val_size = int(n_samples * validation_split)
        if val_size:
            X_val = X[-val_size:]
            y_val = y[-val_size:]
            X_train = X[:-val_size]
            y_train = y[:-val_size]
        else:
            X_val = y_val = None
            X_train = X
            y_train = y

        self._build(X_train.shape[1])
        optimizer = AdamOptimizer(self.layers, learning_rate=0.001)

        for epoch in range(epochs):
            perm = np.random.permutation(len(X_train))
            X_train = X_train[perm]
            y_train = y_train[perm]

            for start in range(0, len(X_train), batch_size):
                end = start + batch_size
                xb = X_train[start:end]
                yb = y_train[start:end]
                probs = self._forward(xb)
                _, grad = self._loss_and_grad(yb, probs)
                back = grad
                for layer in reversed(self.layers):
                    back = layer.backward(back)
                    layer.grad_W /= len(xb)
                    layer.grad_b /= len(xb)
                optimizer.step()

            train_probs = self._forward(X_train)
            train_loss, _ = self._loss_and_grad(y_train, train_probs)
            train_acc = float(np.mean(np.argmax(train_probs, axis=1) == y_train))
            history.history['loss'].append(train_loss)
            history.history['accuracy'].append(train_acc)

            if val_size:
                val_probs = self._forward(X_val)
                val_loss, _ = self._loss_and_grad(y_val, val_probs)
                val_acc = float(np.mean(np.argmax(val_probs, axis=1) == y_val))
                history.history['val_loss'].append(val_loss)
                history.history['val_accuracy'].append(val_acc)

            if verbose:
                message = f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f}"
                if val_size:
                    message += f" - val_accuracy: {val_acc:.4f}"
                print(message)

        return history

    def evaluate(self, X, y, verbose=0):
        probs = self._forward(X.astype(np.float32))
        loss, _ = self._loss_and_grad(y.astype(np.int64), probs)
        acc = float(np.mean(np.argmax(probs, axis=1) == y))
        return loss, acc


def _load_mnist_npz():
    cache_dir = Path.home() / '.keras' / 'datasets'
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_path = cache_dir / 'mnist.npz'
    if not file_path.exists():
        urlretrieve('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz', file_path)
    with np.load(file_path, allow_pickle=False) as data:
        return (data['x_train'], data['y_train']), (data['x_test'], data['y_test'])


mnist = SimpleNamespace(load_data=_load_mnist_npz)
datasets = SimpleNamespace(mnist=mnist)
layers = SimpleNamespace(Dense=Dense)
utils = SimpleNamespace(set_random_seed=set_random_seed)