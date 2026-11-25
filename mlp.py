"""
Small multi-layer perceptron (MLP) from scratch with pure NumPy.
Forward pass, backpropagation (chain rule) and weight updates
are written by hand. No autograd, no PyTorch.
"""

import numpy as np


def relu(z):
    return np.maximum(0.0, z)


def relu_grad(z):
    return (z > 0).astype(z.dtype)


def sigmoid(z):
    # numerically stable
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def binary_cross_entropy(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))


class MLP:
    """2 → h → h → 1  (ReLU, ReLU, sigmoid).

    Weights are initialised with He-style scaling. Backward returns gradients
    in the same order as the parameters.
    """

    def __init__(self, hidden=16, seed=0):
        rng = np.random.default_rng(seed)
        # W1: (2, h), b1: (h,)
        self.W1 = rng.normal(0, np.sqrt(2 / 2), size=(2, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / hidden), size=(hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.W3 = rng.normal(0, np.sqrt(2 / hidden), size=(hidden, 1))
        self.b3 = np.zeros(1)

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, X):
        """X: (N, 2) → probabilities; intermediate values cached."""
        self.X = X
        self.z1 = X @ self.W1 + self.b1          # (N, h)
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2    # (N, h)
        self.a2 = relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3    # (N, 1)
        self.a3 = sigmoid(self.z3)
        return self.a3

    def backward(self, y):
        """y: (N, 1). Returns list [dW1, db1, dW2, db2, dW3, db3].

        Loss = mean binary cross-entropy.
        dL/da3 = (a3 - y) / N   (sigmoid + BCE combined).
        """
        N = y.shape[0]
        # output layer: sigmoid + BCE → simple (a3 - y)
        da3 = (self.a3 - y) / N                  # (N, 1)
        dW3 = self.a2.T @ da3                    # (h, 1)
        db3 = da3.sum(axis=0)                    # (1,)
        da2 = da3 @ self.W3.T                    # (N, h)

        # second hidden layer
        dz2 = da2 * relu_grad(self.z2)
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T

        # first hidden layer
        dz1 = da1 * relu_grad(self.z1)
        dW1 = self.X.T @ dz1
        db1 = dz1.sum(axis=0)

        return [dW1, db1, dW2, db2, dW3, db3]

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
