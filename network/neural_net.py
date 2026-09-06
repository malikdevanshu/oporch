import numpy as np
from utils.activations import ActiFunctions

class NeuralNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        rng = np.random.default_rng()
        std = np.sqrt(2.0 / (input_dim + output_dim))
        self.w1 = rng.normal(0.0, std, size=(hidden_dim, input_dim))#Using Xavier Initialization, because the activation function has squashing behavior!
        self.b1 = np.zeros((hidden_dim, 1))

        self.w2 = rng.normal(0.0, std, size=(output_dim, hidden_dim))
        self.b2 = np.zeros((output_dim, 1))

    def forward_pass(self, X):


        z = np.dot(self.w1, X.T) + self.b1
        acti = ActiFunctions(z).tanh()

        r =  np.dot(self.w2, acti) + self.b2
        r_shifted = r - np.max(r, axis=0, keepdims=True)
        exp_z = np.exp(r_shifted)
        y_hat = exp_z / np.sum(exp_z, axis=0, keepdims=True)

        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)

        cache = {
            "X": X,
            "z": z,
            "acti": acti,
            "r": r,
            "y_hat": y_hat,
        }

        return y_hat, cache

    def backward(self, cache, y):
        X = cache["X"]
        acti = cache["acti"]
        y_hat = cache["y_hat"]
        n = y.shape[1]

        dR = (y_hat - y) / n
        dw2 = dR @ acti.T
        db2 = np.sum(dR, axis=1, keepdims=True)
        dH = self.w2.T @ dR
        dZ = dH * (1 - acti ** 2)

        dw1 = dZ @ X
        db1 = np.sum(dZ, axis=1, keepdims=True)

        gradients = {
            "w1": dw1,
            "b1": db1,
            "w2": dw2,
            "b2": db2,
        }

        return gradients

    def parameters(self):

        return {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }



