import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def softmax(x):

    shifted = x - np.max(
        x,
        axis=0,
        keepdims=True
    )

    exp_x = np.exp(shifted)

    return exp_x / np.sum(
        exp_x,
        axis=0,
        keepdims=True
    )