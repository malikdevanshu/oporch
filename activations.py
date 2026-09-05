import numpy as np

class ActiFunctions():
    def __init__(self, x):
        self.x = x

    def tanh(self):
        return np.tanh(self.x)


    def tanh_derivative(self):
        return 1 - np.tanh(self.x) ** 2


    def softmax(self):

        shifted = self.x - np.max(
            self.x,
            axis=0,
            keepdims=True
        )

        exp_x = np.exp(shifted)

        return exp_x / np.sum(
            exp_x,
            axis=0,
            keepdims=True
        )