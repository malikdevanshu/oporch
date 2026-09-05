import numpy as np

class Loss:
    def __init__(self,y_hat, y):
        self.y_hat = y_hat
        self.y = y
        self.out_dim = y.shape[1]

    def cross_entropy(self):
        n = self.out_dim
        loss = -(1 / n) * (self.y * np.log(self.y_hat) - (1 - self.y)* np.log(1 - self.y_hat))
        return loss 

    def cross_entropy_gradient(self):
        dR = (self.y_hat - self.y) / self.out_dim

        return dR

    def multiclass_cross_entropy(self):
        n = self.out_dim
        loss = -(1 / n) * np.sum(self.y * np.log(self.y_hat))
        return loss