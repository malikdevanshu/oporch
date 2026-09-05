import numpy as np

class Loss:
    def cross_entropy(self, y_hat, y, n):
        
        loss = -(1 / n) * (y * np.log(y_hat) - (1 - y)* np.log(1 - y_hat))
        return loss 

    def cross_entropy_gradient(self, y_hat, y, n):
        dR = (y_hat - y) / n

        return dR

    def multiclass_cross_entropy(self, y_hat, y, n):
        loss = -(1 / n) * np.sum(y * np.log(y_hat))
        return loss