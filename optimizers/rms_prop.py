import numpy as np
from .base_optimizer import BaseOptimizer

class RMSPROP(BaseOptimizer):
    def __init__(self, scaler=0.9, eps=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.scaler = scaler
        self.eps = eps
        self.running_average = None
    def initialize_state(self, model):

            self.running_average = {
            "w1": np.zeros_like(model.w1),
            "b1": np.zeros_like(model.b1),
            "w2": np.zeros_like(model.w2),
            "b2": np.zeros_like(model.b2),
        }

    
    def update(
            self,
            model,
            gradients,
            alpha,
        ):
        self.running_average["w1"] = self.scaler * self.running_average["w1"] + (1 - self.scaler) * gradients["w1"]**2
        self.running_average["w2"] = self.scaler * self.running_average["w2"] + (1 - self.scaler) * gradients["w2"]**2
        self.running_average["b1"] = self.scaler * self.running_average["b1"] + (1 - self.scaler) * gradients["b1"]**2
        self.running_average["b2"] = self.scaler * self.running_average["b2"] + (1 - self.scaler) * gradients["b2"]**2


        model.w1 -= (alpha * gradients["w1"] / np.sqrt(self.running_average["w1"] + self.eps))
        model.b1 -= (alpha * gradients["b1"] / np.sqrt(self.running_average["b1"] + self.eps))

        model.w2 -= (alpha * gradients["w2"] / np.sqrt(self.running_average["w2"] + self.eps))
        model.b2 -= (alpha * gradients["b2"] / np.sqrt(self.running_average["b2"] + self.eps))
        