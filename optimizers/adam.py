import numpy as np
from .base_optimizer import BaseOptimizer

class ADAM(BaseOptimizer):
    def __init__(self, beta=0.9, gamma= 0.999, eps=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.velocity  = None
        self.running_average = None
        self.t = 0

    def initialize_state(self, model):

        self.running_average = {
        "w1": np.zeros_like(model.w1),
        "b1": np.zeros_like(model.b1),
        "w2": np.zeros_like(model.w2),
        "b2": np.zeros_like(model.b2),
    }
        self.velocity = {
        "w1": np.zeros_like(model.w1),
        "b1": np.zeros_like(model.b1),
        "w2": np.zeros_like(model.w2),
        "b2": np.zeros_like(model.b2),
    }
        self.t = 0


    def update(
                self,
                model,
                gradients,
                alpha,
            ):
        self.t += 1

        self.velocity["w1"] = self.beta * self.velocity["w1"] + (1 - self.beta) * gradients["w1"]
        self.velocity["w2"] = self.beta * self.velocity["w2"] + (1 - self.beta) * gradients["w2"]
        self.velocity["b1"] = self.beta * self.velocity["b1"] + (1 - self.beta) * gradients["b1"]
        self.velocity["b2"] = self.beta * self.velocity["b2"] + (1 - self.beta) * gradients["b2"]


        self.running_average["w1"] = self.gamma * self.running_average["w1"] + (1 - self.gamma) * gradients["w1"]**2
        self.running_average["w2"] = self.gamma * self.running_average["w2"] + (1 - self.gamma) * gradients["w2"]**2
        self.running_average["b1"] = self.gamma * self.running_average["b1"] + (1 - self.gamma) * gradients["b1"]**2
        self.running_average["b2"] = self.gamma * self.running_average["b2"] + (1 - self.gamma) * gradients["b2"]**2

        velocity_scaled = {
            key: value / (1 - self.beta ** self.t)
            for key, value in self.velocity.items()
        }

        running_average_scaled = {
            key: value / (1 - self.gamma ** self.t)
            for key, value in self.running_average.items()
        }

        model.w1 -= (alpha * velocity_scaled["w1"]) / (np.sqrt(running_average_scaled["w1"]) + self.eps)
        model.w2 -= (alpha * velocity_scaled["w2"]) / (np.sqrt(running_average_scaled["w2"]) + self.eps)
        model.b1 -= (alpha * velocity_scaled["b1"]) / (np.sqrt(running_average_scaled["b1"]) + self.eps)
        model.b2 -= (alpha * velocity_scaled["b2"]) / (np.sqrt(running_average_scaled["b2"]) + self.eps)



 