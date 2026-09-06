from .base_optimizer import BaseOptimizer
import numpy as np

class SGDM(BaseOptimizer):
    def __init__(self, momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum
        self.velocity = None

    def initialize_state(self, model):

        self.velocity = {
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

        self.velocity["w1"] = (
            self.momentum * self.velocity["w1"]
            + gradients["w1"]
        )

        self.velocity["b1"] = (
            self.momentum * self.velocity["b1"]
            + gradients["b1"]
        )

        self.velocity["w2"] = (
            self.momentum * self.velocity["w2"]
            + gradients["w2"]
        )

        self.velocity["b2"] = (
            self.momentum * self.velocity["b2"]
            + gradients["b2"]
        )

        model.w1 -= alpha * self.velocity["w1"]
        model.b1 -= alpha * self.velocity["b1"]

        model.w2 -= alpha * self.velocity["w2"]
        model.b2 -= alpha * self.velocity["b2"]

    