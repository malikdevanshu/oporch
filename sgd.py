import numpy as np
from base_optimizer import BaseOptimizer
from losses import Loss

class SGD(BaseOptimizer):
    def update(
            self,
            model,
            gradients,
            alpha,
        ):
        model.w1 -= alpha * gradients["w1"]
        model.b1 -= alpha * gradients["b1"]

        model.w2 -= alpha * gradients["w2"]
        model.b2 -= alpha * gradients["b2"]

        






    
                


        
        
        


        



