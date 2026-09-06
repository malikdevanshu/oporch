import numpy as np
from utils.losses import Loss
from .base_optimizer import BaseOptimizer
from .line_search import BackTrackingLineSearch

class FullGD(BaseOptimizer):
    def update(
            self,
            model,
            gradients,     #it doen't have to be inherited from the base class this is seperate so fix it!
            alpha,
        ):
        model.w1 -= alpha * gradients["w1"]
        model.b1 -= alpha * gradients["b1"]

        model.w2 -= alpha * gradients["w2"]
        model.b2 -= alpha * gradients["b2"]
        
    def optimize(self, model, X, y):
            n = y.shape[1]
            cost = Loss()
            search = BackTrackingLineSearch(c1= 1e-4, reduction = 0.5)
    
            for iteration in range(self.max_iter):
    
                y_hat, cache = model.forward_pass(X)
    
                loss = cost.multiclass_cross_entropy(y_hat, y, n)#-(1 / n) * np.sum(y * np.log(y_hat))
    
                gradients = model.backward(
                    cache, 
                    y,
                )
    
                grad_norm = self.gradient_norm_squared(
                    gradients
                )
    
                if np.sqrt(grad_norm) < self.tolerance:
                    break
    
                print("iteration:", iteration)
                print("loss before step:", loss)
                print("gradient norm:", np.sqrt(grad_norm))
    
    
                alpha = search.line_search(
                    model,
                    X,
                    y,
                    gradients,
                    loss,
                    self.learning_rate
                )
                print("alpha:", alpha)
    
                self.update(
                    model,
                    gradients,
                    alpha,
                )
    
                new_y_hat, _ = model.forward_pass(X)
    
                new_loss = -(1 / n) * np.sum(
                    y * np.log(new_y_hat)
                )
    
                print("loss after step:", new_loss)
                print("-------------------")
    
                self.history.append(new_loss)
    
            return self.history   

    







            


    
    
    


    



