import numpy as np
from base_optimizer import BaseOptimizer
from losses import Loss

class SGD(BaseOptimizer):
    def optimize(self, model, X, y):
        n = y.shape[1]
        cost = Loss()

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


            alpha = self.line_search(
                model,
                X,
                y,
                gradients,
                loss,
            )
            print("alpha:", alpha)

            self.step(
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

    
                


        
        
        


        



