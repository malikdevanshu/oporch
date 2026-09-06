import numpy as np
class BackTrackingLineSearch:
    def __init__(self, c1=1e-4, reduction=0.5):
        self.c1 = c1
        self.reduction = reduction 

    def line_search(self, model,
                X, y, gradients,
                current_loss,
                initial_alpha,
                ):

    
            alpha = initial_alpha
    
            grad_norm_sq = (
                         np.sum(gradients["w1"]**2)
                         + np.sum(gradients["b1"]**2)
                         + np.sum(gradients["w2"]**2)
                         + np.sum(gradients["b2"]**2)
                         )
    
            # Save current parameters
            original_w1 = model.w1.copy()
            original_b1 = model.b1.copy()
            original_w2 = model.w2.copy()
            original_b2 = model.b2.copy()
            accepted = False
    
            while alpha > 1e-12:
                model.w1 = original_w1 - alpha * gradients["w1"]
                model.b1 = original_b1 - alpha * gradients["b1"]
    
                model.w2 = original_w2 - alpha * gradients["w2"]
                model.b2 = original_b2 - alpha * gradients["b2"]
    
                y_hat, _ = model.forward_pass(X)
    
                n = y.shape[1]
    
                new_loss = -(1 / n) * np.sum(
                    y * np.log(y_hat)
                )
                if new_loss <= (
                    current_loss
                    - self.c1 * alpha * grad_norm_sq
                ):
                    
                    accepted = True
                    break
    
                alpha *= self.reduction
            model.w1 = original_w1
            model.b1 = original_b1
    
            model.w2 = original_w2
            model.b2 = original_b2
    
            if not accepted:
                return 0.0
    
            return alpha