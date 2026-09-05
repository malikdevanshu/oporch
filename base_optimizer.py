from abc import ABC, abstractmethod
import numpy as np
from rate_scheduler import Scheduler
from losses import Loss
from line_search import BackTrackingLineSearch

class BaseOptimizer(ABC):
    def __init__(
            self,
            learning_rate=1.0,   #I'll further make it user defined rather than general!
            max_iter=100,
            tolerance=1e-6,
            c1=1e-4,
        ):
    
            self.learning_rate = learning_rate
            self.max_iter = max_iter
            self.tolerance = tolerance
            self.c1 = c1
    
            self.history = []

    def gradient_norm_squared(self, gradients):
        grad_norm_sq = (
                         np.sum(gradients["w1"]**2)
                         + np.sum(gradients["b1"]**2)
                         + np.sum(gradients["w2"]**2)
                         + np.sum(gradients["b2"]**2)
                         )             #Here too Later I'll make it general
     
        return grad_norm_sq


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
    

    def mini_batch(self, X, y, batch_size, shuffle):
            n_samples = X.shape[0]
            indices = np.arange(n_samples)
            if shuffle:
                np.random.shuffle(indices)
    
            for i in range(0, n_samples, batch_size):
                last = i + batch_size
                batch_indices = indices[i:last]
                X_batch = X[batch_indices]
    
                y_batch = y[:, batch_indices]
    
                yield X_batch, y_batch

    def scheduled_alpha(self, final_alpha=0.01, tau=10000, decay=1e-4):
            scheduler = Scheduler(learning_rate=self.learning_rate, final_alpha=final_alpha, tau=tau, decay=decay)
    
            return scheduler

    def optimize_mini_batch(
        self,
        model,
        X,
        y,
        batch_size=128,
    ):

        self.history = []

        step_count = 0

        scheduler = self.scheduled_alpha()
        cost = Loss()

        self.initialize_state(model)

        for epoch in range(self.max_iter):

            epoch_loss = 0.0
            num_batches = 0

            for X_batch, y_batch in self.mini_batch(
                X,
                y,
                batch_size=batch_size,
                shuffle=True,
            ):

                alpha = scheduler.learning_rate_schedule(
                    step_count=step_count
                )

                y_hat, cache = model.forward_pass(X_batch)
                n_batch = X_batch.shape[0]

                batch_loss = cost.multiclass_cross_entropy(
                    y_hat,
                    y_batch,
                    n_batch
                )

                gradients = model.backward(
                    cache,
                    y_batch
                )

                self.update(
                    model,
                    gradients,
                    alpha
                )

                step_count += 1

                epoch_loss += batch_loss
                num_batches += 1

            average_loss = epoch_loss / num_batches

            self.history.append(average_loss)

            print(
                f"Epoch {epoch + 1}: "
                f"loss = {average_loss:.6f}, "
                f"alpha = {alpha:.6f}"
            )

        return self.history

    def initialize_state(self, model):
        pass

    @abstractmethod
    def update(
        self,
        model,
        gradients,
        alpha,
    ):
        pass
    
    
    