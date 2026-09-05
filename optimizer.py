import numpy as np
from losses import Loss
from rate_scheduler import Scheduler

class GradientDescent:
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

    def line_search(self, model,
            X, y, gradients,
            current_loss,
            ):

        alpha = self.learning_rate

        grad_norm_sq = self.gradient_norm_squared(gradients)

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

            alpha *= 0.5
        model.w1 = original_w1
        model.b1 = original_b1

        model.w2 = original_w2
        model.b2 = original_b2

        if not accepted:
            return 0.0

        return alpha

    def step(
        self,
        model,
        gradients,
        alpha,
    ):

        model.w1 -= alpha * gradients["w1"]
        model.b1 -= alpha * gradients["b1"]

        model.w2 -= alpha * gradients["w2"]
        model.b2 -= alpha * gradients["b2"]

    def optimize(self, model, X, y):
        n = y.shape[1]

        for iteration in range(self.max_iter):

            y_hat, cache = model.forward_pass(X)

            loss = -(1 / n) * np.sum(y * np.log(y_hat))

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

            self.history.append(loss)

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



    def scheduled_alpha(self, final_alpha=0.01, tau=10000):
        scheduler = Scheduler(learning_rate=self.learning_rate, final_alpha=final_alpha, tau=tau)

        return scheduler

    def optimize_mini_batch( self,
    model,
    X,
    y,
    batch_size=128,):

        self.history = []
        step_count = 0

        for epoch in range(self.max_iter):

            epoch_loss = 0
            num_batches = 0

            for X_batch, y_batch in self.mini_batch(
                X,
                y,
                batch_size=batch_size,
                shuffle=True,
                ):
                alpha = self.scheduled_alpha().learning_rate_schedule(step_count=step_count)

                y_hat, cache = model.forward_pass(X_batch)

                n_batch = X_batch.shape[0]

                batch_loss = -(1 / n_batch) * np.sum(
                y_batch * np.log(y_hat)
            )
                gradients = model.backward(cache, y_batch)

                self.step(
                model,
                gradients,
                alpha,
            )
                step_count += 1

                epoch_loss += batch_loss
                num_batches += 1

            average_loss = epoch_loss / num_batches

            self.history.append(average_loss)

            print(
            f"Epoch {epoch + 1}: "
            f"loss = {average_loss:.6f}"
            f"Alpha: {alpha:.6f}"
        )
        return self.history



                


        
        
        


        



