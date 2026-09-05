class Scheduler:
    def __init__(self, learning_rate, final_alpha = 0.01, tau = 10000, decay=1e-4):
          self.learning_rate = learning_rate
          self.final_alpha = final_alpha
          self.tau = tau
          self.decay=decay

    def constant_rate(self, learning_rate):
          return learning_rate

    

    def learning_rate_decay(self,step_count):
            return self.learning_rate / (
            1 + self.decay * step_count
        )

    def exponential_rate_decay(self, step_count):
          staircase = step_count / self.tau
          return self.learning_rate * (self.decay ** staircase)


    def learning_rate_schedule(self, step_count
        ):
            if step_count >= self.tau:
                return self.final_alpha

            decay = step_count / self.tau

            alpha = (
                (1 - decay) * self.learning_rate
                + decay * self.final_alpha
            )

            return alpha
