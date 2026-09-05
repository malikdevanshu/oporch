from dataset import Dataset
import numpy as np

class OPTIMIZER:
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = Dataset.prepare_data()
        self.w1 = np.random.randint(-2, 3, size=(4, 5))
        self.w2 = np.random.randint(-2, 3, size=(7, 4))
        self.b1 = np.random.rand(4)
        self.b2 = np.random.rand(7)

    def sgd_optimizer(self):
        w1 = self.w1
        w2 = self.w2
        b1 = self.b1
        b2 = self.b2
        X = self.dim_red()
        y = self.y_train
        max_iter = 100
        c1 = 1e-4
        n = X.shape[0]
        tol=1e-6
        cost_history = []

        for iteration in range(max_iter):
            z = np.dot(w1, X.T) + b1[:, None]
            non_linear = np.tanh(z)
    
            r =  np.dot(w2, non_linear) + b2[:, None]
            exp_z = np.exp(r)
            y_hat = exp_z / np.sum(exp_z, axis=0, keepdims=True)

            y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)

            cost = -(1 / n) * np.sum(y * np.log(y_hat))
    
            cost_history.append(cost)
    
            dR = (y_hat - y) / n
            dw2 = dR @ non_linear.T
            db2 = np.sum(dR, axis=1)
            dH = w2.T @ dR
            dZ = dH * (1 - non_linear**2)
    
            dw1 = dZ @ X
            db1 = np.sum(dZ, axis=1)
            
            grad_norm_sq = (
            np.sum(dw1**2)
            + np.sum(db1**2)
            + np.sum(dw2**2)
            + np.sum(db2**2)
            )
    
            if np.sqrt(grad_norm_sq) < tol:
                        break
            
            alpha = 1

            while True:
            
                w_layer1_new = w1 - alpha * dw1
                b_layer1_new = b1 - alpha * db1
                w_layer2_new = w2 - alpha * dw2
                b_layer2_new = b2 - alpha * db2
    
                z_new = np.dot(w_layer1_new, X.T) + b_layer1_new[:, None]
    
                non_linear_new = np.tanh(z_new)
                
                r_new = (
                    np.dot(w_layer2_new, non_linear_new)
                    + b_layer2_new[:, None]
                )
                r_shifted = r_new - np.max(r_new, axis=0, keepdims=True)

                exp_z_new = np.exp(r_shifted)
                y_hat_new = exp_z_new / np.sum(exp_z_new, axis=0, keepdims=True)
    
                y_hat_new = np.clip(y_hat_new, 1e-15, 1 - 1e-15)

                cost_new = -(1 / n) * np.sum(y * np.log(y_hat_new))
    
                if cost_new <= cost - c1 * alpha * grad_norm_sq:
                                break
                
                alpha*= 0.5
                
            w1 = w_layer1_new
            w2 = w_layer2_new
            b1 = b_layer1_new
            b2 = b_layer2_new
    
        return (
            w1,
            b1,
            w2,
            b2,
            cost_history,
            )   


        




def main():
    model = OPTIMIZER()
    w1, b1, w2, b2, cost = model.sgd_optimizer()
    print(w1)
    print(b1)
    print(w2)
    print(b2)
    print(cost)


if __name__ == "__main__":
    main()




