from dataset import Dataset
from neural_net import NeuralNet
from optimizer import GradientDescent
import numpy as np

def main():
    data = Dataset()
    X_train, X_test, y_train, y_test = data.prepare_data()
    model = NeuralNet(input_dim= 5, output_dim=7, hidden_dim=4)

    optimizer = GradientDescent(max_iter=100, tolerance=1e-6)

    history = optimizer.optimize_mini_batch(model, X_train, y_train)

    print(history)


if __name__ == "__main__":
    main()
