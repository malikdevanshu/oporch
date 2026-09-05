from dataset import Dataset
from neural_net import NeuralNet
from sgd import SGD
import numpy as np

def main():
    data = Dataset()
    X_train, X_test, y_train, y_test = data.prepare_data()
    model = NeuralNet(input_dim= 5, output_dim=7, hidden_dim=4)

    optimizer = SGD(max_iter=100, tolerance=1e-6)

    history = optimizer.optimize(model, X_train, y_train)

    print(history)


if __name__ == "__main__":
    main()
