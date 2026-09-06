from data.dataset import Dataset
from network.neural_net import NeuralNet
from optimizers.adam import ADAM

def main():
    data = Dataset()
    X_train, X_test, y_train, y_test = data.prepare_data()
    model = NeuralNet(input_dim= 5, output_dim=7, hidden_dim=4)

    optimizer = ADAM(learning_rate=30, max_iter=100, tolerance=1e-6)

    history = optimizer.optimize_mini_batch(model, X_train, y_train)

    print(history)


if __name__ == "__main__":
    main()
