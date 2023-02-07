import numpy as np


def relu(x):
    return np.maximum(0.0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


# use np because y_real and y_predicted are vectors of values
# mse returns a scalar, MSE of all errors in output
def mse(target_y, predicted_y):
    real_mse = np.mean(np.power(target_y - predicted_y, 2))
    return real_mse


# mse prime returns a vector of dE/dY, output gradient vector for output vector
def mse_prime(target_y, predicted_y):
    return (2 / np.size(target_y)) * (predicted_y - target_y)


class Layer:
    def __init__(self):
        self.inputs: int
        self.output = None

    def forward(self, inputs):
        pass

    def backward(self, output_gradient, learning_rate):
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input = None

        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.uniform(-1, 1, (output_size, input_size))
        self.bias = np.random.uniform(-1, 1, (output_size, 1))

        # self.weights = np.random.randn(output_size, input_size)
        # self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    # using weights = -learning_rate * weights_gradient because normally gradient goes to maximum of the plane
    # with subtraction it goes to the minimum of the plane, minimises the error
    def backward(self, output_gradient, learning_rate):
        input_gradient = np.dot(self.weights.T, output_gradient)
        weights_gradient = np.dot(output_gradient, self.input.T)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient

        return input_gradient


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.input = None
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, inputs):
        self.input = inputs
        self.output = self.activation(inputs)
        return self.output

    # compute input gradient dE/dX = dE/dY * f'(X)
    # self.output seems to work better
    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient * self.activation_prime(self.input)
        # input_gradient = np.multiply(output_gradient, self.activation_prime(self.input))

        return input_gradient


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)

    def feed_forward(self, input) -> np.ndarray:
        nn_input = input
        for layer in self.layers:
            nn_input = layer.forward(nn_input)
        return nn_input

    def get_dense_layers(self) -> [Layer]:
        dense_layers = []
        for layer in self.layers:
            if type(layer) is Dense:
                dense_layers.append(layer)
        return dense_layers

    def reinit_weights_and_biases(self):
        dense_layers = self.get_dense_layers()

        for layer in dense_layers:
            layer.weights = np.random.uniform(-1, 1, (layer.output_size, layer.input_size))
            layer.bias = np.random.uniform(-1, 1, (layer.output_size, 1))

    def train(self, loss, loss_prime, x_train, y_train, learning_rate) -> None:
        error = 10000
        epoch = 0
        # error doesn't have to be too small, when it gets rounded it doesn't matter too much
        while error > 0.1:
            error = 0
            for x, y in zip(x_train, y_train):
                output = self.feed_forward(x)

                error += loss(y, output)

                # gradient is used as the output error / dE/dY of the whole network
                # input gradient of last layer is considered output gradient of penultimate layer
                gradient = loss_prime(y, output)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, learning_rate)

            # error /= len(x_train)
            epoch += 1

            print(f"epoch = {epoch}, error = {error}")
        print(f"final error {error}  \n")

    def print_weights_and_biases(self) -> None:
        dense_layers = []
        for layer in self.layers:
            if type(layer) is Dense:
                dense_layers.append(layer)

        for i, layer in enumerate(dense_layers):
            print("Dense layer : " + str(i + 1))
            print("=============== WEIGHTS ===============")
            print(layer.weights)

            print("=============== BIASES ===============")
            print(layer.bias)
            print("===============")
            print()
