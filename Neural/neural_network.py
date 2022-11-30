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
def mse(y_real, y_predicted):
    return np.mean(np.power(y_real - y_predicted, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


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

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))


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

    def train(self, loss, loss_prime, x_train, y_train, learning_rate) -> None:
        error = 1
        epoch = 1
        while error > 0.0001:
            error = 0
            for x, y in zip(x_train, y_train):
                output = self.feed_forward(x)

                error += loss(y, output)

                # gradient is used as the output of the whole network
                # for the penultimate layer the input gradient of the last layer is used as the output of the penultimate one
                gradient = loss_prime(y, output)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, learning_rate)

            error /= len(x_train)
            epoch += 1

            print(f"epoch = {epoch}, error = {error}")
        print()

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
