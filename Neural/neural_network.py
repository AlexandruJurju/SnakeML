import numpy as np


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

    def train(self, loss, loss_prime, x_train, y_train, epochs, learning_rate) -> None:
        error = 1
        while error > 0.0001:
            error = 0
            for x, y in zip(x_train, y_train):
                output = self.feed_forward(x)

                error += loss(y, output)

                gradient = loss_prime(y, output)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, learning_rate)

            error /= len(x_train)

            print(f"error = {error}")
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
