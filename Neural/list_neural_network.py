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

        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

        # self.weights = np.random.uniform(-1, 1, (output_size, input_size))
        # self.bias = np.random.uniform(-1, 1, (output_size, 1))

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

    def add(self, layer: Layer):
        self.layers.append(layer)

    def reinit_layers(self):
        for layer in self.layers:
            if type(layer) is Dense:
                layer.weights = np.random.uniform(-1, 1, (layer.output_size, layer.input_size))
                layer.bias = np.random.uniform(-1, 1, (layer.output_size, 1))

    def feed_forward(self, input) -> np.ndarray:
        nn_input = input
        for layer in self.layers:
            nn_input = layer.forward(nn_input)

        return nn_input

    def predict(self, input):
        return self.feed_forward(input)

    def train(self, loss, loss_prime, x_train, y_train, epochs, learning_rate):
        for i in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                # forward
                output = self.predict(x)

                # error
                error += loss(y, output)

                # backward
                grad = loss_prime(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            error /= len(x_train)

            print(f"{i + 1}/{epochs}, error={error}")
