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
        self.inputs = None
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(-1, 1, (output_size, input_size))
        self.bias = np.random.uniform(-1, 1, (output_size, 1))

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(self.weights, self.inputs) + self.bias

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.dot(output_gradient, self.weights.T)
        weights_gradient = np.dot(self.inputs.T, output_gradient)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient

        return input_gradient


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.inputs = None
        self.activation = activation
        self.activation_derivated = activation_prime

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.activation(inputs)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return np.multiply(self.activation_derivated(self.inputs), output_gradient)


class KerasNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer: Layer):
        self.layers.append(layer)

    def reinit_layers(self):
        for layer in self.layers:
            if type(layer) is Dense:
                layer.weights = np.random.uniform(-1, 1, (layer.output_size, layer.input_size))
                layer.bias = np.random.uniform(-1, 1, (layer.output_size, 1))

    def feed_forward(self, inputs) -> np.ndarray:
        nn_input = inputs
        for layer in self.layers:
            nn_input = layer.forward(nn_input)

        return nn_input
