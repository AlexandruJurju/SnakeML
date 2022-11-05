import numpy as np
from Neural.neural_network_utils import *


class Layer:
    def __init__(self):
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        pass

    def backward(self, output_gradient, learning_rate):
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.uniform(-1, 1, (output_size, input_size))
        self.bias = np.random.uniform(-1, 1, (output_size, 1))

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(self.weights, self.inputs) + self.bias

    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.inputs.T, output_error)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error

        return input_error


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_derivated = activation_prime

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.activation(inputs)
        return self.output

    def backward(self, output_error, learning_rate):
        return np.multiply(self.activation_derivated(self.inputs), output_error)
