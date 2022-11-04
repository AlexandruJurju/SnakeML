import numpy as np
from neural_network_utils import *


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

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, np.transpose(self.inputs))
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient


class Activation(Layer):
    def __init__(self, activation, activation_derivated):
        super().__init__()
        self.activation = activation
        self.activation_derivated = activation_derivated

    def forward(self, inputs):
        self.inputs = inputs
        return self.activation(inputs)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_derivated(self.inputs))
