import numpy as np
from Neural.neural_network_utils import *


class NeuralNetwork:
    def __init__(self, nn_architecture: {}, hidden_activation=relu, output_activation=softmax):
        self.architecture = nn_architecture
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.weights = {}
        self.biases = {}
        self.outputs = {}
        self.input = []
        self.init_random_network()

    def init_random_network(self) -> None:
        for layer in self.architecture:
            self.weights[layer] = np.random.uniform(-1, 1, (self.architecture[layer][1], self.architecture[layer][0]))
            self.biases[layer] = np.random.uniform(-1, 1, (self.architecture[layer][1], 1))

    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        self.input = inputs
        for i, layer in enumerate(self.architecture):
            layer_weights = self.weights[layer]
            layer_biases = self.biases[layer]
            layer_output = np.dot(layer_weights, inputs) + layer_biases

            if i != len(self.architecture) - 1:
                output = self.hidden_activation(layer_output)
                self.outputs[layer] = output
                inputs = output
            else:
                output = self.output_activation(layer_output)
                self.outputs[layer] = output
                return output

    def feed_back(self, output_gradient, learning_rate) -> np.ndarray:
        for i, layer in reversed(list(enumerate(self.architecture))):
            layer_weight_gradient = np.dot(output_gradient, np.transpose(self.input))
            self.weights[layer] -= learning_rate * layer_weight_gradient
            self.biases[layer] -= learning_rate * output_gradient

            if i == len(self.architecture) - 1:
                return np.dot(np.transpose(self.weights[layer]), output_gradient)
