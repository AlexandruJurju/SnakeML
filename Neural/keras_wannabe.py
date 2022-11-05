from Neural.keras_wannabe_classes import *


class KerasNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer: Layer):
        self.layers.append(layer)

    def feed_forward(self, inputs) -> np.ndarray:
        nn_input = inputs
        for layer in self.layers:
            nn_input = layer.forward(nn_input)

        return nn_input
