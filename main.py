from Neural.neural_network import *
from constants import NNVars, BoardVars, START_SNAKE_SIZE
from controller import Controller
from view import View
from model import Model

if __name__ == '__main__':
    net = NeuralNetwork()
    net.add_layer(Dense(NNVars.NN_INPUT_NEURON_COUNT, NNVars.NN_HIDDEN_NEURON_COUNT))
    net.add_layer(Activation(tanh, tanh_prime))
    net.add_layer(Dense(NNVars.NN_HIDDEN_NEURON_COUNT, NNVars.NN_OUTPUT_NEURON_COUNT))
    net.add_layer(Activation(sigmoid, sigmoid_prime))

    view = View()
    model = Model(BoardVars.BOARD_SIZE, START_SNAKE_SIZE, net)

    game = Controller(model, view)
    game.run()
