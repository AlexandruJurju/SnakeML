from Neural.train_network import *
from controller import Controller
from view import View
from model import Model

if __name__ == '__main__':
    net = NeuralNetwork()
    net.add_layer(Dense(NN_INPUT_NEURON_COUNT, NN_HIDDEN_NEURON_COUNT))
    net.add_layer(Activation(tanh, tanh_prime))
    net.add_layer(Dense(NN_HIDDEN_NEURON_COUNT, NN_OUTPUT_NEURON_COUNT))
    net.add_layer(Activation(sigmoid, sigmoid_prime))

    train_network(net)

    view = View()
    model = Model(BOARD_SIZE, START_SNAKE_SIZE, net)

    game = Controller(model, view)
    game.run()
