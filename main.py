from Neural.train_model import *
from game import Game

if __name__ == '__main__':
    net = NeuralNetwork()
    net.add_layer(Dense(NN_INPUT_NEURON_COUNT, NN_HIDDEN_NEURON_COUNT))
    net.add_layer(Activation(sigmoid, sigmoid_prime))
    net.add_layer(Dense(NN_HIDDEN_NEURON_COUNT, NN_OUTPUT_NEURON_COUNT))
    net.add_layer(Activation(sigmoid, sigmoid_prime))

    train_network(net)

    game = Game(BOARD_SIZE, START_SNAKE_SIZE, net)
    game.run()
