from Neural.neural_network import *
from settings import NNSettings, SnakeSettings
from constants import BoardConsts, State
from game import Game, train_network
from model import Model

if __name__ == '__main__':
    net = NeuralNetwork()
    net.add_layer(Dense(NNSettings.NN_INPUT_NEURON_COUNT, NNSettings.NN_HIDDEN_NEURON_COUNT))
    net.add_layer(Activation(tanh, tanh_prime))
    net.add_layer(Dense(NNSettings.NN_HIDDEN_NEURON_COUNT, NNSettings.NN_OUTPUT_NEURON_COUNT))
    net.add_layer(Activation(sigmoid, sigmoid_prime))

    model = Model(BoardConsts.BOARD_SIZE, SnakeSettings.START_SNAKE_SIZE, net)

    train_network(model.snake.brain)

    game = Game(model, State.MAIN_MENU)
    game.state_machine()
