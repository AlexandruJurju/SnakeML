from constants import BoardConsts, State
from game import Game, train_network
from genetic_operators import *
from model import Model
from settings import NNSettings, SnakeSettings

if __name__ == '__main__':
    net = NeuralNetwork()
    net.add_layer(Dense(NNSettings.INPUT_NEURON_COUNT, NNSettings.HIDDEN_NEURON_COUNT))
    net.add_layer(Activation(tanh, tanh_prime))
    net.add_layer(Dense(NNSettings.HIDDEN_NEURON_COUNT, NNSettings.OUTPUT_NEURON_COUNT))
    net.add_layer(Activation(sigmoid, sigmoid_prime))

    model = Model(BoardConsts.BOARD_SIZE, SnakeSettings.START_SNAKE_SIZE, net)

    # train_network(model.snake.brain)

    game = Game(model, State.MAIN_MENU)
    game.state_machine()
