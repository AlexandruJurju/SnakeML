import genetic_operators
from Neural.neural_network import *
from settings import NNSettings, SnakeSettings
from constants import BoardConsts, State
from game import Game, train_network
from model import Model
from genetic_operators import *


if __name__ == '__main__':
    net = NeuralNetwork()
    net.add_layer(Dense(4, 8))
    net.add_layer(Activation(tanh, tanh_prime))
    net.add_layer(Dense(8, 4))
    net.add_layer(Activation(sigmoid, sigmoid_prime))

    net2 = NeuralNetwork()
    net2.add_layer(Dense(4, 8))
    net2.add_layer(Activation(tanh, tanh_prime))
    net2.add_layer(Dense(8, 4))
    net2.add_layer(Activation(sigmoid, sigmoid_prime))

    weight1 = net.get_dense_layers()[0].weights
    weight2 = net2.get_dense_layers()[0].weights



    # model = Model(BoardConsts.BOARD_SIZE, SnakeSettings.START_SNAKE_SIZE, net)
    #
    # train_network(model.snake.brain)
    #
    # game = Game(model, State.MAIN_MENU)
    # game.state_machine()
