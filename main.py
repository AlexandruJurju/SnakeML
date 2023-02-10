from Neural.neural_network import *
from settings import NNVars, BoardVars, START_SNAKE_SIZE
from game import Game, train_network
from model import Model
from settings import States

if __name__ == '__main__':
    net = NeuralNetwork()
    net.add_layer(Dense(NNVars.NN_INPUT_NEURON_COUNT, NNVars.NN_HIDDEN_NEURON_COUNT))
    net.add_layer(Activation(tanh, tanh_prime))
    net.add_layer(Dense(NNVars.NN_HIDDEN_NEURON_COUNT, NNVars.NN_OUTPUT_NEURON_COUNT))
    net.add_layer(Activation(sigmoid, sigmoid_prime))

    model = Model(BoardVars.BOARD_SIZE, START_SNAKE_SIZE, net)

    train_network(model.snake.brain)

    game = Game(model, States.MAIN_MENU)
    game.state_machine()
