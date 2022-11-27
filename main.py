from Neural.neural_network import *
from Neural.neural_network_functions import *
from game import Game
from Neural.train_model import *

if __name__ == '__main__':
    net = NeuralNetwork()
    net.add_layer(Dense(12, 16))
    net.add_layer(Activation(tanh, tanh_prime))
    net.add_layer(Dense(16, 3))
    net.add_layer(Activation(sigmoid, sigmoid_prime))

    train_network(net)

    game = Game(10, 3, net)
    game.run()
