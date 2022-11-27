from Neural.train_model import *
from game import Game

if __name__ == '__main__':
    net = NeuralNetwork()
    net.add_layer(Dense(VISION_LINES_COUNT * 3, 16))
    net.add_layer(Activation(tanh, tanh_prime))
    net.add_layer(Dense(16, 3))
    net.add_layer(Activation(sigmoid, sigmoid_prime))

    # TODO folosirea datelor de antrenament pentru o tabla mai mare nu merge prea bine
    train_network(net)

    game = Game(BOARD_SIZE, 3, net)
    game.run()
