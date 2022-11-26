from game import *

if __name__ == "__main__":
    # game = Game(10, 3)
    # game.run()

    net = NeuralNetwork()
    net.add(Dense(28, 16))
    net.add(Activation(tanh, tanh_prime))
    net.add(Dense(16, 3))
    net.add(Activation(tanh, tanh_prime))
