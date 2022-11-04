from model import *
from game import *
from neural_network_dictionary import *
from neural_network_utils import *

if __name__ == "__main__":
    # 8 vision lines each with 3 values = 24
    # 4 snake directions : 1 if it's going there 0 if not = 4
    nn_config = {
        "L1": [28, 16],
        "L2": [16, 4]
    }

    nn = NeuralNetwork(nn_config, relu, softmax)

    display = Game(Model(10, 3))
    display.run()
