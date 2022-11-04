from model import *
from game import *
from neural_network_dictionary import *
from neural_network_utils import *

if __name__ == "__main__":
    # 8 vision lines each with 3 values = 24
    # 4 snake directions : 1 if it's going there 0 if not = 4

    display = Game(10, 3)
    display.run()
