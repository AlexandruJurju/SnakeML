from game import *
from Neural.keras_wannabe import *

if __name__ == "__main__":
    # 8 vision lines each with 3 values = 24
    # 4 snake directions : 1 if it's going there 0 if not = 4

    display = Game(10, 3)
    display.run()

    net = KerasNetwork()
    net.add(Dense(1, 2))
    net.add(Activation(relu, relu))
    net.add(Dense(2, 1))
    net.add(Activation(softmax, softmax))

    net.feed_forward([0.24])

