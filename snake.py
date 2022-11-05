from Neural.keras_wannabe import *


class Snake:
    def __init__(self, neural_net: KerasNetwork):
        self.body = []
        self.brain = neural_net
        self.direction = None
        self.vision_lines = {}
