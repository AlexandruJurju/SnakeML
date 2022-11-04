from constants import Direction
from neural_network_dictionary import *


class Snake:
    def __init__(self, neural_net: NeuralNetwork):
        self.body = []
        self.brain = neural_net
        self.direction = None
        self.vision_lines = {}
