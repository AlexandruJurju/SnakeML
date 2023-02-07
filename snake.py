import random

from Neural.neural_network import *
from constants import MAIN_DIRECTIONS


class Snake:
    def __init__(self, neural_net: NeuralNetwork, direction: None):
        self.body = []
        self.brain = neural_net
        self.ttl = 50

        if direction is None:
            self.direction = random.choice(MAIN_DIRECTIONS)
