import random

from Neural.neural_network import *
from settings import SnakeSettings
from constants import *


class Snake:
    def __init__(self, neural_net: NeuralNetwork, direction: None):
        self.body = []
        self.brain = neural_net
        self.ttl = SnakeSettings.SNAKE_MAX_TTL
        self.score = 0

        if direction is None:
            self.direction = random.choice(MAIN_DIRECTIONS)
