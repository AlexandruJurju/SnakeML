import random

from Neural.list_neural_network import *
from constants import MAIN_DIRECTIONS


class Snake:
    def __init__(self, neural_net: KerasNetwork):
        self.body = []
        self.brain = neural_net
        self.direction = random.choice(MAIN_DIRECTIONS)
        self.vision_lines = {}
