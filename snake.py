from constants import Direction
from neural_network_dictionary import *


class Snake:
    def __init__(self):
        self.body = []
        # self.brain = neural_network
        self.direction = None
        self.vision_lines = {}
