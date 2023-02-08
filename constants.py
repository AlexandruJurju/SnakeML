from enum import Enum

WIDTH, HEIGHT = 1000, 800
SQUARE_SIZE = 25
COLOR_BLACK = (0, 0, 0)
COLOR_BACKGROUND = (47, 47, 47)
COLOR_WHITE = (255, 255, 255)
COLOR_SNAKE = (0, 128, 255)
COLOR_SNAKE_HEAD = (128, 0, 128)
COLOR_APPLE = (199, 55, 47)
COLOR_SQUARE_DELIMITER = (64, 64, 64)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)

MAX_FPS = 20

OFFSET_BOARD_X = 50
OFFSET_BOARD_Y = 50

VISION_LINES_RETURN_TYPE = "boolean"
BOARD_SIZE = 10

INPUT_DIRECTION_COUNT = 8
VISION_LINES_COUNT = INPUT_DIRECTION_COUNT

NN_INPUT_NEURON_COUNT = VISION_LINES_COUNT * 3 + 4
NN_HIDDEN_NEURON_COUNT = 24
NN_OUTPUT_NEURON_COUNT = 4

START_SNAKE_SIZE = 3
SNAKE_MAX_TTL = 50


class BoardLetters:
    APPLE = "A"
    WALL = "W"
    EMPTY = "."
    SNAKE_BODY = "S"
    SNAKE_HEAD = "H"


class Direction(Enum):
    UP = [-1, 0]
    DOWN = [1, 0]
    LEFT = [0, -1]
    RIGHT = [0, 1]
    Q1 = [-1, 1]
    Q2 = [1, 1]
    Q3 = [1, -1]
    Q4 = [-1, -1]


DYNAMIC_DIRECTIONS = ["STRAIGHT", "LEFT", "RIGHT"]
MAIN_DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
ALL_DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT, Direction.Q1, Direction.Q2, Direction.Q3, Direction.Q4]
