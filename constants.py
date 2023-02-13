from enum import Enum


class State(Enum):
    RUN_BEST_GENETIC = 1
    MAIN_MENU = 2
    OPTIONS_BACKPROPAGATION = 3
    RUN_BACKPROPAGATION = 4
    RUN_BACKWARD_TRAIN = 5
    OPTIONS_GENETIC = 6
    RUN_GENETIC = 7


class ViewConsts:
    DRAW = True

    MAX_FPS = 5
    OFFSET_BOARD_X = 500
    OFFSET_BOARD_Y = 100
    WIDTH, HEIGHT = 1000, 800
    SQUARE_SIZE = 25
    WINDOW_START_X, WINDOW_START_Y = 50, 50

    NN_DISPLAY_OFFSET_X = 50
    NN_DISPLAY_OFFSET_Y = 150
    NN_DISPLAY_LABEL_HEIGHT_BETWEEN = 8
    NN_DISPLAY_lABEL_OFFSET_X = NN_DISPLAY_OFFSET_X - 40
    NN_DISPLAY_NEURON_WIDTH_BETWEEN = 100
    NN_DISPLAY_NEURON_HEIGHT_BETWEEN = NN_DISPLAY_LABEL_HEIGHT_BETWEEN
    NN_DISPLAY_NEURON_OFFSET_X = NN_DISPLAY_OFFSET_X + 100
    NN_DISPLAY_NEURON_OFFSET_Y = NN_DISPLAY_OFFSET_Y
    NN_DISPLAY_NEURON_RADIUS = 8

    WINDOW_TITLE_X, WINDOW_TITLE_Y = (WIDTH / 2, 10)

    COLOR_BLACK = (0, 0, 0)
    COLOR_BACKGROUND = (47, 47, 47)
    COLOR_WHITE = (255, 255, 255)
    COLOR_SNAKE_SEGMENT = (30, 144, 255)
    COLOR_SNAKE_HEAD = (128, 0, 128)
    COLOR_APPLE = (199, 55, 47)
    COLOR_SQUARE_DELIMITER = (64, 64, 64)
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (255, 0, 0)


class BoardConsts:
    BOARD_SIZE = 10

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
