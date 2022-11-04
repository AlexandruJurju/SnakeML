from enum import Enum

WIDTH, HEIGHT = 1250, 750
SQUARE_SIZE = 25
COLOR_BLACK = (0, 0, 0)
COLOR_BACKGROUND = (47, 47, 47)
COLOR_WHITE = (255, 255, 255)
COLOR_SNAKE = (0, 128, 255)
COLOR_SNAKE_HEAD = (128, 0, 128)
COLOR_APPLE = (199, 55, 47)
COLOR_SQUARE_DELIMITER = (64, 64, 64)

MAX_FPS = 0.0001

OFFSET_BOARD_X = 50
OFFSET_BOARD_Y = 50


class Direction(Enum):
    UP = [-1, 0]
    DOWN = [1, 0]
    LEFT = [0, -1]
    RIGHT = [0, 1]
    Q1 = [-1, 1]
    Q2 = [1, 1]
    Q3 = [1, -1]
    Q4 = [-1, -1]


MAIN_DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
ALL_DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT, Direction.Q1, Direction.Q2, Direction.Q3, Direction.Q4]
