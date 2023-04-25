from enum import Enum


class State(Enum):
    QUIT = 0
    MAIN_MENU = 1
    BACKPROPAGATION_MENU = 2
    GENETIC_MENU = 3
    GENETIC_TRAIN_NEW_NETWORK = 4
    BACKPROPAGATION_TRAIN_NEW_NETWORK = 5
    OPTIONS = 6
    RUN_PRETRAINED = 7


class BoardConsts:
    APPLE = "A"
    WALL = "W"
    EMPTY = "."
    SNAKE_BODY = "S"
    SNAKE_HEAD = "H"


class ViewSettings:
    DRAW = True
    DARK_MODE = True
    MAX_FPS = 40
    WIDTH, HEIGHT = 1366, 768
    SQUARE_SIZE = 25
    # WINDOW_START_X, WINDOW_START_Y = 50, 50

    # NN_DISPLAY_OFFSET_X = 50
    # NN_DISPLAY_OFFSET_Y = 150
    NN_DISPLAY_LABEL_HEIGHT_BETWEEN = 8
    # NN_DISPLAY_lABEL_OFFSET_X = NN_DISPLAY_OFFSET_X - 40
    NN_DISPLAY_NEURON_WIDTH_BETWEEN = 100
    NN_DISPLAY_NEURON_HEIGHT_BETWEEN = NN_DISPLAY_LABEL_HEIGHT_BETWEEN
    # NN_DISPLAY_NEURON_OFFSET_X = NN_DISPLAY_OFFSET_X + 100
    # NN_DISPLAY_NEURON_OFFSET_Y = NN_DISPLAY_OFFSET_Y
    NN_DISPLAY_NEURON_RADIUS = 8

    X_CENTER = WIDTH // 2
    Y_CENTER = HEIGHT // 2

    TITLE_LABEL_DIMENSION = (300, 25)
    TITLE_LABEL_POSITION = (X_CENTER - TITLE_LABEL_DIMENSION[0] // 2, 25)

    BUTTON_BACK_DIMENSION = (125, 35)
    BUTTON_BACK_POSITION = (50, HEIGHT - 75)

    PRETRAINED_BUTTON_DIMENSIONS = (250, 35)
    PRETRAINED_BUTTON_POSITION = (X_CENTER - PRETRAINED_BUTTON_DIMENSIONS[0] // 2 - 200, Y_CENTER - PRETRAINED_BUTTON_DIMENSIONS[1] // 2 - 100)

    OPTIONS_BUTTON_DIMENSIONS = (250, 35)
    OPTIONS_BUTTON_POSITION = (X_CENTER - OPTIONS_BUTTON_DIMENSIONS[0] // 2 + 200, Y_CENTER - OPTIONS_BUTTON_DIMENSIONS[1] // 2 - 100)

    NN_POSITION = (500, 100)
    BOARD_POSITION = (950, 150)

    COLOR_BLACK = (0, 0, 0)
    COLOR_BACKGROUND = (47, 47, 47)
    COLOR_WHITE = (255, 255, 255)
    COLOR_LABEL = COLOR_WHITE if DARK_MODE else COLOR_BLACK
    COLOR_SNAKE_SEGMENT = (30, 144, 255)
    COLOR_SNAKE_HEAD = (128, 0, 128)
    COLOR_APPLE = (199, 55, 47)
    COLOR_SQUARE_DELIMITER = (64, 64, 64)
    COLOR_GREEN = (0, 255, 0)
    COLOR_DODGER_BLUE = (30, 144, 255)
    COLOR_NEURON = COLOR_GREEN
    COLOR_RED = (255, 0, 0)
    COLOR_NEXT_MOVE = (0, 0, 0)
    COLOR_ODD = (42, 52, 68)
    COLOR_EVEN = (34, 41, 54)
    COLOR_FONT = (220, 220, 220)

    COLOR_MAP = {
        BoardConsts.EMPTY: (COLOR_ODD, COLOR_EVEN),
        BoardConsts.SNAKE_BODY: COLOR_SNAKE_SEGMENT,
        BoardConsts.WALL: COLOR_WHITE,
        BoardConsts.APPLE: COLOR_APPLE,
        BoardConsts.SNAKE_HEAD: COLOR_SNAKE_HEAD
    }


class Direction(Enum):
    UP = [-1, 0]
    DOWN = [1, 0]
    LEFT = [0, -1]
    RIGHT = [0, 1]
    Q1 = [-1, 1]
    Q2 = [-1, -1]
    Q3 = [1, -1]
    Q4 = [1, 1]


DYNAMIC_DIRECTIONS = ["STRAIGHT", "LEFT", "RIGHT"]
MAIN_DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
ALL_DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT, Direction.Q1, Direction.Q2, Direction.Q3, Direction.Q4]


class GameSettings:
    INITIAL_SNAKE_SIZE = 3
    SNAKE_MAX_TTL = 100

    INITIAL_BOARD_SIZE = 10

    MUTATION_CHANCE = 0.05
    POPULATION_COUNT = 1000

    AVAILABLE_INPUT_DIRECTIONS = ["4", "8"]
    AVAILABLE_VISION_LINES_RETURN_TYPE = ["boolean", "distance"]

    AVAILABLE_ACTIVATION_FUNCTIONS = ["sigmoid", "tanh", "relu", "leaky_relu"]

    AVAILABLE_SELECTION_OPERATORS = ["roulette_selection", "tournament_selection", "elitist_selection"]
    AVAILABLE_CROSSOVER_OPERATORS = ["one_point_crossover", "two_point_crossover", "uniform_crossover"]
    AVAILABLE_MUTATION_OPERATORS = ["gaussian_mutation"]
    AVAILABLE_DISTANCES = ["chebyshev_distance", "manhattan_distance"]

    GENETIC_NETWORK_FOLDER = "Genetic_Networks/"
    BACKPROPAGATION_NETWORK_FOLDER = "Backpropagation_Networks/"
