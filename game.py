import sys

import pygame

from genetic_operators import *
from model import *
from settings import GeneticSettings
from train_network import *
from view_tools import Button


# TODO add options for using different neural networks, for using different directions 4,8,16
# TODO add dropdown for options
# TODO add dropdown for board size
class Game:
    def __init__(self, model: Model, state: Enum):
        self.model = model
        self.state = state

        # set start window position using variables from ViewVars
        # os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (ViewConsts.WINDOW_START_X, ViewConsts.WINDOW_START_Y)

        pygame.init()
        self.window = pygame.display.set_mode((ViewConsts.WIDTH, ViewConsts.HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.fps_clock = pygame.time.Clock()

        self.universal_font = pygame.font.SysFont("arial", 18)

        self.generation = 0
        self.parent_list: List[Snake] = []
        self.offspring_list: List[NeuralNetwork] = []

    @staticmethod
    def wait_for_key() -> str:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    match event.key:
                        case pygame.K_ESCAPE:
                            pygame.quit()
                            sys.exit()
                        case pygame.K_w:
                            return "W"
                        case pygame.K_s:
                            return "S"
                        case pygame.K_a:
                            return "A"
                        case pygame.K_d:
                            return "D"
                        case pygame.K_RETURN:
                            return ""
                        case pygame.K_x:
                            return "X"

    def train_backpropagation(self) -> None:

        current_example = training_examples[0]
        training_examples.pop(0)

        self.draw_board(current_example.board)
        self.draw_next_snake_direction(current_example.board, self.model.get_nn_output_4directions(current_example.predictions))

        # TODO BAD UPDATE
        pygame.display.update()

        print(f"Model \n {np.matrix(current_example.board)} \n")
        print(f"Current Direction : {current_example.current_direction} \n")
        print(f"Prediction UP : {current_example.predictions[0]}")
        print(f"Prediction DOWN : {current_example.predictions[1]}")
        print(f"Prediction LEFT : {current_example.predictions[2]}")
        print(f"Prediction RIGHT : {current_example.predictions[3]}")
        print()

        print("Enter target outputs for neural network in form")
        print("UP=W DOWN=S LEFT=A RIGHT=D")

        input_string = self.wait_for_key()
        skip = False

        if input_string == "X":
            skip = True
        else:
            if input_string == "":
                target_output = current_example.predictions
            else:
                target_output = [0.0, 0.0, 0.0, 0.0]
                if input_string == "W":
                    target_output[0] = 1.0
                if input_string == "S":
                    target_output[1] = 1.0
                if input_string == "A":
                    target_output[2] = 1.0
                if input_string == "D":
                    target_output[3] = 1.0

            print(target_output)
            print()
            evaluated.append(TrainingExample(copy.deepcopy(current_example.board), current_example.current_direction, current_example.vision_lines, target_output))

        if len(training_examples) == 0 or skip:
            training_examples.clear()
            write_examples_to_json_4d(evaluated)

            evaluated.clear()

            # TODO BAD REINIT
            self.model.snake.brain.reinit_weights_and_biases()
            # TODO add reinit function in model
            self.model = Model(BoardConsts.BOARD_SIZE, SnakeSettings.START_SNAKE_SIZE, self.model.snake.brain)

            train_network(self.model.snake.brain)

            self.state = State.RUN_BACKPROPAGATION

        pygame.display.flip()
        self.fps_clock.tick(ViewConsts.MAX_FPS)



