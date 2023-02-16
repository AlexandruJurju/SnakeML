import sys

import pygame

from genetic_operators import *
from model import *
from settings import GeneticSettings
from train_network import *
from view_tools import Button

training_examples: List[TrainingExample] = []
evaluated: List[TrainingExample] = []


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

    def run_best_genetic(self):
        self.window.fill(ViewConsts.COLOR_BACKGROUND)

        self.model.snake.brain = read_neural_network_from_json()

        vision_lines = get_vision_lines(self.model.board)
        neural_net_prediction = self.model.get_nn_output(vision_lines)

        self.draw_board(self.model.board)
        # self.draw_vision_lines(self.model, vision_lines)
        # self.draw_neural_network(self.model, vision_lines, nn_input, neural_net_prediction)
        # self.write_ttl(self.model.snake.ttl)
        # self.write_score(self.model.snake.score)

        next_direction = self.model.get_nn_output_4directions(neural_net_prediction)
        is_alive = self.model.move_in_direction(next_direction)

        if not is_alive:
            self.model = Model(BoardConsts.BOARD_SIZE, SnakeSettings.START_SNAKE_SIZE, self.model.snake.brain)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        pygame.display.flip()
        self.fps_clock.tick(ViewConsts.MAX_FPS)

    def next_generation(self) -> None:
        self.offspring_list.clear()

        # total_fitness = sum(individual.fitness for individual in self.parent_list)
        best_individual = max(self.parent_list, key=lambda individual: individual.fitness)

        save_neural_network_to_json(self.generation, best_individual.fitness, best_individual.brain)

        print(f"GEN {self.generation + 1}   BEST FITNESS : {best_individual.fitness}")

        parents_for_mating = elitist_selection(self.parent_list, 500)
        np.random.shuffle(parents_for_mating)

        while len(self.offspring_list) < GeneticSettings.POPULATION_COUNT:
            parent1, parent2 = roulette_selection(parents_for_mating, 2)
            child1, child2 = full_crossover(parent1.brain, parent1.brain)

            full_mutation(child1)
            full_mutation(child2)

            self.offspring_list.append(child1)
            self.offspring_list.append(child2)

        self.model.snake.brain.reinit_weights_and_biases()
        self.model = Model(BoardConsts.BOARD_SIZE, SnakeSettings.START_SNAKE_SIZE, self.offspring_list[0])

        self.generation += 1
        self.parent_list.clear()

    def run_genetic(self) -> None:
        # self.window.fill(ViewConsts.COLOR_BACKGROUND)

        vision_lines = get_vision_lines(self.model.board)
        neural_net_prediction = self.model.get_nn_output(vision_lines)

        # self.draw_board(self.model.board)
        # self.draw_vision_lines(self.model, vision_lines)
        # self.draw_neural_network(self.model, vision_lines, nn_input, neural_net_prediction)
        # self.write_ttl(self.model.snake.ttl)
        # self.write_score(self.model.snake.score)

        next_direction = self.model.get_nn_output_4directions(neural_net_prediction)
        is_alive = self.model.move_in_direction(next_direction)

        if not is_alive:
            self.model.snake.calculate_fitness()
            self.parent_list.append(self.model.snake)

            if self.generation == 0:
                self.model.snake.brain.reinit_weights_and_biases()
                self.model = Model(BoardConsts.BOARD_SIZE, SnakeSettings.START_SNAKE_SIZE, self.model.snake.brain)
            else:
                self.model = Model(BoardConsts.BOARD_SIZE, SnakeSettings.START_SNAKE_SIZE, self.offspring_list[len(self.parent_list) - 1])

            if len(self.parent_list) == GeneticSettings.POPULATION_COUNT:
                self.offspring_list.clear()
                self.next_generation()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        # pygame.display.flip()
        # self.fps_clock.tick(ViewConsts.MAX_FPS)

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
        self.window.fill(ViewConsts.COLOR_BACKGROUND)

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

    def run_backpropagation(self) -> None:
        self.window.fill(ViewConsts.COLOR_BACKGROUND)
        button_back = Button((100, 50), 50, 50, "BACK", self.universal_font, ViewConsts.COLOR_WHITE, ViewConsts.COLOR_RED)
        button_back.draw(self.window)

        vision_lines = get_vision_lines(self.model.board)
        nn_output = self.model.get_nn_output(vision_lines)

        example_output = np.where(nn_output == np.max(nn_output), 1, 0)
        example = TrainingExample(copy.deepcopy(self.model.board), self.model.snake.direction, vision_lines, example_output.ravel().tolist())
        training_examples.append(example)

        self.draw_board(self.model.board)
        self.draw_vision_lines(self.model, vision_lines)
        self.draw_neural_network(self.model, vision_lines)
        self.write_ttl(self.model.snake.ttl)
        self.write_score(self.model.snake.score)

        next_direction = self.model.get_nn_output_4directions(nn_output)
        is_alive = self.model.move_in_direction(next_direction)
        if not is_alive:
            self.state = State.RUN_BACKWARD_TRAIN

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_back.check_clicked():
                    self.state = State.OPTIONS_BACKPROPAGATION

        pygame.display.flip()
        self.fps_clock.tick(ViewConsts.MAX_FPS)

    def draw_next_snake_direction(self, board: List[List[str]], prediction: Direction) -> None:
        head = find_snake_head_poz(board)
        current_x = head[1] * ViewConsts.SQUARE_SIZE + ViewConsts.OFFSET_BOARD_X
        current_y = head[0] * ViewConsts.SQUARE_SIZE + ViewConsts.OFFSET_BOARD_Y

        # draw next position of snake
        next_position = [head[0] + prediction.value[0], head[1] + prediction.value[1]]
        next_x = next_position[1] * ViewConsts.SQUARE_SIZE + ViewConsts.OFFSET_BOARD_X
        next_y = next_position[0] * ViewConsts.SQUARE_SIZE + ViewConsts.OFFSET_BOARD_Y
        pygame.draw.rect(self.window, ViewConsts.COLOR_BLACK, pygame.Rect(next_x, next_y, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE))

        # write letters for directions
        right_text = self.universal_font.render("D", True, ViewConsts.COLOR_GREEN)
        self.window.blit(right_text, (current_x + ViewConsts.SQUARE_SIZE, current_y))

        left_text = self.universal_font.render("A", True, ViewConsts.COLOR_GREEN)
        self.window.blit(left_text, (current_x - ViewConsts.SQUARE_SIZE, current_y))

        down_text = self.universal_font.render("S", True, ViewConsts.COLOR_GREEN)
        self.window.blit(down_text, (current_x, current_y + ViewConsts.SQUARE_SIZE))

        up_text = self.universal_font.render("W", True, ViewConsts.COLOR_GREEN)
        self.window.blit(up_text, (current_x, current_y - ViewConsts.SQUARE_SIZE))
