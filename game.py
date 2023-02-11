import copy
import csv
import os
import sys

import pygame

from Neural.neural_network import mse, mse_prime, Dense
from model import *
from settings import GeneticSettings
from view_tools import Button


class TrainingExample:
    def __init__(self, board: List[str], predictions: List[float], current_direction: Direction):
        self.board = board
        self.predictions = predictions
        self.current_direction = current_direction


def read_training_models() -> Tuple:
    file = open(NNSettings.TRAIN_DATA_FILE_LOCATION)
    csvreader = csv.reader(file)

    data = []
    for row in csvreader:
        data.append(row)

    x = []
    y = []

    if len(data) != 0:
        for row in data:
            board = eval(row[0])

            # direction is saved as Direction.UP, but direction.name is just UP, use split to get second part
            direction_string = row[1].split(".")[1]
            real_direction = None
            for direction in MAIN_DIRECTIONS:
                direction_enum_name = direction.name
                if direction_string == direction_enum_name:
                    real_direction = direction
                    break

            vision_lines = get_vision_lines(board)

            x.append(get_parameters_in_nn_input_form(vision_lines, real_direction))

            # dynamic loop over columns in csv, skips board and current direction
            outputs = []
            for i in range(2, len(row)):
                outputs.append(float(row[i]))
            y.append(outputs)

    return x, y


def train_network(network: NeuralNetwork) -> None:
    x, y = read_training_models()

    # example for points
    # x is (10000,2) 10000 lines, 2 columns ; 10000 examples each with x coord and y coord
    # when using a single example x_test from x, x_test is (2,)
    # resizing can be done for the whole training data resize(10000,2,1)
    # or for just one example resize(2,1)
    x = np.reshape(x, (len(x), NNSettings.INPUT_NEURON_COUNT, 1))
    y = np.reshape(y, (len(y), NNSettings.OUTPUT_NEURON_COUNT, 1))

    network.train(mse, mse_prime, x, y, 0.5)

    # for x_test, y_test in zip(x, y):
    #     output = network.feed_forward(x_test)
    #     output_index = list(output).index(max(list(output)))
    #     target_index = list(y_test).index(max(list(y_test)))
    #     print(f"target = {target_index}, output = {output_index}")
    #     print("============================================")


def write_examples_to_csv_4d(examples: List[TrainingExample]) -> None:
    file = open(NNSettings.TRAIN_DATA_FILE_LOCATION, "w+", newline='')
    writer = csv.writer(file)

    examples_to_write = []
    for example in examples:
        up = example.predictions[0]
        down = example.predictions[1]
        left = example.predictions[2]
        right = example.predictions[3]

        examples_to_write.append([example.board, example.current_direction, up, down, left, right])

    writer.writerows(examples_to_write)
    file.close()


# def evaluate_live_examples_4d(examples: List[TrainingExample]) -> None:
#     evaluated = []
#
#     for example in examples:
#         print(f"Model \n {np.matrix(example.board)} \n")
#         print(f"Current Direction : {example.current_direction} \n")
#         print(f"Prediction UP : {example.predictions[0]}")
#         print(f"Prediction DOWN : {example.predictions[1]}")
#         print(f"Prediction LEFT : {example.predictions[2]}")
#         print(f"Prediction RIGHT : {example.predictions[3]}")
#         print()
#
#         # if ViewVars.DRAW:
#         #     self.view.clear_window()
#         #     self.view.draw_board(example.model)
#         #     self.view.update_window()
#
#         print("Enter target outputs for neural network in form")
#         print("UP=W DOWN=S LEFT=A RIGHT=D")
#         target_string = input("")
#
#         if target_string == "":
#             target_output = example.predictions
#         elif target_string == "x":
#             break
#         else:
#             target_output = [0.0, 0.0, 0.0, 0.0]
#             if target_string.__contains__("w"):
#                 target_output[0] = 1.0
#             if target_string.__contains__("s"):
#                 target_output[1] = 1.0
#             if target_string.__contains__("a"):
#                 target_output[2] = 1.0
#             if target_string.__contains__("d"):
#                 target_output[3] = 1.0
#
#         print(target_output)
#         print()
#         evaluated.append(TrainingExample(copy.deepcopy(example.board), target_output, example.current_direction))
#
#     write_examples_to_csv_4d(evaluated)


# TODO add options for using different neural networks, for using different directions 4,8,16
training_examples = []
evaluated = []


# TODO add dropdown for options
# TODO add dropdown for board size


class Game:
    def __init__(self, model: Model, state: Enum):
        self.model = model
        self.state = state

        # set start window position using variables from ViewVars
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (ViewConsts.WINDOW_START_X, ViewConsts.WINDOW_START_Y)
        pygame.init()
        self.window = pygame.display.set_mode((ViewConsts.WIDTH, ViewConsts.HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.fps_clock = pygame.time.Clock()

        self.universal_font = pygame.font.SysFont("arial", 18)

        self.generation = 0
        self.parent_list = []
        self.offspring_list = []

    def state_machine(self):
        while True:
            match self.state:
                case State.MAIN_MENU:
                    self.main_menu()
                case State.RUN_BACKPROPAGATION:
                    self.run_backpropagation()
                case State.BACKWARD_TRAIN:
                    self.train_backpropagation()
                case State.OPTIONS:
                    self.options()
                case State.RUN_GENETIC:
                    self.run_genetic()

            pygame.display.flip()
            self.fps_clock.tick(ViewConsts.MAX_FPS)

    def next_generation(self):
        pass

    def run_genetic(self):
        self.window.fill(ViewConsts.COLOR_BACKGROUND)

        # pygame.display.set_caption("GENETIC")
        # window_title = self.universal_font.render("GENETIC", True, ViewConsts.COLOR_WHITE)
        # self.window.blit(window_title, [ViewConsts.WINDOW_TITLE_X, ViewConsts.WINDOW_TITLE_Y])

        vision_lines = get_vision_lines(self.model.board)
        neural_net_prediction = self.model.get_nn_output(vision_lines)
        nn_input = get_parameters_in_nn_input_form(vision_lines, self.model.snake.direction)

        self.draw_board(self.model.board)
        self.draw_vision_lines(self.model, vision_lines)
        self.draw_neural_network(self.model, vision_lines, nn_input, neural_net_prediction)
        self.write_ttl(self.model.snake.ttl)
        self.write_score(self.model.snake.score)

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

    def main_menu(self):
        pygame.display.set_caption("Main Menu")

        self.window.fill(ViewConsts.COLOR_BACKGROUND)

        button_run = Button((50, 50), 50, 50, "RUN", self.universal_font, ViewConsts.COLOR_WHITE, ViewConsts.COLOR_BLACK)
        button_run.draw(self.window)

        button_options = Button((50, 125), 50, 50, "OPTIONS", self.universal_font, ViewConsts.COLOR_WHITE, ViewConsts.COLOR_BLACK)
        button_options.draw(self.window)

        button_quit = Button((50, 200), 50, 50, "QUIT", self.universal_font, ViewConsts.COLOR_WHITE, ViewConsts.COLOR_BLACK)
        button_quit.draw(self.window)

        button_genetic = Button((50, 300), 50, 50, "GENETIC", self.universal_font, ViewConsts.COLOR_WHITE, ViewConsts.COLOR_BLACK)
        button_genetic.draw(self.window)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_run.check_clicked():
                    self.state = State.RUN_BACKPROPAGATION
                if button_options.check_clicked():
                    self.state = State.OPTIONS
                if button_genetic.check_clicked():
                    self.state = State.RUN_GENETIC
                if button_quit.check_clicked():
                    pygame.quit()
                    sys.exit()

    def options(self):
        pygame.display.set_caption("Options")

        self.window.fill(ViewConsts.COLOR_BACKGROUND)

        button_back = Button((50, 50), 50, 50, "BACK", self.universal_font, ViewConsts.COLOR_WHITE, ViewConsts.COLOR_BLACK)
        button_back.draw(self.window)

        window_title = self.universal_font.render("OPTIONS", True, ViewConsts.COLOR_WHITE)
        self.window.blit(window_title, [ViewConsts.WINDOW_TITLE_X, ViewConsts.WINDOW_TITLE_Y])

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
                    self.state = State.MAIN_MENU

    def wait_for_key(self):
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

    def draw_board_with_directions(self, board: List) -> None:
        # use y,x for index in board instead of x,y because of changed logic
        # x is line y is column ; drawing x is column and y is line
        for x in range(len(board)):
            for y in range(len(board)):
                x_position = x * ViewConsts.SQUARE_SIZE + ViewConsts.OFFSET_BOARD_X
                y_position = y * ViewConsts.SQUARE_SIZE + ViewConsts.OFFSET_BOARD_Y

                match board[y][x]:
                    case BoardConsts.SNAKE_BODY:
                        pygame.draw.rect(self.window, ViewConsts.COLOR_SNAKE, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE))
                    case BoardConsts.WALL:
                        pygame.draw.rect(self.window, ViewConsts.COLOR_WHITE, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE))
                    case BoardConsts.APPLE:
                        pygame.draw.rect(self.window, ViewConsts.COLOR_APPLE, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE))
                    case BoardConsts.SNAKE_HEAD:
                        pygame.draw.rect(self.window, ViewConsts.COLOR_SNAKE_HEAD, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE))

                        # pygame.draw.rect(self.window, ViewConsts.COLOR_APPLE,
                        #                  pygame.Rect(x_position + ViewConsts.SQUARE_SIZE, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE))
                        right_text = self.universal_font.render("D", True, ViewConsts.COLOR_WHITE)
                        self.window.blit(right_text, (x_position + ViewConsts.SQUARE_SIZE, y_position))

                        left_text = self.universal_font.render("A", True, ViewConsts.COLOR_GREEN)
                        self.window.blit(left_text, (x_position - ViewConsts.SQUARE_SIZE, y_position))

                        up_text = self.universal_font.render("W", True, ViewConsts.COLOR_GREEN)
                        self.window.blit(up_text, (x_position, y_position - ViewConsts.SQUARE_SIZE))

                        down_text = self.universal_font.render("S", True, ViewConsts.COLOR_GREEN)
                        self.window.blit(down_text, (x_position, y_position + ViewConsts.SQUARE_SIZE))

                # draw lines between squares
                pygame.draw.rect(self.window, ViewConsts.COLOR_SQUARE_DELIMITER, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE), width=1)

    def train_backpropagation(self):
        self.window.fill(ViewConsts.COLOR_BACKGROUND)

        window_title = self.universal_font.render("TRAIN BACKPROPAGATION", True, ViewConsts.COLOR_WHITE)
        self.window.blit(window_title, [ViewConsts.WINDOW_TITLE_X, ViewConsts.WINDOW_TITLE_Y])

        current_example = training_examples[0]
        training_examples.pop(0)

        self.draw_board_with_directions(current_example.board)

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
            evaluated.append(TrainingExample(copy.deepcopy(current_example.board), target_output, current_example.current_direction))

        if len(training_examples) == 0 or skip:
            training_examples.clear()
            write_examples_to_csv_4d(evaluated)
            evaluated.clear()

            # TODO BAD REINIT
            self.model.snake.brain.reinit_weights_and_biases()
            # TODO add reinit function in model
            self.model = Model(BoardConsts.BOARD_SIZE, SnakeSettings.START_SNAKE_SIZE, self.model.snake.brain)

            train_network(self.model.snake.brain)

            self.state = State.RUN_BACKPROPAGATION

    def run_backpropagation(self):
        self.window.fill(ViewConsts.COLOR_BACKGROUND)
        button_back = Button((100, 50), 50, 50, "BACK", self.universal_font, ViewConsts.COLOR_WHITE, ViewConsts.COLOR_RED)
        button_back.draw(self.window)
        window_title = self.universal_font.render("MAIN RUN", True, ViewConsts.COLOR_WHITE)
        self.window.blit(window_title, [ViewConsts.WINDOW_TITLE_X, ViewConsts.WINDOW_TITLE_Y])

        vision_lines = get_vision_lines(self.model.board)
        neural_net_prediction = self.model.get_nn_output(vision_lines)
        nn_input = get_parameters_in_nn_input_form(vision_lines, self.model.snake.direction)

        example_prediction = np.where(neural_net_prediction == np.max(neural_net_prediction), 1, 0)
        example = TrainingExample(copy.deepcopy(self.model.board), example_prediction.ravel().tolist(), self.model.snake.direction)
        training_examples.append(example)

        self.draw_board(self.model.board)
        self.draw_vision_lines(self.model, vision_lines)
        self.draw_neural_network(self.model, vision_lines, nn_input, neural_net_prediction)
        self.write_ttl(self.model.snake.ttl)
        self.write_score(self.model.snake.score)

        next_direction = self.model.get_nn_output_4directions(neural_net_prediction)
        is_alive = self.model.move_in_direction(next_direction)
        if not is_alive:
            self.state = State.BACKWARD_TRAIN

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
                    self.state = State.MAIN_MENU

    def write_ttl(self, ttl: int):
        score_text = self.universal_font.render("Moves Left: " + str(ttl), True, ViewConsts.COLOR_WHITE)
        self.window.blit(score_text, [ViewConsts.OFFSET_BOARD_X + 25, ViewConsts.OFFSET_BOARD_Y - 75])

    def write_score(self, score: int) -> None:
        score_text = self.universal_font.render("Score: " + str(score), True, ViewConsts.COLOR_WHITE)
        self.window.blit(score_text, [ViewConsts.OFFSET_BOARD_X + 25, ViewConsts.OFFSET_BOARD_Y - 50])

    def draw_board(self, board: List) -> None:
        # use y,x for index in board instead of x,y because of changed logic
        # x is line y is column ; drawing x is column and y is line
        for x in range(len(board)):
            for y in range(len(board)):
                x_position = x * ViewConsts.SQUARE_SIZE + ViewConsts.OFFSET_BOARD_X
                y_position = y * ViewConsts.SQUARE_SIZE + ViewConsts.OFFSET_BOARD_Y

                match board[y][x]:
                    case BoardConsts.SNAKE_BODY:
                        pygame.draw.rect(self.window, ViewConsts.COLOR_SNAKE, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE))
                    case BoardConsts.WALL:
                        pygame.draw.rect(self.window, ViewConsts.COLOR_WHITE, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE))
                    case BoardConsts.APPLE:
                        pygame.draw.rect(self.window, ViewConsts.COLOR_APPLE, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE))
                    case BoardConsts.SNAKE_HEAD:
                        pygame.draw.rect(self.window, ViewConsts.COLOR_SNAKE_HEAD, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE))
                # draw lines between squares
                pygame.draw.rect(self.window, ViewConsts.COLOR_SQUARE_DELIMITER, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE), width=1)

    # def draw_dead(self, board: List) -> None:
    #     for x in range(len(board)):
    #         for y in range(len(board)):
    #             x_position = x * ViewConsts.SQUARE_SIZE + ViewConsts.OFFSET_BOARD_X
    #             y_position = y * ViewConsts.SQUARE_SIZE + ViewConsts.OFFSET_BOARD_Y
    #
    #             match board[y][x]:
    #                 case BoardConsts.SNAKE_BODY:
    #                     pygame.draw.rect(self.window, ViewConsts.COLOR_RED, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE))
    #                 case BoardConsts.SNAKE_HEAD:
    #                     pygame.draw.rect(self.window, ViewConsts.COLOR_RED, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE))
    #             # draw lines between squares
    #             pygame.draw.rect(self.window, ViewConsts.COLOR_SQUARE_DELIMITER, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE), width=1)

    def draw_vision_lines(self, model: Model, vision_lines) -> None:

        # loop over all lines in given vision lines
        for line in vision_lines:
            line_label = self.universal_font.render(line, True, ViewConsts.COLOR_BLACK)

            # render vision line text at wall position
            self.window.blit(line_label, [vision_lines[line].wall_coord[1] * ViewConsts.SQUARE_SIZE + ViewConsts.OFFSET_BOARD_X,
                                          vision_lines[line].wall_coord[0] * ViewConsts.SQUARE_SIZE + ViewConsts.OFFSET_BOARD_Y])

            # draw line from head to wall, draw before body and apple lines
            # drawing uses SQUARE_SIZE//2 so that lines go through the middle of the squares
            line_end_x = model.snake.body[0][1] * ViewConsts.SQUARE_SIZE + ViewConsts.SQUARE_SIZE // 2 + ViewConsts.OFFSET_BOARD_X
            line_end_y = model.snake.body[0][0] * ViewConsts.SQUARE_SIZE + ViewConsts.SQUARE_SIZE // 2 + ViewConsts.OFFSET_BOARD_Y

            # draw line form snake head until wall block
            self.draw_vision_line(ViewConsts.COLOR_APPLE, 1, vision_lines[line].wall_coord[1], vision_lines[line].wall_coord[0], line_end_x, line_end_y)

            # draw another line from snake head to first segment found
            if vision_lines[line].segment_coord is not None:
                self.draw_vision_line(ViewConsts.COLOR_RED, 5, vision_lines[line].segment_coord[1], vision_lines[line].segment_coord[0], line_end_x, line_end_y)

            # draw another line from snake to apple if apple is found
            if vision_lines[line].apple_coord is not None:
                self.draw_vision_line(ViewConsts.COLOR_GREEN, 5, vision_lines[line].apple_coord[1], vision_lines[line].apple_coord[0], line_end_x, line_end_y)

    def draw_vision_line(self, color, width, line_coord_1, line_coord_0, line_end_x, line_end_y) -> None:
        pygame.draw.line(self.window, color,
                         (line_coord_1 * ViewConsts.SQUARE_SIZE + ViewConsts.SQUARE_SIZE // 2 + ViewConsts.OFFSET_BOARD_X,
                          line_coord_0 * ViewConsts.SQUARE_SIZE + ViewConsts.SQUARE_SIZE // 2 + ViewConsts.OFFSET_BOARD_Y),
                         (line_end_x, line_end_y), width=width)

    # TODO draw lines between neurons
    # TODO write direction in inputs
    def draw_neural_network(self, model, vision_lines, nn_input, nn_output) -> None:
        neuron_offset_x = ViewConsts.NN_DISPLAY_OFFSET_X + 100

        label_count = 0
        param_type = ["WALL", "APPLE", "SEGMENT"]
        for line in vision_lines:
            for param in param_type:
                line_label = self.universal_font.render(line + " " + param, True, ViewConsts.COLOR_WHITE)
                self.window.blit(line_label, [ViewConsts.NN_DISPLAY_OFFSET_X, ViewConsts.NN_DISPLAY_LABEL_HEIGHT_BETWEEN * label_count + ViewConsts.NN_DISPLAY_OFFSET_Y - 10])
                label_count += 1

        for direction in MAIN_DIRECTIONS:
            line_label = self.universal_font.render(direction.name, True, ViewConsts.COLOR_WHITE)
            self.window.blit(line_label, [ViewConsts.NN_DISPLAY_OFFSET_X, ViewConsts.NN_DISPLAY_LABEL_HEIGHT_BETWEEN * label_count + ViewConsts.NN_DISPLAY_OFFSET_Y - 10])
            label_count += 1

        self.draw_neurons(model, neuron_offset_x, nn_input, nn_output)

    # TODO color when using distance
    # TODO find neuron positions first then draw them, more efficient
    def draw_neurons(self, model: Model, neuron_offset_x, nn_input, nn_output: np.ndarray) -> None:
        dense_layers = model.snake.brain.get_dense_layers()

        # max distance is used to center the neurons in the next layers, formula for new yOffset is (yLengthPrevious - yLengthCurrent) / 2
        max_y_distance = 0

        # line start and line end are lists that contain the positions of the neurons
        # the lists are used for drawing the lines between neurons
        line_start = []
        line_end = []

        # draw neurons
        for i, layer in enumerate(dense_layers):
            # if it's the first layer, draw neurons using input
            if i == 0:
                for j in range(layer.input_size):
                    # draw the neuron
                    pygame.draw.circle(self.window, ViewConsts.COLOR_WHITE,
                                       (neuron_offset_x, ViewConsts.NN_DISPLAY_NEURON_HEIGHT_BETWEEN * j + ViewConsts.NN_DISPLAY_NEURON_OFFSET_Y),
                                       ViewConsts.NN_DISPLAY_NEURON_RADIUS, width=1)

                    # calculate neuron green color intensity using input parameters
                    activation_color = (0, round(255 * nn_input[j][0]), 0)

                    # draw green circle inside neuron with activation color
                    pygame.draw.circle(self.window, activation_color,
                                       (neuron_offset_x, ViewConsts.NN_DISPLAY_NEURON_HEIGHT_BETWEEN * j + ViewConsts.NN_DISPLAY_NEURON_OFFSET_Y),
                                       ViewConsts.NN_DISPLAY_NEURON_RADIUS - 1)

                    # append neuron position to start list
                    line_start.append([neuron_offset_x, ViewConsts.NN_DISPLAY_NEURON_HEIGHT_BETWEEN * j + ViewConsts.NN_DISPLAY_NEURON_OFFSET_Y])

                # increment current offset to obtain OX offset for next layer
                neuron_offset_x += ViewConsts.NN_DISPLAY_NEURON_WIDTH_BETWEEN

                # calculate maxYDistance for centering neurons of next layer
                max_y_distance = ViewConsts.NN_DISPLAY_NEURON_HEIGHT_BETWEEN * layer.input_size

            # calculate current y distance for centering neurons
            current_y_distance = layer.output_size * ViewConsts.NN_DISPLAY_NEURON_HEIGHT_BETWEEN

            # calculate offset using distance of prev layer and distance of current layer
            hidden_offset_y = (max_y_distance - current_y_distance) // 2

            for j in range(layer.output_size):
                # if it's the output layer
                if i == len(dense_layers) - 1:
                    nn_output[np.where(nn_output != np.max(nn_output))] = 0
                    nn_output[np.where(nn_output == np.max(nn_output))] = 1

                    # draw color inside the neuron
                    pygame.draw.circle(self.window, ViewConsts.COLOR_GREEN * nn_output[j],
                                       (neuron_offset_x, ViewConsts.NN_DISPLAY_NEURON_HEIGHT_BETWEEN * j + ViewConsts.NN_DISPLAY_NEURON_OFFSET_Y + hidden_offset_y),
                                       ViewConsts.NN_DISPLAY_NEURON_RADIUS - 1)
                    # draw white neuron outline
                    pygame.draw.circle(self.window, ViewConsts.COLOR_WHITE,
                                       (neuron_offset_x, ViewConsts.NN_DISPLAY_NEURON_HEIGHT_BETWEEN * j + ViewConsts.NN_DISPLAY_NEURON_OFFSET_Y + hidden_offset_y),
                                       ViewConsts.NN_DISPLAY_NEURON_RADIUS - 1, width=1)

                    # write direction name in output
                    match j:
                        case 0:
                            direction = "UP"
                        case 1:
                            direction = "DOWN"
                        case 2:
                            direction = "LEFT"
                        case 3:
                            direction = "RIGHT"
                        case _:
                            direction = None

                    line_label = self.universal_font.render(direction, True, ViewConsts.COLOR_WHITE)
                    self.window.blit(line_label,
                                     [neuron_offset_x + 15, ViewConsts.NN_DISPLAY_NEURON_HEIGHT_BETWEEN * j + ViewConsts.NN_DISPLAY_NEURON_OFFSET_Y + hidden_offset_y - 5])
                # Draw NN hidden layers outputs
                else:
                    # hidden neuron activation color
                    if model.snake.brain.layers[i + 1].output[j] <= 0:
                        inside_color = ViewConsts.COLOR_BLACK
                    else:
                        inside_color = ViewConsts.COLOR_GREEN

                    # draw color inside the neuron
                    pygame.draw.circle(self.window, inside_color,
                                       (neuron_offset_x, ViewConsts.NN_DISPLAY_NEURON_HEIGHT_BETWEEN * j + ViewConsts.NN_DISPLAY_NEURON_OFFSET_Y + hidden_offset_y),
                                       ViewConsts.NN_DISPLAY_NEURON_RADIUS - 1)

                    # draw neuron outline
                    pygame.draw.circle(self.window, ViewConsts.COLOR_WHITE,
                                       (neuron_offset_x, ViewConsts.NN_DISPLAY_NEURON_HEIGHT_BETWEEN * j + ViewConsts.NN_DISPLAY_NEURON_OFFSET_Y + hidden_offset_y),
                                       ViewConsts.NN_DISPLAY_NEURON_RADIUS - 1, width=1)

                # line end for drawing lines
                line_end.append([neuron_offset_x, ViewConsts.NN_DISPLAY_NEURON_HEIGHT_BETWEEN * j + ViewConsts.NN_DISPLAY_NEURON_OFFSET_Y + hidden_offset_y])
            neuron_offset_x += ViewConsts.NN_DISPLAY_NEURON_WIDTH_BETWEEN

            # self.draw_colored_lines_between_neurons(layer, line_end, line_start)
            # self.draw_lines_between_neurons(line_end, line_start)

            line_start = line_end
            line_end = []

    def draw_lines_between_neurons(self, line_end: List, line_start: List):
        for i in range(len(line_end)):
            for j in range(len(line_start)):
                pygame.draw.line(self.window, ViewConsts.COLOR_WHITE, line_start[j], line_end[i], width=1)

    def draw_colored_lines_between_neurons(self, layer: Dense, line_end: List, line_start: List):
        for i in range(len(line_end)):
            for j in range(len(line_start)):
                if layer.weights[i][j] < 0:
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 0)

                pygame.draw.line(self.window, color, line_start[j], line_end[i], width=1)
