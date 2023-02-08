import pygame
from constants import *
import numpy as np


# TODO add view for board training examples
# TODO add dropdown for options
# TODO add buttons
# TODO add highscore
# TODO add dropdown for board size
class View:
    def __init__(self):
        pygame.init()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.fps_clock = pygame.time.Clock()

    def clear_window(self):
        self.window.fill(COLOR_BACKGROUND)

    def update_window(self):
        pygame.display.update()
        self.fps_clock.tick(MAX_FPS)

    def draw_board(self, model):
        # use y,x for index in board instead of x,y because of changed logic
        # x is line y is column ; drawing x is column and y is line
        for x in range(model.size):
            for y in range(model.size):
                x_position = x * SQUARE_SIZE + OFFSET_BOARD_X
                y_position = y * SQUARE_SIZE + OFFSET_BOARD_Y

                match model.board[y, x]:
                    case "S":
                        pygame.draw.rect(self.window, COLOR_SNAKE, pygame.Rect(x_position, y_position, SQUARE_SIZE, SQUARE_SIZE))
                    case "W":
                        pygame.draw.rect(self.window, COLOR_WHITE, pygame.Rect(x_position, y_position, SQUARE_SIZE, SQUARE_SIZE))
                    case "A":
                        pygame.draw.rect(self.window, COLOR_APPLE, pygame.Rect(x_position, y_position, SQUARE_SIZE, SQUARE_SIZE))
                    case "H":
                        pygame.draw.rect(self.window, COLOR_SNAKE_HEAD, pygame.Rect(x_position, y_position, SQUARE_SIZE, SQUARE_SIZE))
                # draw lines between squares
                pygame.draw.rect(self.window, COLOR_SQUARE_DELIMITER, pygame.Rect(x_position, y_position, SQUARE_SIZE, SQUARE_SIZE), width=1)

    def draw_dead(self, model):
        for x in range(model.size):
            for y in range(model.size):
                x_position = x * SQUARE_SIZE + OFFSET_BOARD_X
                y_position = y * SQUARE_SIZE + OFFSET_BOARD_Y

                match model.board[y, x]:
                    case "S":
                        pygame.draw.rect(self.window, COLOR_RED, pygame.Rect(x_position, y_position, SQUARE_SIZE, SQUARE_SIZE))
                    case "H":
                        pygame.draw.rect(self.window, COLOR_RED, pygame.Rect(x_position, y_position, SQUARE_SIZE, SQUARE_SIZE))
                # draw lines between squares
                pygame.draw.rect(self.window, COLOR_SQUARE_DELIMITER, pygame.Rect(x_position, y_position, SQUARE_SIZE, SQUARE_SIZE), width=1)
        pygame.display.update()

    def draw_vision_lines(self, model, vision_lines):
        font = pygame.font.SysFont("arial", 18)

        # loop over all lines in given vision lines
        for line in vision_lines:
            line_label = font.render(line, True, COLOR_BLACK)

            # render vision line text at wall position
            self.window.blit(line_label, [vision_lines[line].wall_coord[1] * SQUARE_SIZE + OFFSET_BOARD_X, vision_lines[line].wall_coord[0] * SQUARE_SIZE + OFFSET_BOARD_Y])

            # draw line from head to wall, draw before body and apple lines
            # drawing uses SQUARE_SIZE//2 so that lines go through the middle of the squares
            line_end_x = model.snake.body[0][1] * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_X
            line_end_y = model.snake.body[0][0] * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_Y

            # draw line form snake head until wall block
            self.draw_vision_line(COLOR_APPLE, 1, vision_lines[line].wall_coord[1], vision_lines[line].wall_coord[0], line_end_x, line_end_y)

            # draw another line from snake head to first segment found
            if vision_lines[line].segment_coord is not None:
                self.draw_vision_line(COLOR_RED, 5, vision_lines[line].segment_coord[1], vision_lines[line].segment_coord[0], line_end_x, line_end_y)

            # draw another line from snake to apple if apple is found
            if vision_lines[line].apple_coord is not None:
                self.draw_vision_line(COLOR_GREEN, 5, vision_lines[line].apple_coord[1], vision_lines[line].apple_coord[0], line_end_x, line_end_y)

    def draw_vision_line(self, color, width, line_coord_1, line_coord_0, line_end_x, line_end_y):
        pygame.draw.line(self.window, color,
                         (line_coord_1 * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_X,
                          line_coord_0 * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_Y),
                         (line_end_x, line_end_y), width=width)

    # TODO draw lines between neurons
    def draw_neural_network(self, model, vision_lines, nn_input, nn_output):
        font = pygame.font.SysFont("arial", 16)

        input_label_offset_x = 550
        input_label_offset_y = 100
        label_height_between = 27.5

        neuron_width_between = 75
        neuron_height_between = label_height_between
        neuron_offset_x = input_label_offset_x + 100
        neuron_offset_y = input_label_offset_y
        neuron_radius = 12

        # TODO bug with param type changing
        label_count = 0
        param_type = ["WALL", "APPLE", "SEGMENT"]
        for line in vision_lines:
            for param in param_type:
                line_label = font.render(line + " " + param, True, COLOR_WHITE)
                self.window.blit(line_label, [input_label_offset_x, label_height_between * label_count + input_label_offset_y - 10])
                label_count += 1

        self.draw_neurons(model, neuron_height_between, neuron_offset_x, neuron_offset_y, neuron_radius, neuron_width_between, font, nn_input, nn_output)

    def draw_neurons(self, model, neuron_height_between, neuron_offset_x, neuron_offset_y, neuron_radius, neuron_width_between, font, nn_input, nn_output):
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
                    pygame.draw.circle(self.window, COLOR_WHITE, (neuron_offset_x, neuron_height_between * j + neuron_offset_y), neuron_radius, width=1)

                    # calculate neuron green color intensity using input parameters
                    activation_color = (0, round(255 * nn_input[j][0]), 0)

                    # draw green circle inside neuron with activation color
                    pygame.draw.circle(self.window, activation_color, (neuron_offset_x, neuron_height_between * j + neuron_offset_y), neuron_radius - 1)

                    # append neuron position to start list
                    line_start.append([neuron_offset_x, neuron_height_between * j + neuron_offset_y])

                # increment current offset to obtain OX offset for next layer
                neuron_offset_x += neuron_width_between

                # calculate maxYDistance for centering neurons of next layer
                max_y_distance = neuron_height_between * layer.input_size

            # calculate current y distance for centering neurons
            current_y_distance = layer.output_size * neuron_height_between

            # calculate offset using distance of prev layer and distance of current layer
            hidden_offset_y = (max_y_distance - current_y_distance) // 2

            for j in range(layer.output_size):
                # if it's the output layer
                if i == len(dense_layers) - 1:
                    nn_output[np.where(nn_output != np.max(nn_output))] = 0
                    nn_output[np.where(nn_output == np.max(nn_output))] = 1

                    pygame.draw.circle(self.window, (0, 255 * nn_output[j], 0), (neuron_offset_x, neuron_height_between * j + neuron_offset_y + hidden_offset_y),
                                       neuron_radius - 1)
                    pygame.draw.circle(self.window, COLOR_WHITE, (neuron_offset_x, neuron_height_between * j + neuron_offset_y + hidden_offset_y), neuron_radius - 1, width=1)

                    # write direction name in output
                    match j:
                        case 0:
                            direction = "STRAIGHT"
                        case 1:
                            direction = "LEFT"
                        case 2:
                            direction = "RIGHT"
                        case _:
                            direction = None

                    line_label = font.render(direction, True, (255, 255, 255))
                    self.window.blit(line_label, [neuron_offset_x + 15, neuron_height_between * j + neuron_offset_y + hidden_offset_y - 5])
                # Draw NN hidden layers outputs
                else:
                    # hidden neuron activation color
                    if model.snake.brain.layers[i + 1].output[j] <= 0:
                        color = (0, 0, 0)
                    else:
                        color = (0, 255, 0)
                    pygame.draw.circle(self.window, color, (neuron_offset_x, neuron_height_between * j + neuron_offset_y + hidden_offset_y), neuron_radius - 1)
                    pygame.draw.circle(self.window, COLOR_WHITE, (neuron_offset_x, neuron_height_between * j + neuron_offset_y + hidden_offset_y), neuron_radius - 1, width=1)

                # line end for drawing lines
                line_end.append([neuron_offset_x, neuron_height_between * j + neuron_offset_y + hidden_offset_y])
            neuron_offset_x += neuron_width_between

            # self.__draw_colored_lines_between_neurons(layer, line_end, line_start)
            # self.__draw_lines_between_neurons(line_end, line_start)

            line_start = line_end
            line_end = []
