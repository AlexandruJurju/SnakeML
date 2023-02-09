import time

import pygame
from Neural.neural_network import Dense
from constants import *
import numpy as np
from typing import List
from model import Model
import os


# TODO add view for board training examples
# TODO add dropdown for options
# TODO add buttons
# TODO add highscore
# TODO add dropdown for board size
class View:
    def __init__(self):
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (ViewConsts.WINDOW_START_X, ViewConsts.WINDOW_START_Y)

        pygame.init()
        self.window = pygame.display.set_mode((ViewConsts.WIDTH, ViewConsts.HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.fps_clock = pygame.time.Clock()

    def clear_window(self) -> None:
        self.window.fill(ViewConsts.COLOR_BACKGROUND)

    def update_window(self) -> None:
        pygame.display.update()
        self.fps_clock.tick(ViewConsts.MAX_FPS)

    def draw_score(self, score: int) -> None:
        font = pygame.font.SysFont("arial", 18)

        score_text = font.render("Score: " + str(score), True, ViewConsts.COLOR_WHITE)
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

    def draw_dead(self, board: List) -> None:
        for x in range(len(board)):
            for y in range(len(board)):
                x_position = x * ViewConsts.SQUARE_SIZE + ViewConsts.OFFSET_BOARD_X
                y_position = y * ViewConsts.SQUARE_SIZE + ViewConsts.OFFSET_BOARD_Y

                match board[y][x]:
                    case BoardConsts.SNAKE_BODY:
                        pygame.draw.rect(self.window, ViewConsts.COLOR_RED, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE))
                    case BoardConsts.SNAKE_HEAD:
                        pygame.draw.rect(self.window, ViewConsts.COLOR_RED, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE))
                # draw lines between squares
                pygame.draw.rect(self.window, ViewConsts.COLOR_SQUARE_DELIMITER, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE), width=1)
        pygame.display.update()

    def draw_vision_lines(self, model: Model, vision_lines) -> None:
        font = pygame.font.SysFont("arial", 18)

        # loop over all lines in given vision lines
        for line in vision_lines:
            line_label = font.render(line, True, ViewConsts.COLOR_BLACK)

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
        font = pygame.font.SysFont("arial", 16)

        neuron_offset_x = ViewConsts.NN_DISPLAY_OFFSET_X + 100

        # TODO bug with param type changing, not a bug just something that happens when using dynamic directions
        label_count = 0
        param_type = ["WALL", "APPLE", "SEGMENT"]
        for line in vision_lines:
            for param in param_type:
                line_label = font.render(line + " " + param, True, ViewConsts.COLOR_WHITE)
                self.window.blit(line_label, [ViewConsts.NN_DISPLAY_OFFSET_X, ViewConsts.NN_DISPLAY_LABEL_HEIGHT_BETWEEN * label_count + ViewConsts.NN_DISPLAY_OFFSET_Y - 10])
                label_count += 1

        for direction in MAIN_DIRECTIONS:
            line_label = font.render(direction.name, True, ViewConsts.COLOR_WHITE)
            self.window.blit(line_label, [ViewConsts.NN_DISPLAY_OFFSET_X, ViewConsts.NN_DISPLAY_LABEL_HEIGHT_BETWEEN * label_count + ViewConsts.NN_DISPLAY_OFFSET_Y - 10])
            label_count += 1

        self.draw_neurons(model, neuron_offset_x, font, nn_input, nn_output)

    # TODO color when using distance
    # TODO find neuron positions first then draw them, more efficient
    def draw_neurons(self, model: Model, neuron_offset_x, font, nn_input, nn_output: np.ndarray) -> None:
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

                    line_label = font.render(direction, True, ViewConsts.COLOR_WHITE)
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
