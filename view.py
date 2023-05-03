from typing import List, Tuple

import numpy as np
import pygame

import vision
from game_config import ViewSettings, MAIN_DIRECTIONS, Direction, BoardConsts
from model import Model
from neural_network import Dense
from vision import find_snake_head_poz


def draw_board(window, board: np.ndarray, offset_x, offset_y) -> None:
    # use y,x for index in board instead of x,y because of changed logic
    # x is line y is column ; drawing x is column and y is line
    board_size = len(board)
    square_rect = pygame.Rect(0, 0, ViewSettings.SQUARE_SIZE, ViewSettings.SQUARE_SIZE)

    for x in range(board_size):
        for y in range(board_size):
            x_position = x * ViewSettings.SQUARE_SIZE + offset_x
            y_position = y * ViewSettings.SQUARE_SIZE + offset_y

            color = ViewSettings.COLOR_MAP[board[y][x]]
            val = board[y][x]
            if val == BoardConsts.EMPTY:
                color = color[(x + y) % 2]
            pygame.draw.rect(window, color, square_rect.move(x_position, y_position))
            pygame.draw.rect(window, ViewSettings.COLOR_SQUARE_DELIMITER, square_rect.move(x_position, y_position), width=1)


def draw_vision_lines(window, snake_head, vision_lines: List[vision.VisionLine], offset_x, offset_y) -> None:
    # loop over all lines in given vision lines

    font = pygame.font.SysFont("arial", 17, bold=True)
    # TODO put direction name in visionline
    for line in vision_lines:
        line_label = font.render(line.direction.name[:2] if line.direction.name.startswith("Q") else line.direction.name[0], True, ViewSettings.COLOR_BLACK)

        # render vision line text at wall position
        line_center_x = line.wall_coord[1] * ViewSettings.SQUARE_SIZE + ViewSettings.SQUARE_SIZE // 2 + offset_x
        line_center_y = line.wall_coord[0] * ViewSettings.SQUARE_SIZE + ViewSettings.SQUARE_SIZE // 2 + offset_y
        text_rect = line_label.get_rect(center=(line_center_x, line_center_y))
        window.blit(line_label, text_rect)

        # draw line from head to wall, draw before body and apple lines
        # drawing uses SQUARE_SIZE//2 so that lines go through the middle of the squares
        line_end_x = snake_head[1] * ViewSettings.SQUARE_SIZE + ViewSettings.SQUARE_SIZE // 2 + offset_x
        line_end_y = snake_head[0] * ViewSettings.SQUARE_SIZE + ViewSettings.SQUARE_SIZE // 2 + offset_y

        # draw line form snake head until wall block
        draw_vision_line(window, ViewSettings.COLOR_APPLE, 1, line.wall_coord[1], line.wall_coord[0], line_end_x, line_end_y, offset_x, offset_y)

        # draw another line from snake head to first segment found
        if line.segment_coord is not None:
            draw_vision_line(window, ViewSettings.COLOR_RED, 5, line.segment_coord[1], line.segment_coord[0], line_end_x, line_end_y, offset_x, offset_y)

        # draw another line from snake to apple if apple is found
        if line.apple_coord is not None:
            draw_vision_line(window, ViewSettings.COLOR_GREEN, 5, line.apple_coord[1], line.apple_coord[0], line_end_x, line_end_y, offset_x, offset_y)


def draw_vision_line(window, color, width, line_coord_1, line_coord_0, line_end_x, line_end_y, offset_x, offset_y) -> None:
    pygame.draw.line(window, color,
                     (line_coord_1 * ViewSettings.SQUARE_SIZE + ViewSettings.SQUARE_SIZE // 2 + offset_x,
                      line_coord_0 * ViewSettings.SQUARE_SIZE + ViewSettings.SQUARE_SIZE // 2 + offset_y),
                     (line_end_x, line_end_y), width=width)


def draw_neural_network_complete(window, model: Model, vision_lines: List[vision.VisionLine], offset_x, offset_y):
    nn_layers = model.snake.brain.layers
    dense_layers = model.snake.brain.get_dense_layers()
    neuron_offset_x = 100 + offset_x
    neuron_offset_y = offset_y

    line_start_positions: List[Tuple[int, int]] = []
    line_end_positions: List[Tuple[int, int]] = []

    param_type = ["WALL", "APPLE", "SEGMENT"]
    font = pygame.font.SysFont("arial", 12)

    max_y = -1
    for layer in dense_layers:
        max_y_input = layer.input_size * (ViewSettings.NN_DISPLAY_NEURON_HEIGHT_BETWEEN + ViewSettings.NN_DISPLAY_NEURON_RADIUS * 2)
        max_y_output = layer.output_size * (ViewSettings.NN_DISPLAY_NEURON_HEIGHT_BETWEEN + ViewSettings.NN_DISPLAY_NEURON_RADIUS * 2)
        max_layer = max_y_input if max_y_input > max_y_output else max_y_output
        if max_layer > max_y:
            max_y = max_layer

    for layer_count, layer in enumerate(nn_layers):
        if type(layer) is Dense:
            input_neuron_count = layer.input_size
            output_neuron_count = layer.output_size

            # if it's the first layer only draw input
            if layer_count == 0:
                current_max_y = input_neuron_count * (ViewSettings.NN_DISPLAY_NEURON_HEIGHT_BETWEEN + ViewSettings.NN_DISPLAY_NEURON_RADIUS * 2)
                offset = (max_y - current_max_y) // 2
                neuron_offset_y += offset

                for i in range(input_neuron_count):
                    neuron_x = neuron_offset_x
                    neuron_y = neuron_offset_y
                    neuron_offset_y += ViewSettings.NN_DISPLAY_NEURON_HEIGHT_BETWEEN + ViewSettings.NN_DISPLAY_NEURON_RADIUS * 2
                    line_start_positions.append((neuron_x, neuron_y))

                    # first ones are vision lines input, the last 4 are current snake direction
                    if i < input_neuron_count - 4:
                        # divide by number of attributes in vision line, 0 0 0, 1 1 1, 2 2 2
                        line_label = font.render(vision_lines[int(i / 3)].direction.name + " " + param_type[i % (len(param_type))], True, ViewSettings.COLOR_LABEL)
                        window.blit(line_label, (neuron_x - 125, neuron_y - 10))
                    else:
                        line_label = font.render(MAIN_DIRECTIONS[i % 4].name, True, ViewSettings.COLOR_LABEL)
                        window.blit(line_label, (neuron_x - 125, neuron_y - 10))

                    inner_color = ViewSettings.COLOR_NEURON * layer.input[i]
                    inner_color[inner_color > 255] = 255
                    inner_color[inner_color < 0] = 0
                    pygame.draw.circle(window, inner_color, (neuron_x, neuron_y), ViewSettings.NN_DISPLAY_NEURON_RADIUS)

                    pygame.draw.circle(window, ViewSettings.COLOR_WHITE, (neuron_x, neuron_y), ViewSettings.NN_DISPLAY_NEURON_RADIUS, width=1)

                neuron_offset_x += ViewSettings.NN_DISPLAY_NEURON_WIDTH_BETWEEN
                neuron_offset_y = offset_y

            # always draw output neurons
            current_max_y = output_neuron_count * (ViewSettings.NN_DISPLAY_NEURON_HEIGHT_BETWEEN + ViewSettings.NN_DISPLAY_NEURON_RADIUS * 2)
            offset = (max_y - current_max_y) // 2
            neuron_offset_y += offset

            for j in range(output_neuron_count):
                neuron_x = neuron_offset_x
                neuron_y = neuron_offset_y
                neuron_offset_y += ViewSettings.NN_DISPLAY_NEURON_HEIGHT_BETWEEN + ViewSettings.NN_DISPLAY_NEURON_RADIUS * 2
                line_end_positions.append((neuron_x, neuron_y))

                if layer_count == len(nn_layers) - 2:
                    line_label = font.render(MAIN_DIRECTIONS[j].name, True, ViewSettings.COLOR_LABEL)
                    window.blit(line_label, (neuron_x + 25, neuron_y - 10))

                    max_neuron_output = np.max(nn_layers[layer_count + 1].output)

                    if nn_layers[layer_count + 1].output[j] == max_neuron_output:
                        inner_color = (0, 255, 0)
                    else:
                        inner_color = (0, 0, 0)
                else:
                    neuron_output = nn_layers[layer_count + 1].output[j]
                    if neuron_output <= 0:
                        inner_color = ViewSettings.COLOR_BLACK
                    else:
                        inner_color = ViewSettings.COLOR_NEURON * neuron_output
                        inner_color = tuple(int(min(x, 255)) for x in inner_color)

                pygame.draw.circle(window, inner_color, (neuron_x, neuron_y), ViewSettings.NN_DISPLAY_NEURON_RADIUS)
                pygame.draw.circle(window, ViewSettings.COLOR_WHITE, (neuron_x, neuron_y), ViewSettings.NN_DISPLAY_NEURON_RADIUS, width=1)

            neuron_offset_x += ViewSettings.NN_DISPLAY_NEURON_WIDTH_BETWEEN
            neuron_offset_y = offset_y

            # self.draw_lines_between_neurons(line_start_positions, line_end_positions)
            line_start_positions = line_end_positions
            line_end_positions = []


def draw_lines_between_neurons(window, line_end: List[Tuple], line_start: List[Tuple]):
    for start_pos in line_start:
        for end_pos in line_end:
            pygame.draw.line(window, ViewSettings.COLOR_WHITE, start_pos, end_pos, width=1)


def draw_colored_lines_between_neurons(window, layer: Dense, line_end: List, line_start: List):
    for i in range(len(line_end)):
        for j in range(len(line_start)):
            if layer.weights[i][j] < 0:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)

            pygame.draw.line(window, color, line_start[j], line_end[i], width=1)


def draw_next_snake_direction(window, board: np.ndarray, prediction: Direction, offset_x, offset_y) -> None:
    head = find_snake_head_poz(board)
    font_size = 15
    current_x = head[1] * ViewSettings.SQUARE_SIZE + offset_x + ViewSettings.SQUARE_SIZE // 2
    current_y = head[0] * ViewSettings.SQUARE_SIZE + offset_y + ViewSettings.SQUARE_SIZE // 2
    font = pygame.font.SysFont("arial", font_size)

    # draw next position of snake
    next_position = [head[0] + prediction.value[0], head[1] + prediction.value[1]]
    next_x = next_position[1] * ViewSettings.SQUARE_SIZE + offset_x
    next_y = next_position[0] * ViewSettings.SQUARE_SIZE + offset_y
    pygame.draw.rect(window, ViewSettings.COLOR_NEXT_MOVE, pygame.Rect(next_x, next_y, ViewSettings.SQUARE_SIZE, ViewSettings.SQUARE_SIZE))

    # write letters for directions
    right_text = font.render("D", True, ViewSettings.COLOR_FONT)
    right_text_width, right_text_height = right_text.get_size()
    right_x = current_x - right_text_width // 2
    right_y = current_y - right_text_height // 2
    window.blit(right_text, (right_x + ViewSettings.SQUARE_SIZE, right_y))

    left_text = font.render("A", True, ViewSettings.COLOR_FONT)
    left_text_width, left_text_height = left_text.get_size()
    left_x = current_x - left_text_width // 2
    left_y = current_y - left_text_height // 2
    window.blit(left_text, (left_x - ViewSettings.SQUARE_SIZE, left_y))

    down_text = font.render("S", True, ViewSettings.COLOR_FONT)
    down_text_width, down_text_height = down_text.get_size()
    down_x = current_x - down_text_width // 2
    down_y = current_y - down_text_height // 2
    window.blit(down_text, (down_x, down_y + ViewSettings.SQUARE_SIZE))

    up_text = font.render("W", True, ViewSettings.COLOR_FONT)
    up_text_width, up_text_height = up_text.get_size()
    up_x = current_x - up_text_width // 2
    up_y = current_y - up_text_height // 2
    window.blit(up_text, (up_x, up_y - ViewSettings.SQUARE_SIZE))


def write_controls(window, pos_x, pos_y) -> None:
    font = pygame.font.SysFont("Arial", 20)
    text_lines = ["CONTROLS", "ESCAPE to return to menu", "WASD to move the snake", "ENTER to keep the same move", "X to end training and skip the rest of the example"]

    for i, line in enumerate(text_lines):
        text_surface = font.render(line, True, ViewSettings.COLOR_FONT)
        text_rect = text_surface.get_rect()
        text_rect.center = (pos_x, pos_y + i * (font.get_linesize() + 8))
        window.blit(text_surface, text_rect)
