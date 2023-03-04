from typing import List, Tuple

import pygame

from constants import ViewConsts, BoardConsts, MAIN_DIRECTIONS, Direction
from model import Model
from neural_network import Dense
from vision import VisionLine, find_snake_head_poz


def draw_board(window, board: List, offset_x, offset_y) -> None:
    # use y,x for index in board instead of x,y because of changed logic
    # x is line y is column ; drawing x is column and y is line
    for x in range(len(board)):
        for y in range(len(board)):
            x_position = x * ViewConsts.SQUARE_SIZE + offset_x
            y_position = y * ViewConsts.SQUARE_SIZE + offset_y

            match board[y][x]:
                case BoardConsts.SNAKE_BODY:
                    pygame.draw.rect(window, ViewConsts.COLOR_SNAKE_SEGMENT, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE))
                case BoardConsts.WALL:
                    pygame.draw.rect(window, ViewConsts.COLOR_WHITE, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE))
                case BoardConsts.APPLE:
                    pygame.draw.rect(window, ViewConsts.COLOR_APPLE, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE))
                case BoardConsts.SNAKE_HEAD:
                    pygame.draw.rect(window, ViewConsts.COLOR_SNAKE_HEAD, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE))
            # draw lines between squares
            pygame.draw.rect(window, ViewConsts.COLOR_SQUARE_DELIMITER, pygame.Rect(x_position, y_position, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE), width=1)


def draw_vision_lines(window, model: Model, vision_lines: List[VisionLine], offset_x, offset_y) -> None:
    # loop over all lines in given vision lines

    font = pygame.font.SysFont("arial", 20)

    for line in vision_lines:
        line_label = font.render(line.direction.name[0], True, ViewConsts.COLOR_BLACK)

        # render vision line text at wall position
        window.blit(line_label, [line.wall_coord[1] * ViewConsts.SQUARE_SIZE + ViewConsts.SQUARE_SIZE // 4 + offset_x, line.wall_coord[0] * ViewConsts.SQUARE_SIZE + offset_y])

        # draw line from head to wall, draw before body and apple lines
        # drawing uses SQUARE_SIZE//2 so that lines go through the middle of the squares
        line_end_x = model.snake.body[0][1] * ViewConsts.SQUARE_SIZE + ViewConsts.SQUARE_SIZE // 2 + offset_x
        line_end_y = model.snake.body[0][0] * ViewConsts.SQUARE_SIZE + ViewConsts.SQUARE_SIZE // 2 + offset_y

        # draw line form snake head until wall block
        draw_vision_line(window, ViewConsts.COLOR_APPLE, 1, line.wall_coord[1], line.wall_coord[0], line_end_x, line_end_y, offset_x, offset_y)

        # draw another line from snake head to first segment found
        if line.segment_coord is not None:
            draw_vision_line(window, ViewConsts.COLOR_RED, 5, line.segment_coord[1], line.segment_coord[0], line_end_x, line_end_y, offset_x, offset_y)

        # draw another line from snake to apple if apple is found
        if line.apple_coord is not None:
            draw_vision_line(window, ViewConsts.COLOR_GREEN, 5, line.apple_coord[1], line.apple_coord[0], line_end_x, line_end_y, offset_x, offset_y)


def draw_vision_line(window, color, width, line_coord_1, line_coord_0, line_end_x, line_end_y, offset_x, offset_y) -> None:
    pygame.draw.line(window, color,
                     (line_coord_1 * ViewConsts.SQUARE_SIZE + ViewConsts.SQUARE_SIZE // 2 + offset_x, line_coord_0 * ViewConsts.SQUARE_SIZE + ViewConsts.SQUARE_SIZE // 2 + offset_y),
                     (line_end_x, line_end_y), width=width)


def draw_neural_network_complete(window, model: Model, vision_lines: List[VisionLine], offset_x, offset_y):
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
        max_y_input = layer.input_size * (ViewConsts.NN_DISPLAY_NEURON_HEIGHT_BETWEEN + ViewConsts.NN_DISPLAY_NEURON_RADIUS * 2)
        max_y_output = layer.output_size * (ViewConsts.NN_DISPLAY_NEURON_HEIGHT_BETWEEN + ViewConsts.NN_DISPLAY_NEURON_RADIUS * 2)
        max_layer = max_y_input if max_y_input > max_y_output else max_y_output
        if max_layer > max_y:
            max_y = max_layer

    for layer_count, layer in enumerate(nn_layers):
        if type(layer) is Dense:
            input_neuron_count = layer.input_size
            output_neuron_count = layer.output_size

            # if it's the first layer only draw input
            if layer_count == 0:
                current_max_y = input_neuron_count * (ViewConsts.NN_DISPLAY_NEURON_HEIGHT_BETWEEN + ViewConsts.NN_DISPLAY_NEURON_RADIUS * 2)
                offset = (max_y - current_max_y) // 2
                neuron_offset_y += offset

                for i in range(input_neuron_count):
                    neuron_x = neuron_offset_x
                    neuron_y = neuron_offset_y
                    neuron_offset_y += ViewConsts.NN_DISPLAY_NEURON_HEIGHT_BETWEEN + ViewConsts.NN_DISPLAY_NEURON_RADIUS * 2
                    line_start_positions.append((neuron_x, neuron_y))

                    if i < input_neuron_count - 4:
                        line_label = font.render(vision_lines[int(i / model.snake.brain.get_dense_layers()[0].input_size)].direction.name + " " + param_type[i % (len(param_type))], True, ViewConsts.COLOR_WHITE)
                        window.blit(line_label, (neuron_x - 125, neuron_y - 10))
                    else:
                        line_label = font.render(MAIN_DIRECTIONS[i % 4].name, True, ViewConsts.COLOR_WHITE)
                        window.blit(line_label, (neuron_x - 125, neuron_y - 10))

                    inner_color = ViewConsts.COLOR_GREEN * layer.input[i]
                    inner_color[inner_color > 255] = 255
                    inner_color[inner_color < 0] = 0
                    pygame.draw.circle(window, inner_color, (neuron_x, neuron_y), ViewConsts.NN_DISPLAY_NEURON_RADIUS)

                    pygame.draw.circle(window, ViewConsts.COLOR_WHITE, (neuron_x, neuron_y), ViewConsts.NN_DISPLAY_NEURON_RADIUS, width=1)

                neuron_offset_x += ViewConsts.NN_DISPLAY_NEURON_WIDTH_BETWEEN
                neuron_offset_y = offset_y

            # always draw output neurons
            current_max_y = output_neuron_count * (ViewConsts.NN_DISPLAY_NEURON_HEIGHT_BETWEEN + ViewConsts.NN_DISPLAY_NEURON_RADIUS * 2)
            offset = (max_y - current_max_y) // 2
            neuron_offset_y += offset

            for j in range(output_neuron_count):
                neuron_x = neuron_offset_x
                neuron_y = neuron_offset_y
                neuron_offset_y += ViewConsts.NN_DISPLAY_NEURON_HEIGHT_BETWEEN + ViewConsts.NN_DISPLAY_NEURON_RADIUS * 2
                line_end_positions.append((neuron_x, neuron_y))

                if layer_count == len(nn_layers) - 2:
                    line_label = font.render(MAIN_DIRECTIONS[j].name, True, ViewConsts.COLOR_WHITE)
                    window.blit(line_label, (neuron_x + 25, neuron_y - 10))

                neuron_output = nn_layers[layer_count + 1].output[j]
                if neuron_output <= 0:
                    inner_color = ViewConsts.COLOR_BLACK
                else:
                    inner_color = ViewConsts.COLOR_GREEN * neuron_output
                pygame.draw.circle(window, inner_color, (neuron_x, neuron_y), ViewConsts.NN_DISPLAY_NEURON_RADIUS)

                pygame.draw.circle(window, ViewConsts.COLOR_WHITE, (neuron_x, neuron_y), ViewConsts.NN_DISPLAY_NEURON_RADIUS, width=1)

            neuron_offset_x += ViewConsts.NN_DISPLAY_NEURON_WIDTH_BETWEEN
            neuron_offset_y = offset_y

            # self.draw_lines_between_neurons(line_start_positions, line_end_positions)
            line_start_positions = line_end_positions
            line_end_positions = []


def draw_lines_between_neurons(window, line_end: List[Tuple], line_start: List[Tuple]):
    for start_pos in line_start:
        for end_pos in line_end:
            pygame.draw.line(window, ViewConsts.COLOR_WHITE, start_pos, end_pos, width=1)


def draw_colored_lines_between_neurons(window, layer: Dense, line_end: List, line_start: List):
    for i in range(len(line_end)):
        for j in range(len(line_start)):
            if layer.weights[i][j] < 0:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)

            pygame.draw.line(window, color, line_start[j], line_end[i], width=1)


def draw_next_snake_direction(window, board: List[List[str]], prediction: Direction, offset_x, offset_y) -> None:
    head = find_snake_head_poz(board)
    font_size = 15
    current_x = head[1] * ViewConsts.SQUARE_SIZE + offset_x + ViewConsts.SQUARE_SIZE // 2 - font_size // 2
    current_y = head[0] * ViewConsts.SQUARE_SIZE + offset_y + ViewConsts.SQUARE_SIZE // 2 - font_size // 2
    font = pygame.font.SysFont("arial", font_size)

    # draw next position of snake
    next_position = [head[0] + prediction.value[0], head[1] + prediction.value[1]]
    next_x = next_position[1] * ViewConsts.SQUARE_SIZE + offset_x
    next_y = next_position[0] * ViewConsts.SQUARE_SIZE + offset_y
    pygame.draw.rect(window, ViewConsts.COLOR_BLACK, pygame.Rect(next_x, next_y, ViewConsts.SQUARE_SIZE, ViewConsts.SQUARE_SIZE))

    # write letters for directions
    right_text = font.render("D", True, ViewConsts.COLOR_GREEN)
    window.blit(right_text, (current_x + ViewConsts.SQUARE_SIZE, current_y))

    left_text = font.render("A", True, ViewConsts.COLOR_GREEN)
    window.blit(left_text, (current_x - ViewConsts.SQUARE_SIZE, current_y))

    down_text = font.render("S", True, ViewConsts.COLOR_GREEN)
    window.blit(down_text, (current_x, current_y + ViewConsts.SQUARE_SIZE))

    up_text = font.render("W", True, ViewConsts.COLOR_GREEN)
    window.blit(up_text, (current_x, current_y - ViewConsts.SQUARE_SIZE))
