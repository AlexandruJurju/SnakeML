
import numpy as np

from game_config import *


def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def chebyshev_distance(a: np.ndarray, b: np.ndarray):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def find_snake_head_poz(board: np.ndarray) -> np.ndarray:
    indices = np.argwhere(board == BoardConsts.SNAKE_HEAD)
    if len(indices) > 0:
        return indices[0]
    else:
        return np.array([])

def get_parameters_in_nn_input_form_4d(vision_lines, current_direction: Direction) -> np.ndarray:
    nn_input = []
    for line in vision_lines:
        nn_input.append(line.wall_dist)
        nn_input.append(line.apple_dist)
        nn_input.append(line.segment_dist)

    for direction in MAIN_DIRECTIONS:
        if current_direction == direction:
            nn_input.append(1)
        else:
            nn_input.append(0)

    return np.reshape(nn_input, (len(nn_input), 1))


def get_parameters_in_nn_input_form_2d(vision_lines, current_direction: Direction) -> np.ndarray:
    size = len(vision_lines) * 3 + 2
    nn_input = [0] * size

    nn_input[:-2:3] = [line.wall_distance for line in vision_lines]
    nn_input[1:-2:3] = [line.apple_distance for line in vision_lines]
    nn_input[2:-2:3] = [line.segment_distance for line in vision_lines]
    nn_input[-2:] = current_direction.value

    return np.reshape(nn_input, (len(nn_input), 1))
