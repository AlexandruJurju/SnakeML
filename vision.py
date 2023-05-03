from typing import List

import numpy as np

import cvision
from game_config import *


def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def chebyshev_distance(a: np.ndarray, b: np.ndarray):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class VisionLine:
    def __init__(self, wall_coord, wall_distance: float, apple_coord, apple_distance: float, segment_coord, segment_distance: float, direction: Direction):
        self.wall_coord = wall_coord
        self.wall_distance: float = wall_distance
        self.apple_coord = apple_coord
        self.apple_distance: float = apple_distance
        self.segment_coord = segment_coord
        self.segment_distance: float = segment_distance
        self.direction = direction

    def __eq__(self, other):
        return self.wall_coord == other.wall_coord and self.wall_distance == other.wall_distance and \
            self.apple_coord == other.apple_coord and self.apple_distance == other.apple_distance and \
            self.segment_coord == other.segment_coord and self.segment_distance == other.segment_distance


def find_snake_head_poz(board: np.ndarray) -> np.ndarray:
    indices = np.argwhere(board == BoardConsts.SNAKE_HEAD)
    if len(indices) > 0:
        return indices[0]
    else:
        return np.array([])


def get_parameters_in_nn_input_form_2d(vision_lines, current_direction: Direction) -> np.ndarray:
    size = len(vision_lines) * 3 + 2
    nn_input = [0] * size

    nn_input[:-2:3] = [line.wall_distance for line in vision_lines]
    nn_input[1:-2:3] = [line.apple_distance for line in vision_lines]
    nn_input[2:-2:3] = [line.segment_distance for line in vision_lines]
    nn_input[-2:] = current_direction.value

    return np.reshape(nn_input, (len(nn_input), 1))


def cvision_to_old_vision(old: List[cvision.VisionLine]):
    lines = []
    i = 0
    for line in old:
        segment_coord = None
        apple_coord = None
        wall_coord = None
        if line.segment_coord["x"] != -1:
            segment_coord = [0] * 2
            segment_coord[0] = line.segment_coord["x"]
            segment_coord[1] = line.segment_coord["y"]

        if line.apple_coord["x"] != -1:
            apple_coord = [0] * 2
            apple_coord[0] = line.apple_coord["x"]
            apple_coord[1] = line.apple_coord["y"]

        if line.wall_coord["x"] != -1:
            wall_coord = [0] * 2
            wall_coord[0] = line.wall_coord["x"]
            wall_coord[1] = line.wall_coord["y"]

        new_line = VisionLine(wall_coord, line.wall_distance, apple_coord, line.apple_distance, segment_coord, line.segment_distance, ALL_DIRECTIONS[i])
        lines.append(new_line)
        i += 1

    return lines
