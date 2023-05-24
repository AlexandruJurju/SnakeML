from typing import List

import numpy as np
from numba import float64, njit, jit
from numba.experimental import jitclass

import cvision
from game_config import *


@njit
def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@jitclass(spec=[
    ('wall_distance', float64),
    ('apple_distance', float64),
    ('segment_distance', float64)
])
class VisionLine:
    def __init__(self, wall_distance: float, apple_distance: float, segment_distance: float):
        self.wall_distance: float = wall_distance
        self.apple_distance: float = apple_distance
        self.segment_distance: float = segment_distance

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


@jit(nopython=True)
def get_vision_lines_snake_head(board: np.ndarray, snake_head, vision_direction_count: int, apple_return_type: str, segment_return_type: str) -> List[VisionLine]:
    APPLE = 2
    WALL = -1
    SNAKE_BODY = -2

    directions = [[0, 0] for _ in range(8)]
    directions[0][0] = -1
    directions[0][1] = 0
    directions[1][0] = 1
    directions[1][1] = 0
    directions[2][0] = 0
    directions[2][1] = -1
    directions[3][0] = 0
    directions[3][1] = 1
    directions[4][0] = -1
    directions[4][1] = 1
    directions[5][0] = -1
    directions[5][1] = -1
    directions[6][0] = 1
    directions[6][1] = -1
    directions[7][0] = 1
    directions[7][1] = 1

    vision_lines = []
    for i in range(vision_direction_count):
        apple_coord = None
        segment_coord = None

        # search starts at one block in the given direction otherwise head is also check in the loop
        current_block = [snake_head[0] + directions[i][0], snake_head[1] + directions[i][1]]

        # loop the blocks in the given direction and store position and coordinates
        while board[current_block[0]][current_block[1]] != WALL:
            if board[current_block[0]][current_block[1]] == APPLE and apple_coord is None:
                apple_coord = current_block
            elif board[current_block[0]][current_block[1]] == SNAKE_BODY and segment_coord is None:
                segment_coord = current_block
            current_block = [current_block[0] + directions[i][0], current_block[1] + directions[i][1]]

        wall_coord = current_block
        wall_distance = manhattan_distance(snake_head, wall_coord)
        wall_output = 1 / wall_distance

        if apple_return_type == "boolean":
            apple_output = 1.0 if apple_coord is not None else 0.0
        else:
            apple_output = 1.0 / manhattan_distance(snake_head, apple_coord) if apple_coord is not None else 0.0

        if segment_return_type == "boolean":
            segment_output = 1.0 if segment_coord is not None else 0.0
        else:
            segment_output = 1.0 / manhattan_distance(snake_head, segment_coord) if segment_coord is not None else 0.0

        vision_lines.append(VisionLine(wall_output, apple_output, segment_output))
    return vision_lines


def get_parameters_in_nn_input_form_0d(vision_lines, current_direction: Direction) -> np.ndarray:
    size = len(vision_lines) * 3
    nn_input = [0] * size

    nn_input[::3] = [line.wall_distance for line in vision_lines]
    nn_input[1::3] = [line.apple_distance for line in vision_lines]
    nn_input[2::3] = [line.segment_distance for line in vision_lines]

    return np.reshape(nn_input, (len(nn_input), 1))


def get_parameters_in_nn_input_form_2d(vision_lines, current_direction: Direction) -> np.ndarray:
    size = len(vision_lines) * 3 + 2
    nn_input = [0] * size

    nn_input[:-2:3] = [line.wall_distance for line in vision_lines]
    nn_input[1:-2:3] = [line.apple_distance for line in vision_lines]
    nn_input[2:-2:3] = [line.segment_distance for line in vision_lines]
    nn_input[-2:] = current_direction.value

    return np.reshape(nn_input, (len(nn_input), 1))


def get_parameters_in_nn_input_form_4d(vision_lines, current_direction: Direction) -> np.ndarray:
    size = len(vision_lines) * 3 + 4
    nn_input = [0] * size

    nn_input[:-4:3] = [line.wall_distance for line in vision_lines]
    nn_input[1:-4:3] = [line.apple_distance for line in vision_lines]
    nn_input[2:-4:3] = [line.segment_distance for line in vision_lines]

    if current_direction == Direction.UP:
        nn_input[-4] = 1
    if current_direction == Direction.DOWN:
        nn_input[-3] = 1
    if current_direction == Direction.LEFT:
        nn_input[-2] = 1
    if current_direction == Direction.RIGHT:
        nn_input[-1] = 1

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
