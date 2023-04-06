from typing import List, Tuple

import numpy as np
from scipy.spatial.distance import chebyshev

from game_config import *


def distance(a, b):
    return chebyshev_distance(a, b)


def manhattan_distance(a, b):
    return sum(abs(val1 - val2) for val1, val2 in zip(a, b))


def chebyshev_distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def find_snake_head_poz(board: np.ndarray) -> np.ndarray:
    indices = np.argwhere(board == BoardConsts.SNAKE_HEAD)
    if len(indices) > 0:
        return indices[0]
    else:
        return np.array([])


class VisionLine:
    def __init__(self, wall_coord, wall_distance, apple_coord, apple_distance, segment_coord, segment_distance, direction: Direction):
        self.wall_coord = wall_coord
        self.wall_distance = wall_distance
        self.apple_coord = apple_coord
        self.apple_distance = apple_distance
        self.segment_coord = segment_coord
        self.segment_distance = segment_distance
        self.direction = direction

    def __eq__(self, other):
        return self.wall_coord == other.wall_coord and self.wall_distance == other.wall_distance and self.apple_coord == other.apple_coord and self.apple_distance == other.apple_distance and self.segment_coord == other.segment_coord and self.segment_distance == other.segment_distance


def get_vision_lines(board: np.ndarray, input_direction_count: int, vision_return_type: str) -> List[VisionLine]:
    directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
    if input_direction_count == 8:
        directions += [Direction.Q1, Direction.Q2, Direction.Q3, Direction.Q4]

    vision_lines = [look_in_direction(board, d, vision_return_type) for d in directions]
    return vision_lines


def look_in_direction(board: np.ndarray, direction: Direction, vision_return_type: str) -> VisionLine:
    apple_distance = np.inf
    segment_distance = np.inf
    apple_coord = None
    segment_coord = None

    # search starts at one block in the given direction
    # otherwise head is also check in the loop
    head_position = find_snake_head_poz(board)
    current_block = [head_position[0] + direction.value[0], head_position[1] + direction.value[1]]

    # booleans are used to store the first value found
    apple_found = False
    segment_found = False

    # loop are blocks in the given direction and store position and coordinates of apple and snake segments
    while board[current_block[0]][current_block[1]] != BoardConsts.WALL:
        if board[current_block[0]][current_block[1]] == BoardConsts.APPLE and apple_found is False:
            apple_distance = distance(head_position, current_block)
            apple_coord = [int(current_block[0]), int(current_block[1])]
            apple_found = True
        elif board[current_block[0]][current_block[1]] == BoardConsts.SNAKE_BODY and segment_found is False:
            segment_distance = distance(head_position, current_block)
            segment_coord = [int(current_block[0]), int(current_block[1])]
            segment_found = True
        current_block = [current_block[0] + direction.value[0], current_block[1] + direction.value[1]]

    wall_distance = distance(head_position, current_block)
    wall_coord = [int(current_block[0]), int(current_block[1])]

    if vision_return_type == "boolean":
        wall_distance_output = 1 / wall_distance
        apple_boolean = 1 / apple_distance
        segment_boolean = 1 / segment_distance

        return VisionLine(wall_coord, wall_distance_output, apple_coord, apple_boolean, segment_coord, segment_boolean, direction)


# TODO distance for segment
# TODO normalize distance using dist/max_distance


def look_in_direction_snake_head(board: np.ndarray, snake_head, direction: Direction, vision_return_type: str) -> VisionLine:
    apple_distance = np.inf
    segment_distance = np.inf
    apple_coord = None
    segment_coord = None
    apple_found = False
    segment_found = False

    # search starts at one block in the given direction otherwise head is also check in the loop
    current_block = [snake_head[0] + direction.value[0], snake_head[1] + direction.value[1]]

    # loop the blocks in the given direction and store position and coordinates
    while board[current_block[0]][current_block[1]] != BoardConsts.WALL:
        if board[current_block[0]][current_block[1]] == BoardConsts.APPLE and apple_found is False:
            apple_coord = current_block
            apple_distance = distance(snake_head, current_block)
            apple_found = True
        elif board[current_block[0]][current_block[1]] == BoardConsts.SNAKE_BODY and segment_found is False:
            segment_coord = current_block
            segment_distance = distance(snake_head, current_block)
            segment_found = True
        current_block = [current_block[0] + direction.value[0], current_block[1] + direction.value[1]]

    wall_distance = distance(snake_head, current_block)
    wall_coord = current_block

    if vision_return_type == "boolean":
        wall_distance_output = 1 / wall_distance
        apple_boolean = 1 / apple_distance
        segment_boolean = 1 / segment_distance

        vision_line = VisionLine(wall_coord, wall_distance_output, apple_coord, apple_boolean, segment_coord, segment_boolean, direction)
        return vision_line


def get_vision_lines_snake_head(board: np.ndarray, snake_head, vision_direction_count: int, vision_return_type: str) -> List[VisionLine]:
    directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
    if vision_direction_count == 8:
        directions += [Direction.Q1, Direction.Q2, Direction.Q3, Direction.Q4]

    vision_lines = [look_in_direction_snake_head(board, snake_head, d, vision_return_type) for d in directions]
    return vision_lines


def get_parameters_in_nn_input_form(vision_lines: List[VisionLine], current_direction: Direction) -> np.ndarray:
    nn_input = []
    for line in vision_lines:
        nn_input.append(line.wall_distance)
        nn_input.append(line.apple_distance)
        nn_input.append(line.segment_distance)

    for direction in MAIN_DIRECTIONS:
        if current_direction == direction:
            nn_input.append(1)
        else:
            nn_input.append(0)

    return np.reshape(nn_input, (len(nn_input), 1))
