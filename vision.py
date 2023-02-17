from typing import List

import numpy as np

from constants import *
from settings import NNSettings


def distance(a, b):
    return manhattan_distance(a, b)


def manhattan_distance(a, b):
    return sum(abs(val1 - val2) for val1, val2 in zip(a, b))


def find_snake_head_poz(board: List[List[str]]) -> []:
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == BoardConsts.SNAKE_HEAD:
                return [i, j]


class VisionLine:
    def __init__(self, wall_coord, wall_distance, apple_coord, apple_distance, segment_coord, segment_distance, direction: Direction):
        self.wall_coord = wall_coord
        self.wall_distance = wall_distance
        self.apple_coord = apple_coord
        self.apple_distance = apple_distance
        self.segment_coord = segment_coord
        self.segment_distance = segment_distance
        self.direction = direction


def look_in_direction(board: List[List[str]], direction: Direction, vision_return_type: str) -> VisionLine:
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
            apple_coord = current_block
            apple_found = True
        elif board[current_block[0]][current_block[1]] == BoardConsts.SNAKE_BODY and segment_found is False:
            segment_distance = distance(head_position, current_block)
            segment_coord = current_block
            segment_found = True
        current_block = [current_block[0] + direction.value[0], current_block[1] + direction.value[1]]

    wall_distance = distance(head_position, current_block)
    wall_coord = current_block

    if NNSettings.VISION_LINES_RETURN_TYPE == "boolean":
        wall_distance_output = 1 / wall_distance
        apple_boolean = 1.0 if apple_found else 0.0
        segment_boolean = 1.0 if segment_found else 0.0

        return VisionLine(wall_coord, wall_distance_output, apple_coord, apple_boolean, segment_coord, segment_boolean, direction)

    elif NNSettings.VISION_LINES_RETURN_TYPE == "distance":
        wall_distance_output = wall_distance
        apple_distance_output = 1 / apple_distance
        segment_distance_output = segment_distance

        return VisionLine(wall_coord, wall_distance_output, apple_coord, apple_distance_output, segment_coord, segment_distance_output, direction)


def get_vision_lines(board: List[List[str]], input_direction_count: int, vision_return_type: str) -> List[VisionLine]:
    if input_direction_count == 8:
        vision_lines = [look_in_direction(board, Direction.RIGHT, vision_return_type),
                        look_in_direction(board, Direction.LEFT, vision_return_type),
                        look_in_direction(board, Direction.DOWN, vision_return_type),
                        look_in_direction(board, Direction.UP, vision_return_type),
                        look_in_direction(board, Direction.Q1, vision_return_type),
                        look_in_direction(board, Direction.Q2, vision_return_type),
                        look_in_direction(board, Direction.Q3, vision_return_type),
                        look_in_direction(board, Direction.Q4, vision_return_type)]
    else:
        vision_lines = [look_in_direction(board, Direction.RIGHT, vision_return_type),
                        look_in_direction(board, Direction.LEFT, vision_return_type),
                        look_in_direction(board, Direction.DOWN, vision_return_type),
                        look_in_direction(board, Direction.UP, vision_return_type)]

    return vision_lines


# todo add vision return type parameter
def get_dynamic_vision_lines(board: List[List[str]], current_direction: Direction) -> List[VisionLine]:
    vision_lines = []
    match current_direction:
        case Direction.UP:
            vision_lines = [look_in_direction(board, Direction.RIGHT), look_in_direction(board, Direction.LEFT), look_in_direction(board, Direction.UP)]
        case Direction.DOWN:
            vision_lines = [look_in_direction(board, Direction.RIGHT), look_in_direction(board, Direction.LEFT), look_in_direction(board, Direction.DOWN)]
        case Direction.RIGHT:
            vision_lines = [look_in_direction(board, Direction.RIGHT), look_in_direction(board, Direction.DOWN), look_in_direction(board, Direction.UP)]
        case Direction.LEFT:
            vision_lines = [look_in_direction(board, Direction.LEFT), look_in_direction(board, Direction.DOWN), look_in_direction(board, Direction.UP)]
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
