from typing import List

import numpy as np
from scipy.spatial.distance import chebyshev

from game_config import *


# todo chebishev distance
def distance(a, b):
    return chebyshev(a, b)


def manhattan_distance(a, b):
    return sum(abs(val1 - val2) for val1, val2 in zip(a, b))


def find_snake_head_poz(board: np.ndarray) -> np.ndarray:
    indices = np.argwhere(board == BoardConsts.SNAKE_HEAD)
    if len(indices) > 0:
        return indices[0]
    else:
        return np.array([])


class VisionLine:
    # TODO maybe use just obstacle, remove segment and wall -> obstacle
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


def look_in_direction(board: np.ndarray, direction: Direction, vision_return_type: str) -> VisionLine:
    apple_distance = 99999
    segment_distance = 99999
    apple_coord = None
    segment_coord = None

    # search starts at one block in the given direction
    # otherwise head is also checked in the loop
    # Find the head position
    head_position = find_snake_head_poz(board)
    current_block = [head_position[0] + direction.value[0], head_position[1] + direction.value[1]]

    # booleans are used to store the first value found
    apple_found = False
    segment_found = False

    # loop the blocks in the given direction and store position and coordinates of apple and snake segments
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

    # TODO when using distance, different values for wall, segment  and apple
    # TODO segment distance instead of bool when using boolean vision
    if vision_return_type == "boolean":
        wall_distance_output = 1 / wall_distance
        apple_boolean = 1.0 if apple_found else 0.0
        segment_boolean = 1.0 if segment_found else 0.0

        return VisionLine(wall_coord, wall_distance_output, apple_coord, apple_boolean, segment_coord, segment_boolean, direction)

    elif vision_return_type == "distance":
        wall_distance_output = wall_distance
        apple_distance_output = 1 / apple_distance
        # 1/segment distance otherwise segment_distance output is infinite
        segment_distance_output = 1 / segment_distance

        return VisionLine(wall_coord, wall_distance_output, apple_coord, apple_distance_output, segment_coord, segment_distance_output, direction)


def get_vision_lines(board: np.ndarray, input_direction_count: int, vision_return_type: str) -> List[VisionLine]:
    directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
    if input_direction_count == 8:
        directions.extend([Direction.Q1, Direction.Q2, Direction.Q3, Direction.Q4])

    vision_lines = [look_in_direction(board, direction, vision_return_type) for direction in directions]

    return vision_lines


# def get_dynamic_vision_lines(board: List[List[str]], current_direction: Direction, vision_return_type: str) -> List[VisionLine]:
#     vision_lines = []
#     match current_direction:
#         case Direction.UP:
#             vision_lines = [look_in_direction(board, Direction.RIGHT, vision_return_type), look_in_direction(board, Direction.LEFT, vision_return_type), look_in_direction(board, Direction.UP, vision_return_type)]
#         case Direction.DOWN:
#             vision_lines = [look_in_direction(board, Direction.RIGHT, vision_return_type), look_in_direction(board, Direction.LEFT, vision_return_type), look_in_direction(board, Direction.DOWN, vision_return_type)]
#         case Direction.RIGHT:
#             vision_lines = [look_in_direction(board, Direction.RIGHT, vision_return_type), look_in_direction(board, Direction.DOWN, vision_return_type), look_in_direction(board, Direction.UP, vision_return_type)]
#         case Direction.LEFT:
#             vision_lines = [look_in_direction(board, Direction.LEFT, vision_return_type), look_in_direction(board, Direction.DOWN, vision_return_type), look_in_direction(board, Direction.UP, vision_return_type)]
#     return vision_lines


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
