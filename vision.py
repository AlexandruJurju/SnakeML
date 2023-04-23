from typing import List

import numpy as np

from game_config import *


def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


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


# def get_vision_lines_snake_model(model: Model, vision_direction_count: int, apple_return_type: str, segment_return_type: str, distance_function) -> List[VisionLine]:
#     directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
#     if vision_direction_count == 8:
#         directions += [Direction.Q1, Direction.Q2, Direction.Q3, Direction.Q4]
#
#     vision_lines = []
#     for direction in directions:
#         apple_coord = None
#         segment_coord = None
#
#         # search starts at one block in the given direction otherwise head is also check in the loop
#         current_block = [model.snake.body[0][0] + direction.value[0], model.snake.body[0][1] + direction.value[1]]
#
#         # loop the blocks in the given direction and store position and coordinates
#         while model.board[current_block[0]][current_block[1]] != BoardConsts.WALL:
#             if model.board[current_block[0]][current_block[1]] == BoardConsts.APPLE and apple_coord is None:
#                 apple_coord = current_block
#             elif model.board[current_block[0]][current_block[1]] == BoardConsts.SNAKE_BODY and segment_coord is None:
#                 segment_coord = current_block
#             current_block = [current_block[0] + direction.value[0], current_block[1] + direction.value[1]]
#
#         wall_distance = distance_function(model.snake.body[0], current_block)
#         wall_output = 1 / wall_distance
#         wall_coord = current_block
#
#         if wall_output > 1:
#             print("GREATER")
#
#         if apple_return_type == "boolean":
#             apple_output = 1.0 if apple_coord is not None else 0.0
#         else:
#             apple_output = 1.0 / distance_function(model.snake.body[0], apple_coord) if apple_coord is not None else 0.0
#
#         if segment_return_type == "boolean":
#             segment_output = 1.0 if segment_coord is not None else 0.0
#         else:
#             segment_output = 1.0 / distance_function(model.snake.body[0], segment_coord) if segment_coord is not None else 0.0
#
#         vision_lines.append(VisionLine(wall_coord, wall_output, apple_coord, apple_output, segment_coord, segment_output, direction))
#     return vision_lines
#
#
# def look_in_direction_snake_head(board: np.ndarray, snake_head, direction: Direction, max_dist, apple_return_type, segment_return_type, distance_function) -> VisionLine:
#     apple_coord = None
#     segment_coord = None
#
#     # search starts at one block in the given direction otherwise head is also check in the loop
#     current_block = [snake_head[0] + direction.value[0], snake_head[1] + direction.value[1]]
#
#     # loop the blocks in the given direction and store position and coordinates
#     while board[current_block[0]][current_block[1]] != BoardConsts.WALL:
#         if board[current_block[0]][current_block[1]] == BoardConsts.APPLE and apple_coord is None:
#             apple_coord = current_block
#         elif board[current_block[0]][current_block[1]] == BoardConsts.SNAKE_BODY and segment_coord is None:
#             segment_coord = current_block
#         current_block = [current_block[0] + direction.value[0], current_block[1] + direction.value[1]]
#
#     wall_distance = distance_function(snake_head, current_block)
#     wall_output = 1 / wall_distance
#     wall_coord = current_block
#
#     if wall_output > 1:
#         print("GREATER")
#
#     if apple_return_type == "boolean":
#         apple_output = 1.0 if apple_coord is not None else 0.0
#     else:
#         apple_output = 1.0 / distance_function(snake_head, apple_coord) if apple_coord is not None else 0.0
#
#     if segment_return_type == "boolean":
#         segment_output = 1.0 if segment_coord is not None else 0.0
#     else:
#         segment_output = 1.0 / distance_function(snake_head, segment_coord) if segment_coord is not None else 0.0
#
#     vision_line = VisionLine(wall_coord, wall_output, apple_coord, apple_output, segment_coord, segment_output, direction)
#     return vision_line


def get_vision_lines_snake_head(board: np.ndarray, snake_head, vision_direction_count: int, max_dist, apple_return_type: str, segment_return_type: str, distance_function) -> List[VisionLine]:
    directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
    if vision_direction_count == 8:
        directions += [Direction.Q1, Direction.Q2, Direction.Q3, Direction.Q4]

    vision_lines = []

    for direction in directions:
        apple_coord = None
        segment_coord = None
        x_offset = direction.value[0]
        y_offset = direction.value[1]

        # search starts at one block in the given direction otherwise head is also check in the loop
        current_block = [snake_head[0] + x_offset, snake_head[1] + y_offset]

        # loop the blocks in the given direction and store position and coordinates
        while board[current_block[0]][current_block[1]] != BoardConsts.WALL:
            if board[current_block[0]][current_block[1]] == BoardConsts.APPLE and apple_coord is None:
                apple_coord = current_block
            elif board[current_block[0]][current_block[1]] == BoardConsts.SNAKE_BODY and segment_coord is None:
                segment_coord = current_block
            current_block = [current_block[0] + x_offset, current_block[1] + y_offset]

        wall_coord = current_block
        wall_distance = distance_function(snake_head, wall_coord)
        wall_output = 1 / wall_distance

        if apple_return_type == "boolean":
            apple_output = 1.0 if apple_coord is not None else 0.0
        else:
            apple_output = 1.0 / distance_function(snake_head, apple_coord) if apple_coord is not None else 0.0

        if segment_return_type == "boolean":
            segment_output = 1.0 if segment_coord is not None else 0.0
        else:
            segment_output = 1.0 / distance_function(snake_head, segment_coord) if segment_coord is not None else 0.0

        vision_lines.append(VisionLine(wall_coord, wall_output, apple_coord, apple_output, segment_coord, segment_output, direction))

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
