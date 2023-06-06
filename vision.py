from typing import List

import numpy as np

import cvision
from game_config import *


def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


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


def get_vision_lines(board: np.ndarray, snake_head, vision_direction_count: int, apple_return_type: str, segment_return_type: str) -> List[VisionLine]:
    directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
    if vision_direction_count == 8:
        directions += [Direction.Q1, Direction.Q2, Direction.Q3, Direction.Q4]

    vision_lines = []
    for direction in directions:
        apple_coord = None
        segment_coord = None

        # search starts at one block in the given direction otherwise head is also check in the loop
        current_block = [snake_head[0] + direction.value[0], snake_head[1] + direction.value[1]]

        # loop the blocks in the given direction and store position and coordinates
        while board[current_block[0]][current_block[1]] != BoardConsts.WALL:
            if board[current_block[0]][current_block[1]] == BoardConsts.APPLE and apple_coord is None:
                apple_coord = current_block
            elif board[current_block[0]][current_block[1]] == BoardConsts.SNAKE_BODY and segment_coord is None:
                segment_coord = current_block
            current_block = [current_block[0] + direction.value[0], current_block[1] + direction.value[1]]

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

        vision_lines.append(VisionLine(wall_coord, wall_output, apple_coord, apple_output, segment_coord, segment_output, direction))
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


def different(l1, l2):
    for i in range(len(l1)):
        if l1[i] != l2[i]:
            return True
    return False


def get_parameters_in_nn_input_form_4d(vision_lines, current_direction: Direction) -> np.ndarray:
    size = len(vision_lines) * 3 + 4
    nn_input1 = [0] * size

    nn_input1[:-4:3] = [line.wall_distance for line in vision_lines]
    nn_input1[1:-4:3] = [line.apple_distance for line in vision_lines]
    nn_input1[2:-4:3] = [line.segment_distance for line in vision_lines]

    if current_direction == Direction.UP:
        nn_input1[-4] = 1
    if current_direction == Direction.DOWN:
        nn_input1[-3] = 1
    if current_direction == Direction.LEFT:
        nn_input1[-2] = 1
    if current_direction == Direction.RIGHT:
        nn_input1[-1] = 1

    # nn_input2 = []
    # for line in vision_lines:
    #     nn_input2.append(line.wall_distance)
    #     nn_input2.append(line.apple_distance)
    #     nn_input2.append(line.segment_distance)
    #
    # for direction in MAIN_DIRECTIONS:
    #     if current_direction == direction:
    #         nn_input2.append(1)
    #     else:
    #         nn_input2.append(0)
    #
    # if different(nn_input1, nn_input2):
    #     print(nn_input1)
    #     print(nn_input2)
    #     print("FK")

    return np.reshape(nn_input1, (len(nn_input1), 1))

    # nn_input = [value for line in vision_lines for value in (line.wall_distance, line.apple_distance, line.segment_distance)]
    #
    # for direction in MAIN_DIRECTIONS:
    #     nn_input.append(1 if current_direction == direction else 0)
    #
    # return np.reshape(nn_input, (len(nn_input), 1))


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


def put_distances(board: np.ndarray, head):
    head = [5, 5]
    board = np.full((len(board), len(board)), np.nan)  # Initialize as float array
    for i in range(len(board)):
        for j in range(len(board)):
            if [i, j] != head:
                board[i, j] = 1.0 / manhattan_distance(head, [i, j])

    board[head[0], head[1]] = 0

    np.set_printoptions(precision=3)  # Set precision to 3 decimal places
    print(np.asarray(board, dtype=float))
