import numpy as np
from constants import *
from typing import List


def distance(a, b):
    return manhattan_distance(a, b)


def manhattan_distance(a, b):
    return sum(abs(val1 - val2) for val1, val2 in zip(a, b))


class VisionLine:
    def __init__(self, wall_coord, wall_distance, apple_coord, apple_distance, segment_coord, segment_distance):
        self.wall_coord = wall_coord
        self.wall_distance = wall_distance
        self.apple_coord = apple_coord
        self.apple_distance = apple_distance
        self.segment_coord = segment_coord
        self.segment_distance = segment_distance


class Vision:
    @staticmethod
    def find_snake_head_poz(board: List[str]) -> []:
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == "H":
                    return [i, j]

    @staticmethod
    def look_in_direction(board: List[str], direction: Direction) -> {}:
        apple_distance = np.inf
        segment_distance = np.inf
        apple_coord = None
        segment_coord = None

        # search starts at one block in the given direction
        # otherwise head is also check in the loop
        head_position = Vision.find_snake_head_poz(board)
        current_block = [head_position[0] + direction.value[0], head_position[1] + direction.value[1]]

        # booleans are used to store the first value found
        apple_found = False
        segment_found = False

        # loop are blocks in the given direction and store position and coordinates of apple and snake segments
        while board[current_block[0]][current_block[1]] != "W":
            if board[current_block[0]][current_block[1]] == "A" and apple_found == False:
                apple_distance = distance(head_position, current_block)
                apple_coord = current_block
                apple_found = True
            elif board[current_block[0]][current_block[1]] == "S" and segment_found == False:
                segment_distance = distance(head_position, current_block)
                segment_coord = current_block
                segment_found = True
            current_block = [current_block[0] + direction.value[0], current_block[1] + direction.value[1]]

        wall_distance = distance(head_position, current_block)
        wall_coord = current_block

        if VISION_LINES_RETURN_TYPE == "boolean":
            wall_distance_output = 1 / wall_distance
            apple_boolean = 1.0 if apple_found else 0.0
            segment_boolean = 1.0 if segment_found else 0.0

            return VisionLine(wall_coord, wall_distance_output, apple_coord, apple_boolean, segment_coord, segment_boolean)

        elif VISION_LINES_RETURN_TYPE == "distance":
            wall_distance_output = wall_distance
            apple_distance_output = 1 / apple_distance
            segment_distance_output = segment_distance

            return VisionLine(wall_coord, wall_distance_output, apple_coord, apple_distance_output, segment_coord, segment_distance_output)

    @staticmethod
    def get_vision_lines(board: List[str]) -> {}:
        if INPUT_DIRECTION_COUNT == 8:
            return {
                "+X": Vision.look_in_direction(board, Direction.RIGHT),
                "-X": Vision.look_in_direction(board, Direction.LEFT),
                "-Y": Vision.look_in_direction(board, Direction.DOWN),
                "+Y": Vision.look_in_direction(board, Direction.UP),
                "Q1": Vision.look_in_direction(board, Direction.Q1),
                "Q2": Vision.look_in_direction(board, Direction.Q2),
                "Q3": Vision.look_in_direction(board, Direction.Q3),
                "Q4": Vision.look_in_direction(board, Direction.Q4)
            }
        else:
            return {
                "+X": Vision.look_in_direction(board, Direction.RIGHT),
                "-X": Vision.look_in_direction(board, Direction.LEFT),
                "-Y": Vision.look_in_direction(board, Direction.DOWN),
                "+Y": Vision.look_in_direction(board, Direction.UP)
            }

    @staticmethod
    def get_dynamic_vision_lines(board: List[str], current_direction: Direction) -> {}:
        match current_direction:
            case Direction.UP:
                return {
                    "+X": Vision.look_in_direction(board, Direction.RIGHT),
                    "-X": Vision.look_in_direction(board, Direction.LEFT),
                    "+Y": Vision.look_in_direction(board, Direction.UP)
                }
            case Direction.DOWN:
                return {
                    "+X": Vision.look_in_direction(board, Direction.RIGHT),
                    "-X": Vision.look_in_direction(board, Direction.LEFT),
                    "-Y": Vision.look_in_direction(board, Direction.DOWN)
                }
            case Direction.RIGHT:
                return {
                    "+X": Vision.look_in_direction(board, Direction.RIGHT),
                    "-Y": Vision.look_in_direction(board, Direction.DOWN),
                    "+Y": Vision.look_in_direction(board, Direction.UP)
                }
            case Direction.LEFT:
                return {
                    "-X": Vision.look_in_direction(board, Direction.LEFT),
                    "-Y": Vision.look_in_direction(board, Direction.DOWN),
                    "+Y": Vision.look_in_direction(board, Direction.UP)
                }

    @staticmethod
    def get_parameters_in_nn_input_form(vision_lines, current_direction: Direction) -> np.ndarray:
        nn_input = []
        for line in vision_lines:
            nn_input.append(vision_lines[line].wall_distance)
            nn_input.append(vision_lines[line].apple_distance)
            nn_input.append(vision_lines[line].segment_distance)

        for direction in MAIN_DIRECTIONS:
            if current_direction == direction:
                nn_input.append(1)
            else:
                nn_input.append(0)

        return np.reshape(nn_input, (len(nn_input), 1))
