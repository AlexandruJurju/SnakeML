import math

import numpy as np

from constants import Direction, MAIN_DIRECTIONS


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
    def find_snake_head_poz(model):
        for i in range(1, len(model)):
            for j in range(1, len(model)):
                if model[i, j] == "H":
                    return [i, j]

    @staticmethod
    def look_in_direction(model, direction: Direction, return_type: str) -> {}:
        apple_distance = np.inf
        segment_distance = np.inf
        apple_coord = None
        segment_coord = None

        # search starts at one block in the given direction
        # otherwise head is also check in the loop
        head = Vision.find_snake_head_poz(model)
        current_block = [head[0] + direction.value[0], head[1] + direction.value[1]]

        # booleans are used to store the first value found
        apple_found = False
        segment_found = False

        # loop are blocks in the given direction and store position and coordinates of apple and snake segments
        while model[current_block[0], current_block[1]] != "W":
            if model[current_block[0], current_block[1]] == "A" and apple_found == False:
                apple_distance = distance(head, current_block)
                apple_coord = current_block
                apple_found = True
            elif model[current_block[0], current_block[1]] == "S" and segment_found == False:
                segment_distance = distance(head, current_block)
                segment_coord = current_block
                segment_found = True
            current_block = [current_block[0] + direction.value[0], current_block[1] + direction.value[1]]

        wall_distance = distance(head, current_block)
        wall_coord = current_block

        if return_type == "boolean":
            wall_distance_output = 1 / wall_distance
            apple_boolean = 1.0 if apple_found else 0.0
            segment_boolean = 1.0 if segment_found else 0.0

            return VisionLine(wall_coord, wall_distance_output, apple_coord, apple_boolean, segment_coord, segment_boolean)

        elif return_type == "distance":
            wall_distance_output = wall_distance
            apple_distance_output = 1 / apple_distance
            segment_distance_output = segment_distance

            return VisionLine(wall_coord, wall_distance_output, apple_coord, apple_distance_output, segment_coord, segment_distance_output)

    @staticmethod
    def get_vision_lines(model, vision_line_number: int, return_type: str) -> {}:
        if vision_line_number == 8:
            return {
                "+X": Vision.look_in_direction(model, Direction.RIGHT, return_type),
                "-X": Vision.look_in_direction(model, Direction.LEFT, return_type),
                "-Y": Vision.look_in_direction(model, Direction.DOWN, return_type),
                "+Y": Vision.look_in_direction(model, Direction.UP, return_type),
                "Q1": Vision.look_in_direction(model, Direction.Q1, return_type),
                "Q2": Vision.look_in_direction(model, Direction.Q2, return_type),
                "Q3": Vision.look_in_direction(model, Direction.Q3, return_type),
                "Q4": Vision.look_in_direction(model, Direction.Q4, return_type)
            }
        else:
            return {
                "+X": Vision.look_in_direction(model, Direction.RIGHT, return_type),
                "-X": Vision.look_in_direction(model, Direction.LEFT, return_type),
                "-Y": Vision.look_in_direction(model, Direction.DOWN, return_type),
                "+Y": Vision.look_in_direction(model, Direction.UP, return_type)
            }

    @staticmethod
    def get_dynamic_vision_lines(model, current_direction: Direction) -> {}:
        match current_direction:
            case Direction.UP:
                return {
                    "+X": Vision.look_in_direction(model, Direction.RIGHT, "boolean"),
                    "-X": Vision.look_in_direction(model, Direction.LEFT, "boolean"),
                    "+Y": Vision.look_in_direction(model, Direction.UP, "boolean")
                }
            case Direction.DOWN:
                return {
                    "+X": Vision.look_in_direction(model, Direction.RIGHT, "boolean"),
                    "-X": Vision.look_in_direction(model, Direction.LEFT, "boolean"),
                    "-Y": Vision.look_in_direction(model, Direction.DOWN, "boolean")
                }
            case Direction.RIGHT:
                return {
                    "+X": Vision.look_in_direction(model, Direction.RIGHT, "boolean"),
                    "-Y": Vision.look_in_direction(model, Direction.DOWN, "boolean"),
                    "+Y": Vision.look_in_direction(model, Direction.UP, "boolean")
                }
            case Direction.LEFT:
                return {
                    "-X": Vision.look_in_direction(model, Direction.LEFT, "boolean"),
                    "-Y": Vision.look_in_direction(model, Direction.DOWN, "boolean"),
                    "+Y": Vision.look_in_direction(model, Direction.UP, "boolean")
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
