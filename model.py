import math
import random
from typing import Tuple, List

from Neural.list_neural_network import *
from constants import MAIN_DIRECTIONS, Direction
from snake import Snake


class VisionLine:
    def __init__(self, wall_coord, wall_distance, apple_coord, apple_distance, segment_coord, segment_distance):
        self.wall_coord = wall_coord
        self.wall_distance = wall_distance
        self.apple_coord = apple_coord
        self.apple_distance = apple_distance
        self.segment_coord = segment_coord
        self.segment_distance = segment_distance


class Model:
    def __init__(self, board_size: int, snake_size: int, neural_net: KerasNetwork):
        self.snake_size = snake_size

        self.size = board_size + 2
        self.board = np.empty((self.size, self.size), dtype=object)
        self.snake = Snake(neural_net)

        self.__make_board()
        self.__place_new_apple()
        self.__create_random_snake(snake_size)
        self.__update_board_from_snake()

    def __make_board(self) -> None:
        for i in range(0, self.size):
            for j in range(0, self.size):
                # place walls on the borders and nothing inside
                if i == 0 or i == self.size - 1 or j == 0 or j == self.size - 1:
                    self.board[i, j] = "W"
                else:
                    self.board[i, j] = "X"

    def __clear_snake_on_board(self):
        for i in range(1, self.size):
            for j in range(1, self.size):
                if self.board[i, j] == "S":
                    self.board[i, j] = "X"

    def __get_random_empty_block(self) -> []:
        empty = []

        # find all empty spots on the board
        for i in range(1, self.size):
            for j in range(1, self.size):
                if self.board[i, j] == "X":
                    empty.append([i, j])

        # return random empty block from found empty blocks
        return random.choice(empty)

    def __place_new_apple(self) -> None:
        rand_block = self.__get_random_empty_block()
        self.board[rand_block[0], rand_block[1]] = "A"

    def __get_valid_direction_for_block(self, block: Tuple) -> List[Direction]:
        valid_directions = []

        # check all main direction that the block has
        for direction in MAIN_DIRECTIONS:
            new_block = [direction.value[0] + block[0], direction.value[1] + block[1]]

            #  if it's not a wall or a snake part then it's a valid direction
            if (self.board[new_block[0], new_block[1]] != "W") and (self.board[new_block[0], new_block[1]] != "S"):
                valid_directions.append(direction)

        return valid_directions

    def __create_random_snake(self, snake_size: int) -> None:
        # head is the first block of the snake, the block where the search starts
        head = self.__get_random_empty_block()
        self.snake.body.append(head)

        while len(self.snake.body) < snake_size:
            # get all possible directions of block
            valid_directions = self.__get_valid_direction_for_block(head)

            # choose random direction for new snake piece position
            random_direction = random.choice(valid_directions)

            # get block in chosen direction
            new_block = [head[0] + random_direction.value[0], head[1] + random_direction.value[1]]

            # redundant check if new position is empty and check if piece is already in body
            if self.board[new_block[0], new_block[1]] == "X" and (new_block not in self.snake.body):
                self.snake.body.append(new_block)
                head = new_block

    def __update_board_from_snake(self):
        # remove previous snake position on board
        self.__clear_snake_on_board()

        # loop all snake pieces and put S on board using their coordinates
        for piece in self.snake.body:
            self.board[piece[0], piece[1]] = "S"

    def __look_in_direction(self, direction: Direction, return_type: str) -> {}:
        apple_distance = np.inf
        segment_distance = np.inf

        apple_coord = None
        segment_coord = None

        # search starts at one block in the given direction
        # otherwise head is also check in the loop
        head = self.snake.body[0]
        current_block = [head[0] + direction.value[0], head[1] + direction.value[1]]

        # booleans are used to store the first value found
        apple_found = False
        segment_found = False

        # loop are blocks in the given direction and store position and coordinates of apple and snake segments
        while self.board[current_block[0], current_block[1]] != "W":
            if self.board[current_block[0], current_block[1]] == "A" and apple_found == False:
                apple_distance = math.dist(head, current_block)
                apple_coord = current_block
                apple_found = True
            elif self.board[current_block[0], current_block[1]] == "S" and segment_found == False:
                segment_distance = math.dist(head, current_block)
                segment_coord = current_block
                segment_found = True
            current_block = [current_block[0] + direction.value[0], current_block[1] + direction.value[1]]

        wall_distance = math.dist(head, current_block)
        wall_coord = current_block

        if return_type == "boolean":
            wall_distance_output = 1 / wall_distance
            apple_boolean = 1.0 if apple_found else 0.0
            segment_boolean = 1.0 if segment_found else 0.0

            # vision = {
            #     "W": [wall_coord, wall_distance_output],
            #     "A": [apple_coord, apple_boolean],
            #     "S": [segment_coord, segment_boolean]
            # }

            return VisionLine(wall_coord, 1 / wall_distance, apple_coord, apple_boolean, segment_coord, segment_boolean)

        elif return_type == "distance":
            wall_distance_output = 1 / wall_distance
            apple_distance_output = apple_distance
            segment_distance_output = segment_distance

            # vision = {
            #     "W": [wall_coord, 1 / wall_distance_output],
            #     "A": [apple_coord, apple_distance_output],
            #     "S": [segment_coord, 1 / segment_distance_output]
            # }
            return VisionLine(wall_coord, 1 / wall_distance, apple_coord, apple_distance, segment_coord, 1 / segment_distance)

    def get_vision_lines(self, vision_line_number: int, return_type: str) -> {}:
        if vision_line_number == 8:
            return {
                "+X": self.__look_in_direction(Direction.RIGHT, return_type),
                "-X": self.__look_in_direction(Direction.LEFT, return_type),
                "-Y": self.__look_in_direction(Direction.DOWN, return_type),
                "+Y": self.__look_in_direction(Direction.UP, return_type),
                "Q1": self.__look_in_direction(Direction.Q1, return_type),
                "Q2": self.__look_in_direction(Direction.Q2, return_type),
                "Q3": self.__look_in_direction(Direction.Q3, return_type),
                "Q4": self.__look_in_direction(Direction.Q4, return_type)
            }
        else:
            return {
                "+X": self.__look_in_direction(Direction.RIGHT, return_type),
                "-X": self.__look_in_direction(Direction.LEFT, return_type),
                "-Y": self.__look_in_direction(Direction.DOWN, return_type),
                "+Y": self.__look_in_direction(Direction.UP, return_type)
            }

    def move_random_direction(self) -> bool:
        head = self.snake.body[0]
        valid_directions = self.__get_valid_direction_for_block(head)

        # if block has no valid directions then snake is dead, return false
        if len(valid_directions) == 0:
            return False

        direction = random.choice(valid_directions)

        # new head is the next block in the chosen direction
        new_head = [head[0] + direction.value[0], head[1] + direction.value[1]]

        # insert new head at the start of the list for moving other segments
        self.snake.body.insert(0, new_head)

        # if snake doesn't find an apple then all segments except last are moved one position forward, old_head old1 old2 -> new_head old_head old1 ; same as moving
        # if snake finds an apple then the last segments is not removed when moving
        if self.board[new_head[0], new_head[1]] == "A":
            self.__update_board_from_snake()
            self.__place_new_apple()
        else:
            self.snake.body = self.snake.body[:-1]

        self.__update_board_from_snake()

        return True

    def move_in_direction(self, new_direction: Direction) -> bool:
        self.snake.direction = new_direction

        head = self.snake.body[0]
        next_head = [head[0] + new_direction.value[0], head[1] + new_direction.value[1]]

        new_head_value = self.board[next_head[0], next_head[1]]
        if (new_head_value == "W") or (new_head_value == "S"):
            return False

        self.snake.body.insert(0, next_head)

        if new_head_value == "A":
            self.__update_board_from_snake()
            self.__place_new_apple()
        else:
            self.snake.body = self.snake.body[:-1]
            self.__update_board_from_snake()

        return True

    def get_parameters_in_nn_input_form(self) -> np.ndarray:
        nn_input = []
        vision_lines = self.get_vision_lines(8, "boolean")
        for line in vision_lines:
            nn_input.append(vision_lines[line].wall_distance)
            nn_input.append(vision_lines[line].apple_distance)
            nn_input.append(vision_lines[line].segment_distance)

        for direction in MAIN_DIRECTIONS:
            if self.snake.direction == direction:
                nn_input.append(1.0)
            else:
                nn_input.append(0.0)

        return np.reshape(nn_input, (len(nn_input), 1))

    def get_nn_output(self) -> np.ndarray:
        nn_input = self.get_parameters_in_nn_input_form()
        return self.snake.brain.feed_forward(nn_input)

    # def get_direction_from_nn_output(self) -> Direction:
    #     nn_input = self.get_parameters_in_nn_input_form()
    #     output = self.snake.brain.feed_forward(nn_input)
    #
    #     next_direction = MAIN_DIRECTIONS[list(output).index(max(list(output)))]
    #     return next_direction

    def get_3_directions_from_neural_net(self) -> Direction:
        nn_input = self.get_parameters_in_nn_input_form()
        output = self.snake.brain.feed_forward(nn_input)

        direction_index = list(output).index(max(list(output)))

        if direction_index == 0:
            return self.snake.direction

        if direction_index == 1:
            match self.snake.direction:
                case Direction.UP:
                    return Direction.LEFT
                case Direction.LEFT:
                    return Direction.DOWN
                case Direction.DOWN:
                    return Direction.RIGHT
                case Direction.RIGHT:
                    return Direction.UP

        if direction_index == 2:
            match self.snake.direction:
                case Direction.UP:
                    return Direction.RIGHT
                case Direction.LEFT:
                    return Direction.UP
                case Direction.DOWN:
                    return Direction.LEFT
                case Direction.RIGHT:
                    return Direction.DOWN

    def reinit_model(self):
        self.board = np.empty((self.size, self.size), dtype=object)
        self.__make_board()
        self.__place_new_apple()
        self.snake.body = []
        self.snake.direction = random.choice(MAIN_DIRECTIONS)
        self.__create_random_snake(self.snake_size)
        self.__update_board_from_snake()
        self.snake.brain.reinit_layers()
