import math
import random
from typing import Tuple, List

import numpy as np

from constants import MAIN_DIRECTIONS, Direction
from snake import Snake


class Model:
    def __init__(self, board_size: int, snake_size: int):
        self.size = board_size + 2
        self.board = np.empty((self.size, self.size), dtype=object)
        self.snake = Snake()

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

    # TODO make sure there are no duplicates
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
            if self.board[new_block[0], new_block[1]] == "X":
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
        start = [head[0] + direction.value[0], head[1] + direction.value[1]]

        # booleans are used to store the first value found
        apple_found = False
        segment_found = False

        # loop are blocks in the given direction and store position and coordinates of apple and snake segments
        while self.board[start[0]][start[1]] != "W":
            if self.board[start[0]][start[1]] == "A" and apple_found == False:
                apple_distance = math.dist(head, start)
                apple_coord = start
                apple_found = True
            elif self.board[start[0]][start[1]] == "S" and segment_found == False:
                segment_distance = math.dist(head, start)
                segment_coord = start
                segment_found = True
            start = [start[0] + direction.value[0], start[1] + direction.value[1]]

        # at the end of the loop start block is at the position of a wall
        wall_distance = math.dist(head, start)
        wall_coord = start

        if return_type == "boolean":
            apple_boolean = 1.0 if apple_found else 0.0
            segment_boolean = 1.0 if segment_found else 0.0

            # use 1/wall distance to get values between 0 and 1 , better for neural network
            vision = {
                "W": [wall_coord, 1 / wall_distance],
                "A": [apple_coord, apple_boolean],
                "S": [segment_coord, segment_boolean]
            }
            return vision

    def get_vision_lines(self, vision_line_number: int, return_type: str) -> {}:

        if vision_line_number == 8:
            return {
                "+X": self.__look_in_direction(Direction.RIGHT, return_type),
                "-X": self.__look_in_direction(Direction.LEFT, return_type),
                "-Y": self.__look_in_direction(Direction.DOWN, return_type),
                "+Y": self.__look_in_direction(Direction.UP, return_type),
                "Q1": self.__look_in_direction(Direction.QUADRANT1, return_type),
                "Q2": self.__look_in_direction(Direction.QUADRANT2, return_type),
                "Q3": self.__look_in_direction(Direction.QUADRANT3, return_type),
                "Q4": self.__look_in_direction(Direction.QUADRANT4, return_type)
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

        if len(valid_directions) == 0:
            return False

        direction = random.choice(valid_directions)

        new_head = [head[0] + direction.value[0], head[1] + direction.value[1]]
        self.snake.body.insert(0, new_head)

        if self.board[new_head[0], new_head[1]] == "A":
            self.__update_board_from_snake()
            self.__place_new_apple()
        else:
            self.snake.body = self.snake.body[:-1]

        self.__update_board_from_snake()

        return True
