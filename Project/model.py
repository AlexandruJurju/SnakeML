import random
from typing import List, Tuple

import numpy as np

import cvision
from game_config import BoardConsts, Direction, GameSettings, MAIN_DIRECTIONS
from neural_network import NeuralNetwork


class Snake:
    def __init__(self, neural_network: NeuralNetwork, max_ttl: int):
        self.brain = neural_network
        self.score = 0
        self.fitness = 0
        self.steps_to_apple = 0

        self.body: List = []
        self.TTL: int = GameSettings.SNAKE_MAX_TTL
        self.MAX_TTL: int = max_ttl
        self.steps_taken: int = 0
        self.won: bool = False
        self.past_direction: Direction = None

    def calculate_fitness(self) -> None:
        fitness_score = self.method1()
        self.fitness = fitness_score

    def method1(self) -> float:
        fitness_score = self.steps_taken + ((2 ** self.score) + (self.score ** 2.1) * 500) - (((.25 * self.steps_taken) ** 1.3) * (self.score ** 1.2))
        if self.score == 97:
            fitness_score *= 10 ** 5 * (self.score / self.steps_taken)
        return fitness_score

    def method2(self):
        if self.steps_taken == 0:
            return 0
        return (self.steps_taken * self.score) / (self.steps_taken + self.score)


class Model:
    def __init__(self, model_size: int, snake_size: int, net: NeuralNetwork):
        self.size: int = model_size + 2
        self.board = np.full((self.size, self.size), BoardConsts.EMPTY)

        self.max_score = model_size ** 2 - snake_size
        self.snake_size: int = snake_size
        self.snake: Snake = Snake(net, model_size ** 2)

        self.make_board()
        self.create_random_snake()
        # self.place_snake_at_given_position([[5, 6], [4, 6], [4, 5]], Direction.DOWN)
        self.update_board_from_snake()
        self.place_new_apple()

    def find(self):
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if self.board[i][j] == BoardConsts.APPLE:
                    return True
        return False

    def make_board(self) -> None:
        self.board[1:-1, 1:-1] = BoardConsts.EMPTY
        self.board[0, :] = self.board[-1, :] = self.board[:, 0] = self.board[:, -1] = BoardConsts.WALL

    def place_apple_at_coords(self, position) -> None:
        self.board[position[0]][position[1]] = BoardConsts.APPLE

    def place_snake_at_given_position(self, positions: [], direction: Direction) -> None:
        for i, position in enumerate(positions):
            if i == 0:
                self.board[position[0]][position[1]] = BoardConsts.SNAKE_HEAD
            else:
                self.board[position[0]][position[1]] = BoardConsts.SNAKE_BODY
            self.snake.body.append([position[0], position[1]])
        self.snake.past_direction = direction

    def get_random_empty_block(self) -> []:
        rows, cols = self.board.shape
        empty_blocks = cvision.get_all_random_blocks(self.board, rows, cols)
        return random.choice(empty_blocks)

    def place_new_apple(self) -> None:
        rand_block = self.get_random_empty_block()
        self.board[rand_block[0]][rand_block[1]] = BoardConsts.APPLE

    def get_valid_direction_for_block(self, block: Tuple) -> List[Direction]:
        valid_directions = []

        # check all main direction that the block has
        for direction in MAIN_DIRECTIONS:
            new_block = [direction.value[0] + block[0], direction.value[1] + block[1]]

            #  if it's not a wall or a snake part then it's a valid direction
            if (self.board[new_block[0]][new_block[1]] == BoardConsts.EMPTY) and (new_block not in self.snake.body):
                valid_directions.append(direction)

        return valid_directions

    def create_random_snake(self) -> None:
        # head is the first block of the snake, the block where the search starts
        head = self.get_random_empty_block()
        self.snake.body.append(head)

        while len(self.snake.body) < self.snake_size:
            # get all possible directions of block
            valid_directions = self.get_valid_direction_for_block(head)

            # choose random direction for new snake piece position
            random_direction = random.choice(valid_directions)

            # get block in chosen direction
            new_block = [head[0] + random_direction.value[0], head[1] + random_direction.value[1]]

            self.snake.body.append(new_block)
            head = new_block

        head = self.snake.body[0]
        first_segment = self.snake.body[1]
        for direction in MAIN_DIRECTIONS:
            block_in_direction = [first_segment[0] + direction.value[0], first_segment[1] + direction.value[1]]
            if block_in_direction[0] == head[0] and block_in_direction[1] == head[1]:
                self.snake.past_direction = direction
                break

    def update_board_from_snake(self) -> None:
        snake_body_array = np.array(self.snake.body, dtype=np.int32)
        self.board = cvision.update_board_from_snake(self.board, snake_body_array)

    def move(self, new_direction: Direction) -> bool:
        self.snake.past_direction = new_direction

        head = self.snake.body[0]
        next_head = [head[0] + new_direction.value[0], head[1] + new_direction.value[1]]
        new_head_value = self.board[next_head[0]][next_head[1]]

        if (new_head_value == BoardConsts.WALL) or (new_head_value == BoardConsts.SNAKE_BODY):
            return False

        self.snake.body.insert(0, next_head)
        self.snake.steps_taken += 1

        # if snake eats an apple, the last segment isn't removed from the body list when moving
        if new_head_value == BoardConsts.APPLE:
            self.snake.steps_to_apple = 0
            self.update_board_from_snake()
            self.snake.score = self.snake.score + 1

            if self.max_score == self.snake.score:
                self.snake.won = True
                return False

            self.place_new_apple()
            self.snake.TTL = GameSettings.SNAKE_MAX_TTL

        else:
            self.snake.body = self.snake.body[:-1]
            self.snake.steps_to_apple += 1
            self.update_board_from_snake()
            self.snake.TTL = self.snake.TTL - 1

            if self.snake.TTL == 0:
                return False

        return True

    @staticmethod
    def get_nn_output_4directions(nn_output) -> Direction:
        direction_index = np.argmax(nn_output)

        if direction_index == 0:
            return Direction.UP
        elif direction_index == 1:
            return Direction.DOWN
        elif direction_index == 2:
            return Direction.LEFT
        else:
            return Direction.RIGHT
