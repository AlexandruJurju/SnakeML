import random
from typing import Tuple

from neural_network import NeuralNetwork
from settings import SnakeSettings
from vision import *


class Individual:
    def __init__(self, neural_network: NeuralNetwork):
        self.score = 0
        self.fitness = 0
        self.brain = neural_network

    def calculate_fitness(self):
        pass


class Snake(Individual):
    def __init__(self, neural_network: NeuralNetwork, starting_direction: None):
        super().__init__(neural_network)
        self.body = []
        self.ttl = SnakeSettings.SNAKE_MAX_TTL
        self.steps_taken = 0
        self.won = False

        # TODO BAD direction , can cause collisions
        if starting_direction is None:
            self.direction = random.choice(MAIN_DIRECTIONS)

    def calculate_fitness(self) -> None:
        win_bonus = 10 ^ 15 if self.won else 1
        self.fitness = (self.steps_taken + ((2 ** self.score) + (self.score ** 2) * 500)) - (((.25 * self.steps_taken) ** 1.3) * (self.score ** 1.2))


class Model:
    def __init__(self, model_size: int, snake_size: int, start_random: bool, net: NeuralNetwork):
        self.size = model_size + 2
        self.board = [[BoardConsts.EMPTY for _ in range(self.size)] for _ in range(self.size)]

        self.snake_size = snake_size
        self.snake = Snake(net, None)

        self.make_board()
        if start_random:
            self.place_new_apple()
            self.create_random_snake()
        else:
            self.place_apple_at_coords([5, 5])
            self.place_snake_in_given_position([[10, 1], [9, 1], [8, 1]], Direction.DOWN)
        self.update_board_from_snake()

    def make_board(self) -> None:
        for i in range(0, self.size):
            for j in range(0, self.size):
                # place walls on the borders and nothing inside
                if i == 0 or i == self.size - 1 or j == 0 or j == self.size - 1:
                    self.board[i][j] = BoardConsts.WALL
                else:
                    self.board[i][j] = BoardConsts.EMPTY

    def place_apple_at_coords(self, position) -> None:
        self.board[position[0]][position[1]] = BoardConsts.APPLE

    def place_snake_in_given_position(self, positions: [], direction: Direction) -> None:
        for i, position in enumerate(positions):
            if i == 0:
                self.board[position[0]][position[1]] = BoardConsts.SNAKE_HEAD
            else:
                self.board[position[0]][position[1]] = BoardConsts.SNAKE_BODY
            self.snake.body.append([position[0], position[1]])
        self.snake.direction = direction

    def find_snake_head(self) -> []:
        for i in range(0, self.size):
            for j in range(0, self.size):
                if self.board[i][j] == BoardConsts.SNAKE_HEAD:
                    return [i, j]

    def get_random_empty_block(self) -> []:
        empty = []
        for i in range(1, self.size):
            for j in range(1, self.size):
                if self.board[i][j] == BoardConsts.EMPTY:
                    empty.append([i, j])

        return random.choice(empty)

    def place_new_apple(self) -> None:
        rand_block = self.get_random_empty_block()
        self.board[rand_block[0]][rand_block[1]] = BoardConsts.APPLE

    def get_valid_direction_for_block(self, block: Tuple) -> List[Direction]:
        valid_directions = []

        # check all main direction that the block has
        for direction in MAIN_DIRECTIONS:
            new_block = [direction.value[0] + block[0], direction.value[1] + block[1]]

            #  if it's not a wall or a snake part then it's a valid direction
            if (self.board[new_block[0]][new_block[1]] != BoardConsts.WALL) and (self.board[new_block[0]][new_block[1]] != BoardConsts.SNAKE_BODY) and (new_block not in self.snake.body):
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

        self.snake.direction = random.choice(self.get_valid_direction_for_block(self.snake.body[0]))

    def update_board_from_snake(self) -> None:
        # remove previous snake position on board
        self.clear_snake_on_board()

        # loop all snake pieces and put S on board using their coordinates
        for piece in self.snake.body:
            if piece == self.snake.body[0]:
                self.board[piece[0]][piece[1]] = BoardConsts.SNAKE_HEAD
            else:
                self.board[piece[0]][piece[1]] = BoardConsts.SNAKE_BODY

    def clear_snake_on_board(self) -> None:
        for i in range(1, self.size):
            for j in range(1, self.size):
                if self.board[i][j] == BoardConsts.SNAKE_BODY or self.board[i][j] == BoardConsts.SNAKE_HEAD:
                    self.board[i][j] = BoardConsts.EMPTY

    def move_in_direction(self, new_direction: Direction) -> bool:
        self.snake.direction = new_direction

        head = self.snake.body[0]
        next_head = [head[0] + new_direction.value[0], head[1] + new_direction.value[1]]

        new_head_value = self.board[next_head[0]][next_head[1]]
        if (new_head_value == BoardConsts.WALL) or (new_head_value == BoardConsts.SNAKE_BODY):
            return False

        self.snake.body.insert(0, next_head)
        self.snake.steps_taken += 1

        if new_head_value == BoardConsts.APPLE:
            self.update_board_from_snake()
            self.place_new_apple()
            self.snake.ttl = SnakeSettings.SNAKE_MAX_TTL
            self.snake.score = self.snake.score + 1
        else:
            self.snake.body = self.snake.body[:-1]
            self.update_board_from_snake()
            self.snake.ttl = self.snake.ttl - 1

        if self.snake.ttl == 0:
            return False

        # TODO replace with win condition
        if self.check_win_condition():
            self.snake.won = True
            return False

        return True

    def check_win_condition(self):
        for i in range(1, self.size):
            for j in range(1, self.size):
                if self.board[i][j] == BoardConsts.EMPTY:
                    return False
        return True

    def get_nn_output(self, vision_lines) -> np.ndarray:
        nn_input = get_parameters_in_nn_input_form(vision_lines, self.snake.direction)
        output = self.snake.brain.feed_forward(nn_input)
        return output

    @staticmethod
    def get_nn_output_4directions(nn_output) -> Direction:
        direction_index = list(nn_output).index(max(list(nn_output)))

        match direction_index:
            case 0:
                return Direction.UP
            case 1:
                return Direction.DOWN
            case 2:
                return Direction.LEFT
            case 3:
                return Direction.RIGHT

    def get_nn_output_3directions_dynamic(self, nn_output) -> Direction:
        direction_index = list(nn_output).index(max(list(nn_output)))

        # STRAIGHT
        if direction_index == 0:
            return self.snake.direction

        # LEFT
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
        # RIGHT
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
