import random
from typing import Tuple

from neural_network import NeuralNetwork
from vision import *


class Individual:
    def __init__(self, neural_network: NeuralNetwork):
        self.score: int = 0
        self.fitness: float = 0
        self.brain: NeuralNetwork = neural_network

    def calculate_fitness(self):
        pass


class Snake(Individual):
    def __init__(self, neural_network: NeuralNetwork):
        super().__init__(neural_network)
        self.body: List = []
        self.TTL: int = GameSettings.SNAKE_MAX_TTL
        self.steps_taken: int = 0
        self.won: bool = False
        self.direction: Direction = None

    def calculate_fitness(self) -> None:
        fitness_score = self.steps_taken + ((2 ** self.score) + (self.score ** 2.1) * 500) - (((.25 * self.steps_taken) ** 1.3) * (self.score ** 1.2))

        # over = self.steps_taken - (self.steps_taken / (self.score + 1))
        # under = self.steps_taken + (self.steps_taken / (self.score + 1))
        # fitness_score = self.score + 0.5 + (0.5 * (over / under))
        # fitness_score = fitness_score * 10000

        self.fitness = fitness_score


class Model:
    def __init__(self, model_size: int, snake_size: int, start_random: bool, net: NeuralNetwork):
        self.size: int = model_size + 2
        self.board: np.ndarray = np.full((self.size, self.size), BoardConsts.EMPTY)

        self.snake_size: int = snake_size
        self.snake: Snake = Snake(net)

        self.max_score = model_size ** 2 - snake_size
        self.make_board()

        if start_random:
            self.place_new_apple()
            self.create_random_snake()
        else:
            self.place_apple_at_coords([5, 5])
            self.place_snake_at_given_position([[10, 1], [9, 1], [8, 1]], Direction.DOWN)
        self.update_board_from_snake()

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
        self.snake.direction = direction

    def get_random_empty_block(self) -> []:
        empty = [[i, j] for i in range(1, self.size) for j in range(1, self.size) if self.board[i][j] == BoardConsts.EMPTY]
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
        head_i, head_j = self.snake.body[0]
        self.board[head_i][head_j] = BoardConsts.SNAKE_HEAD
        body_coords = self.snake.body[1:]
        self.board[[i for i, _ in body_coords], [j for _, j in body_coords]] = BoardConsts.SNAKE_BODY

    def clear_snake_on_board(self) -> None:
        body_mask = self.board == BoardConsts.SNAKE_BODY
        head_mask = self.board == BoardConsts.SNAKE_HEAD
        self.board[body_mask | head_mask] = BoardConsts.EMPTY

    def move(self, new_direction: Direction) -> bool:
        self.snake.direction = new_direction
        self.snake.steps_taken += 1

        head = self.snake.body[0]
        next_head = [head[0] + new_direction.value[0], head[1] + new_direction.value[1]]
        new_head_value = self.board[next_head[0]][next_head[1]]

        if (new_head_value == BoardConsts.WALL) or (new_head_value == BoardConsts.SNAKE_BODY):
            self.snake.hit_obstacle = True
            return False

        self.snake.body.insert(0, next_head)

        # if snake eats an apple, the last segment isn't removed from the body list when moving
        if new_head_value == BoardConsts.APPLE:
            self.update_board_from_snake()
            self.snake.score = self.snake.score + 1

            if self.max_score == self.snake.score:
                self.snake.won = True
                return False

            self.place_new_apple()
            self.snake.TTL = GameSettings.SNAKE_MAX_TTL

        else:
            self.snake.body = self.snake.body[:-1]
            self.update_board_from_snake()
            self.snake.TTL = self.snake.TTL - 1

            if self.snake.TTL == 0:
                return False

        return True

    def check_win_condition(self):
        return not any(cell == BoardConsts.EMPTY for row in self.board for cell in row)

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
