import pygame

from Neural.neural_network import NeuralNetwork
from constants import *
from model import *


class Game:
    def __init__(self, model_size: int, snake_size: int, net: NeuralNetwork):
        pygame.init()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.fps_clock = pygame.time.Clock()

        self.running = True
        self.model = Model(model_size, snake_size, net)

    def draw_board(self):
        # use y,x for index in board instead of x,y because of changed logic
        # x is line y is column ; drawing x is column and y is line
        for x in range(self.model.size):
            for y in range(self.model.size):
                x_position = x * SQUARE_SIZE + OFFSET_BOARD_X
                y_position = y * SQUARE_SIZE + OFFSET_BOARD_Y

                match self.model.board[y, x]:
                    case "S":
                        pygame.draw.rect(self.window, COLOR_SNAKE, pygame.Rect(x_position, y_position, SQUARE_SIZE, SQUARE_SIZE))
                    case "W":
                        pygame.draw.rect(self.window, COLOR_WHITE, pygame.Rect(x_position, y_position, SQUARE_SIZE, SQUARE_SIZE))
                    case "A":
                        pygame.draw.rect(self.window, COLOR_APPLE, pygame.Rect(x_position, y_position, SQUARE_SIZE, SQUARE_SIZE))
                    case "H":
                        pygame.draw.rect(self.window, COLOR_SNAKE_HEAD, pygame.Rect(x_position, y_position, SQUARE_SIZE, SQUARE_SIZE))
                # draw lines between squares
                pygame.draw.rect(self.window, COLOR_SQUARE_DELIMITER, pygame.Rect(x_position, y_position, SQUARE_SIZE, SQUARE_SIZE), width=1)

    def run(self):
        self.draw_board()
        pygame.display.update()
        self.fps_clock.tick(MAX_FPS)

        print(self.model.board)
        print(Vision.get_parameters_in_nn_input_form(self.model.board, VISION_LINES_COUNT, VISION_LINES_RETURN))

        while self.running:
            self.window.fill(COLOR_BACKGROUND)

            next_direction = self.model.get_neural_network_direction_output_3(VISION_LINES_COUNT, VISION_LINES_RETURN)
            self.running = self.model.move_in_direction(next_direction)

            print(self.model.board)
            print(Vision.get_parameters_in_nn_input_form(self.model.board, VISION_LINES_COUNT, VISION_LINES_RETURN))

            if self.running:
                self.draw_board()
                # self.__draw_vision_lines()
                # self.__draw_network()
            else:
                # self.running = True
                # self.model.reinit_model()
                pass

            pygame.display.update()
            self.fps_clock.tick(MAX_FPS)
