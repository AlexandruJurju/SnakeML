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
                self.draw_vision_lines(Vision.get_vision_lines(self.model.board, VISION_LINES_COUNT, VISION_LINES_RETURN))
                # self.__draw_network()
            else:
                # self.running = True
                # self.model.reinit_model()
                pass

            pygame.display.update()
            self.fps_clock.tick(MAX_FPS)

    def draw_vision_lines(self, vision_lines):
        font = pygame.font.SysFont("arial", 18)

        # loop over all lines in given vision lines
        for line in vision_lines:
            line_label = font.render(line, True, COLOR_BLACK)

            # render vision line text at wall position
            self.window.blit(line_label, [vision_lines[line].wall_coord[1] * SQUARE_SIZE + OFFSET_BOARD_X, vision_lines[line].wall_coord[0] * SQUARE_SIZE + OFFSET_BOARD_Y])

            # draw line from head to wall, draw before body and apple lines
            # drawing uses SQUARE_SIZE//2 so that lines go through the middle of the squares
            line_end_x = self.model.snake.body[0][1] * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_X
            line_end_y = self.model.snake.body[0][0] * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_Y

            # draw line form snake head until wall block
            self.__draw_vision_line(COLOR_APPLE, 1, vision_lines[line].wall_coord[1], vision_lines[line].wall_coord[0], line_end_x, line_end_y)

            # draw another line from snake head to first segment found
            if vision_lines[line].segment_coord is not None:
                self.__draw_vision_line(COLOR_RED, 5, vision_lines[line].segment_coord[1], vision_lines[line].segment_coord[0], line_end_x, line_end_y)

            # draw another line from snake to apple if apple is found
            if vision_lines[line].apple_coord is not None:
                self.__draw_vision_line(COLOR_GREEN, 5, vision_lines[line].apple_coord[1], vision_lines[line].apple_coord[0], line_end_x, line_end_y)

    def __draw_vision_line(self, color, width, line_coord_1, line_coord_0, line_end_x, line_end_y):
        pygame.draw.line(self.window, color,
                         (line_coord_1 * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_X,
                          line_coord_0 * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_Y),
                         (line_end_x, line_end_y), width=width)
