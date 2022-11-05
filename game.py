import pygame

from constants import *
from model import *


class Game:
    def __init__(self, model_size: int, snake_size: int):
        pygame.init()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.fps_clock = pygame.time.Clock()

        net = KerasNetwork()
        net.add(Dense(28, 16))
        net.add(Activation(relu, relu))
        net.add(Dense(16, 4))
        net.add(Activation(softmax, softmax))

        self.running = True
        self.model = Model(model_size, snake_size, net)
        self.direction = Direction.UP

    def __draw_board(self):
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

                # draw snake head using another color
                head = self.model.snake.body[0]
                pygame.draw.rect(self.window, COLOR_SNAKE_HEAD,
                                 pygame.Rect(head[1] * SQUARE_SIZE + OFFSET_BOARD_X, head[0] * SQUARE_SIZE + OFFSET_BOARD_Y, SQUARE_SIZE, SQUARE_SIZE))

                # draw lines between squares
                pygame.draw.rect(self.window, COLOR_SQUARE_DELIMITER, pygame.Rect(x_position, y_position, SQUARE_SIZE, SQUARE_SIZE), width=1)

    def __draw_vision_lines(self):
        vision_lines = self.model.get_vision_lines(8, "boolean")
        font = pygame.font.SysFont("arial", 18)

        # loop over all lines in given vision lines
        for line in vision_lines:
            line_label = font.render(line, True, COLOR_BLACK)

            # render vision line text at wall position
            self.window.blit(line_label, [vision_lines[line].wall_coord[1] * SQUARE_SIZE + OFFSET_BOARD_X,
                                          vision_lines[line].wall_coord[0] * SQUARE_SIZE + OFFSET_BOARD_Y])

            # draw line from head to wall, draw before body and apple lines
            # drawing uses SQUARE_SIZE//2 so that lines go through the middle of the squares
            line_end_x = self.model.snake.body[0][1] * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_X
            line_end_y = self.model.snake.body[0][0] * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_Y

            # draw line form snake head until wall block
            pygame.draw.line(self.window, COLOR_APPLE,
                             ((vision_lines[line].wall_coord[1]) * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_X,
                              (vision_lines[line].wall_coord[0]) * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_Y),
                             (line_end_x, line_end_y), width=1)

            # draw another line from snake head to first segment found
            if vision_lines[line].segment_coord is not None:
                pygame.draw.line(self.window, COLOR_RED,
                                 (vision_lines[line].segment_coord[1] * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_X,
                                  vision_lines[line].segment_coord[0] * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_Y),
                                 (line_end_x, line_end_y), width=5)

            # draw another line from snake to apple if apple is found
            if vision_lines[line].apple_coord is not None:
                pygame.draw.line(self.window, COLOR_GREEN,
                                 (vision_lines[line].apple_coord[1] * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_X,
                                  vision_lines[line].apple_coord[0] * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_Y),
                                 (line_end_x, line_end_y), width=5)

    def __manage_key_inputs(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                match event.key:
                    case pygame.K_ESCAPE:
                        self.running = False
                    case pygame.K_UP:
                        self.direction = Direction.UP
                    case pygame.K_DOWN:
                        self.direction = Direction.DOWN
                    case pygame.K_LEFT:
                        self.direction = Direction.LEFT
                    case pygame.K_RIGHT:
                        self.direction = Direction.RIGHT

    def run(self) -> None:
        while self.running:
            self.__manage_key_inputs()
            self.window.fill(COLOR_BACKGROUND)

            next_direction = self.model.get_direction_from_nn_output()
            self.running = self.model.move_in_direction(next_direction)

            if self.running:
                self.__draw_board()
                self.__draw_vision_lines()

                pygame.display.update()
                self.fps_clock.tick(MAX_FPS)
