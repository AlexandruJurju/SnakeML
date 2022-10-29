import pygame
from constants import *
from model import *


class Game:
    def __init__(self, model: Model):
        pygame.init()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.fps_clock = pygame.time.Clock()

        self.running = True
        self.model = model

    def __draw_board(self):
        # use y and x for index instead of x and y
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

                head = self.model.snake.body[0]
                pygame.draw.rect(self.window, COLOR_SNAKE_HEAD,
                                 pygame.Rect(head[1] * SQUARE_SIZE + OFFSET_BOARD_X,
                                             head[0] * SQUARE_SIZE + OFFSET_BOARD_Y,
                                             SQUARE_SIZE, SQUARE_SIZE))

                # draw lines between squares
                pygame.draw.rect(self.window, COLOR_SQUARE_DELIMITER, pygame.Rect(x_position, y_position, SQUARE_SIZE, SQUARE_SIZE), width=1)

    def __manage_key_inputs(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                match event.key:
                    case pygame.K_ESCAPE:
                        self.running = False

    def run(self) -> None:
        while self.running:
            self.window.fill(COLOR_BLACK)

            self.__draw_board()
            self.__manage_key_inputs()
            self.running = self.model.move_random_direction()

            pygame.display.update()
            self.fps_clock.tick(MAX_FPS)
