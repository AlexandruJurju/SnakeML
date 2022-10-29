import pygame
from constants import *
from model import *


class Display:
    def __init__(self, model: Model):
        pygame.init()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.fps_clock = pygame.time.Clock()

        self.running = True
        self.model = model

    def __draw_board(self):

        # use y and x for matrix index because
        for x in range(self.model.size):
            for y in range(self.model.size):
                match self.model.board[y, x]:
                    case "S":
                        pygame.draw.rect(self.window, COLOR_SNAKE, pygame.Rect(x * SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                    case "W":
                        pygame.draw.rect(self.window, COLOR_WHITE, pygame.Rect(x * SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                    case "A":
                        pygame.draw.rect(self.window, COLOR_APPLE, pygame.Rect(x * SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

                pygame.draw.rect(self.window, (15, 15, 15), pygame.Rect(x * SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), width=1)

    def run(self) -> None:
        while self.running:
            self.window.fill(COLOR_BLACK)

            self.__draw_board()
            pygame.display.update()
            self.fps_clock.tick(MAX_FPS)
