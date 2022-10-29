import pygame
from constants import *


class Display:
    def __init__(self):
        pygame.init()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.fps_clock = pygame.time.Clock()

        self.running = True

    def run(self) -> None:
        while self.running:
            pass
