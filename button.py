from typing import Tuple

import pygame

from settings import *


class Button:
    def __init__(self, position: Tuple, width: int, height: int, text: str, font: pygame.font, text_color, rectangle_color: Tuple):
        self.position = position

        # used for solving a bug, where the button is pressed twice for one click, waits for mouse click release before allowing clicking again
        self.clicked = False

        self.button_rectangle = pygame.Rect(position, (width, height))
        self.button_rectangle_color = rectangle_color

        # text is a surface, I use text rectangle to center it
        self.text_surface = font.render(text, True, text_color)
        self.text_rectangle = self.text_surface.get_rect(center=self.button_rectangle.center)

    def draw(self, surface: pygame.surface) -> bool:
        self.on_hover()
        pygame.draw.rect(surface, self.button_rectangle_color, self.button_rectangle)
        surface.blit(self.text_surface, self.text_rectangle)

        # only call it once, button is redraw multiple times during execution
        return self.check_clicked()

    def on_hover(self):
        if self.button_rectangle.collidepoint(pygame.mouse.get_pos()):
            self.button_rectangle_color = ViewVars.COLOR_RED

    def check_clicked(self) -> bool:
        action = False
        mouse_position = pygame.mouse.get_pos()

        if self.button_rectangle.collidepoint(mouse_position):
            if pygame.mouse.get_pressed()[0] and self.clicked == False:
                self.clicked = True
                action = True
            if pygame.mouse.get_pressed()[0] == 0:
                self.clicked = False
        return action

# TODO add view for board training examples
# TODO add dropdown for options
# TODO add buttons
# TODO add highscore
# TODO add dropdown for board size
