import pygame
import pygame_gui
from pygame_gui import UIManager
from pygame_gui.elements import UILabel, UIButton

from States.base_state import BaseState
from States.state_manager import StateManager
from constants import State


class MainMenu(BaseState):
    def __init__(self, state_manager: StateManager, ui_manager: UIManager):
        super().__init__(State.MAIN_MENU, state_manager)

        self.ui_manager = ui_manager

        self.title_label = None
        self.button_backpropagation_menu = None
        self.button_genetic_menu = None
        self.button_quit = None

    def start(self):
        self.title_label = UILabel(pygame.Rect((87, 40), (850, 180)), "Main Menu", self.ui_manager, object_id="#window_label")
        self.button_backpropagation_menu = UIButton(pygame.Rect((437, 200), (150, 35)), "Option Backpropagation", self.ui_manager)
        self.button_genetic_menu = UIButton(pygame.Rect((437, 250), (150, 35)), "Option Genetic", self.ui_manager)
        self.button_quit = UIButton(pygame.Rect((437, 300), (150, 35)), "QUIT", self.ui_manager)

    def end(self):
        self.title_label.kill()
        self.button_backpropagation_menu.kill()
        self.button_genetic_menu.kill()
        self.button_quit.kill()

    def run(self, surface: pygame.Surface, time_delta):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.set_target_state_name(State.QUIT)
                self.trigger_transition()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.set_target_state_name(State.QUIT)
                    self.trigger_transition()

            self.ui_manager.process_events(event)

            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.button_backpropagation_menu:
                    self.set_target_state_name(State.BACKPROPAGATION_MENU)
                    self.trigger_transition()
                if event.ui_element == self.button_genetic_menu:
                    self.set_target_state_name(State.GENETIC_MENU)
                    self.trigger_transition()
                if event.ui_element == self.button_quit:
                    self.set_target_state_name(State.QUIT)
                    self.trigger_transition()

        surface.fill(self.ui_manager.ui_theme.get_colour("dark_bg"))

        self.ui_manager.update(time_delta)

        self.ui_manager.draw_ui(surface)
