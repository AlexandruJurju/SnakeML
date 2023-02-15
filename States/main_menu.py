import pygame
import pygame_gui
from pygame_gui import UIManager
from pygame_gui.elements import UILabel, UIButton

from States.base_state import BaseState
from States.state_manager import StateManager


class MainMenu(BaseState):
    def __init__(self, state_manager: StateManager, ui_manager: UIManager):
        super().__init__("main_menu", "quit", state_manager)

        self.ui_manager = ui_manager

        self.title_label = None
        self.button_backpropagation_options = None
        self.button_genetic_options = None
        self.button_quit = None

    def start(self):
        self.title_label = UILabel(pygame.Rect((87, 40), (850, 180)), "Main Menu", self.ui_manager, object_id="#main_menu_title")
        self.button_backpropagation_options = UIButton(pygame.Rect((437, 200), (150, 35)), "Option Back", self.ui_manager)
        self.button_genetic_options = UIButton(pygame.Rect((437, 250), (150, 35)), "Option Genetic", self.ui_manager)
        self.button_quit = UIButton(pygame.Rect((437, 300), (150, 35)), "QUIT", self.ui_manager)

    def end(self):
        self.title_label.kill()
        self.button_backpropagation_options.kill()
        self.button_genetic_options.kill()
        self.button_quit.kill()

    def run(self, surface: pygame.Surface, time_delta):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.set_target_state_name("quit")
                self.trigger_transition()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.set_target_state_name("quit")
                    self.trigger_transition()

            self.ui_manager.process_events(event)

            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.button_backpropagation_options:
                    self.set_target_state_name("options_backpropagation")
                    self.trigger_transition()
                if event.ui_element == self.button_genetic_options:
                    self.set_target_state_name("options_genetic")
                    self.trigger_transition()
                if event.ui_element == self.button_quit:
                    self.set_target_state_name("quit")
                    self.trigger_transition()

        # blit background ?

        self.ui_manager.update(time_delta)

        self.ui_manager.draw_ui(surface)
