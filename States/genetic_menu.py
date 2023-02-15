import pygame
import pygame_gui
from pygame_gui import UIManager
from pygame_gui.elements import UILabel, UIButton

from States.base_state import BaseState
from States.state_manager import StateManager
from constants import State


class MenuGenetic(BaseState):
    def __init__(self, state_manager: StateManager, ui_manager: UIManager):
        super().__init__(State.GENETIC_MENU, state_manager)

        self.ui_manager = ui_manager

        self.title_label = None
        self.button_options_genetic = None
        self.button_run_trained_network = None
        self.button_back = None

    def start(self):
        self.title_label = UILabel(pygame.Rect((87, 40), (800, 180)), "Genetic Menu", self.ui_manager, object_id="#window_label")
        self.button_options_genetic = UIButton(pygame.Rect((600, 200), (150, 35)), "Options Genetic", self.ui_manager)
        self.button_run_trained_network = UIButton(pygame.Rect((300, 200), (200, 35)), "Run Trained Network", self.ui_manager)
        self.button_back = UIButton(pygame.Rect((437, 300), (150, 35)), "BACK", self.ui_manager)

    def end(self):
        self.title_label.kill()
        self.button_options_genetic.kill()
        self.button_run_trained_network.kill()
        self.button_back.kill()

    def run(self, surface, time_delta):
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
                if event.ui_element == self.button_run_trained_network:
                    self.set_target_state_name(State.GENETIC_RUN_TRAINED_NETWORK)
                    self.trigger_transition()
                if event.ui_element == self.button_options_genetic:
                    self.set_target_state_name(State.GENETIC_MENU)
                    self.trigger_transition()
                if event.ui_element == self.button_back:
                    self.set_target_state_name(State.MAIN_MENU)
                    self.trigger_transition()
        surface.fill(self.ui_manager.ui_theme.get_colour("dark_bg"))

        self.ui_manager.update(time_delta)

        self.ui_manager.draw_ui(surface)
