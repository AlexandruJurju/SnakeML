import pygame
import pygame_gui
from pygame_gui import UIManager
from pygame_gui.elements import UILabel, UIButton

from States.base_state import BaseState
from States.state_manager import StateManager
from game_config import State, ViewSettings


class GeneticMenu(BaseState):
    def __init__(self, ui_manager: UIManager):
        super().__init__(State.GENETIC_MENU)

        self.ui_manager = ui_manager

        self.title_label = None
        self.button_options_genetic = None
        self.button_run_pretrained_network = None
        self.button_back = None

    def start(self):
        self.title_label = UILabel(pygame.Rect(ViewSettings.TITLE_LABEL_POSITION, ViewSettings.TITLE_LABEL_DIMENSION), "Genetic Menu", self.ui_manager, object_id="#window_label")
        self.button_back = UIButton(pygame.Rect(ViewSettings.BUTTON_BACK_POSITION, ViewSettings.BUTTON_BACK_DIMENSION), "BACK", self.ui_manager)

        self.button_options_genetic = UIButton(pygame.Rect(ViewSettings.OPTIONS_BUTTON_POSITION, ViewSettings.OPTIONS_BUTTON_DIMENSIONS), "Train New Network", self.ui_manager)
        self.button_run_pretrained_network = UIButton(pygame.Rect(ViewSettings.PRETRAINED_BUTTON_POSITION, ViewSettings.PRETRAINED_BUTTON_DIMENSIONS), "Run Pretrained Networks", self.ui_manager)

    def end(self):
        self.title_label.kill()
        self.button_options_genetic.kill()
        self.button_run_pretrained_network.kill()
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
                if event.ui_element == self.button_run_pretrained_network:
                    self.set_target_state_name(State.RUN_PRETRAINED)
                    self.data_to_send = {
                        "state": "genetic"
                    }
                    self.trigger_transition()
                if event.ui_element == self.button_options_genetic:
                    self.set_target_state_name(State.OPTIONS)
                    self.data_to_send = {
                        "state": "genetic"
                    }
                    self.trigger_transition()
                if event.ui_element == self.button_back:
                    self.set_target_state_name(State.MAIN_MENU)
                    self.trigger_transition()
        surface.fill(self.ui_manager.ui_theme.get_colour("dark_bg"))

        self.ui_manager.update(time_delta)

        self.ui_manager.draw_ui(surface)
