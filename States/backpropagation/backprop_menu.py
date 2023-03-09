import pygame
import pygame_gui
from pygame_gui import UIManager
from pygame_gui.elements import UILabel, UIButton

from States.base_state import BaseState
from States.state_manager import StateManager
from settings import State, ViewConsts


class BackpropMenu(BaseState):
    def __init__(self, state_manager: StateManager, ui_manager: UIManager):
        super().__init__(State.BACKPROPAGATION_MENU, state_manager)

        self.ui_manager = ui_manager

        self.title_label = None
        self.button_back = None

        self.button_run_pretrained_network = None
        self.button_train_network_options = None

    def start(self):
        self.title_label = UILabel(pygame.Rect(ViewConsts.TITLE_LABEL_POSITION, ViewConsts.TITLE_LABEL_DIMENSION), "Backpropagation Menu", self.ui_manager, object_id="#window_label")
        self.button_back = UIButton(pygame.Rect(ViewConsts.BUTTON_BACK_POSITION, ViewConsts.BUTTON_BACK_DIMENSION), "BACK", self.ui_manager)

        self.button_train_network_options = UIButton(pygame.Rect(ViewConsts.OPTIONS_BUTTON_POSITION, ViewConsts.OPTIONS_BUTTON_DIMENSIONS), "Train New Network", self.ui_manager)
        self.button_run_pretrained_network = UIButton(pygame.Rect(ViewConsts.PRETRAINED_BUTTON_POSITION, ViewConsts.PRETRAINED_BUTTON_DIMENSIONS), "Run Pretrained Network", self.ui_manager)

    def end(self):
        self.title_label.kill()
        self.button_back.kill()

        self.button_run_pretrained_network.kill()
        self.button_train_network_options.kill()

    def run(self, surface, time_delta):
        surface.fill(self.ui_manager.ui_theme.get_colour("dark_bg"))

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
                if event.ui_element == self.button_back:
                    self.set_target_state_name(State.MAIN_MENU)
                    self.trigger_transition()
                if event.ui_element == self.button_run_pretrained_network:
                    self.set_target_state_name(State.BACKPROPAGATION_TRAINED_NETWORK)
                    self.trigger_transition()
                if event.ui_element == self.button_train_network_options:
                    self.set_target_state_name(State.BACKPROPAGATION_TRAIN_NEW_NETWORK_OPTIONS)
                    self.trigger_transition()

        self.ui_manager.update(time_delta)

        self.ui_manager.draw_ui(surface)
