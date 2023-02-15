import pygame
import pygame_gui
from pygame_gui import UIManager
from pygame_gui.elements import UILabel, UIButton

from States.base_state import BaseState
from States.state_manager import StateManager
from constants import State


class GeneticTrainNetwork(BaseState):
    def __init__(self, state_manager: StateManager, ui_manager: UIManager):
        super().__init__(State.GENETIC_TRAIN_NETWORK, state_manager)

        self.ui_manager = ui_manager

        self.title_label = None
        self.button_back = None

    def start(self):
        self.title_label = UILabel(pygame.Rect((87, 40), (800, 25)), "Training Genetic Network", self.ui_manager, object_id="#window_label")
        self.button_back = UIButton(pygame.Rect((25, 725), (125, 35)), "BACK", self.ui_manager)

    def end(self):
        self.title_label.kill()
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
                if event.ui_element == self.button_back:
                    self.set_target_state_name(State.GENETIC_MENU)
                    self.trigger_transition()

        surface.fill(self.ui_manager.ui_theme.get_colour("dark_bg"))

        self.ui_manager.update(time_delta)

        self.ui_manager.draw_ui(surface)
