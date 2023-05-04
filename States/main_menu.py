import pygame
import pygame_gui
from pygame_gui import UIManager
from pygame_gui.elements import UILabel, UIButton

from States.base_state import BaseState
from game_config import State, ViewSettings


class MainMenu(BaseState):
    def __init__(self, ui_manager: UIManager):
        super().__init__(State.MAIN_MENU)

        self.ui_manager = ui_manager

        self.title_label = None
        self.button_backpropagation_menu = None
        self.button_genetic_menu = None
        self.button_quit = None
        self.button_run = None

    def start(self):
        x_positions = {"left-left": ViewSettings.X_CENTER - 500,
                       "left-center": ViewSettings.X_CENTER - 250,
                       "center": ViewSettings.X_CENTER,
                       "right-center": ViewSettings.X_CENTER + 250,
                       "right-right": ViewSettings.X_CENTER + 500}
        y_positions = [ViewSettings.Y_CENTER - 50 - 100, ViewSettings.Y_CENTER - 100, ViewSettings.Y_CENTER + 125 - 100]
        self.title_label = UILabel(pygame.Rect(ViewSettings.TITLE_LABEL_POSITION, ViewSettings.TITLE_LABEL_DIMENSION), "Main Menu", self.ui_manager)
        self.button_quit = UIButton(pygame.Rect(((ViewSettings.WIDTH - 150) // 2, 650), (150, 35)), "QUIT", self.ui_manager)

        self.button_genetic_menu = UIButton(pygame.Rect((x_positions["right-center"] - 350 // 2, y_positions[1]), (350, 35)), "Train using Genetic Algorithm", self.ui_manager)

        self.button_backpropagation_menu = UIButton(pygame.Rect((x_positions["right-center"] - 350 // 2, y_positions[2]), (350, 35)), "Train using Backpropagation", self.ui_manager)

        self.button_run = UIButton(pygame.Rect((x_positions["left-center"] - 250 // 2, y_positions[1]), (250, 35)), "Run Trained Network", self.ui_manager)

    def end(self):
        self.ui_manager.clear_and_reset()

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
                if event.ui_element == self.button_run:
                    self.set_target_state_name(State.RUN_TRAINED)
                    self.trigger_transition()
                if event.ui_element == self.button_backpropagation_menu:
                    self.set_target_state_name(State.OPTIONS)
                    self.trigger_transition()
                    self.data_to_send = {
                        "state": "backpropagation"
                    }
                if event.ui_element == self.button_genetic_menu:
                    self.set_target_state_name(State.OPTIONS)
                    self.trigger_transition()
                    self.data_to_send = {
                        "state": "genetic"
                    }
                if event.ui_element == self.button_quit:
                    self.set_target_state_name(State.QUIT)
                    self.trigger_transition()

        surface.fill(self.ui_manager.ui_theme.get_colour("main_bg"))

        self.ui_manager.update(time_delta)
        self.ui_manager.draw_ui(surface)
