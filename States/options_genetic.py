from pygame_gui import UIManager

from States.base_state import BaseState
from States.state_manager import StateManager


class MenuGenetic(BaseState):
    def __init__(self, state_manager: StateManager, ui_manager: UIManager):
        super().__init__("menu_genetic", "quit", state_manager)

        self.ui_manager = ui_manager

        self.title_label = None
        self.button_run_genetic = None
        self.button_run_best_snake = None
        self.button_back = None

    def start(self):
        pass

    def end(self):
        pass

    def run(self, surface, time_delta):
        pass
