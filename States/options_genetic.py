from pygame_gui import UIManager

from States.base_state import BaseState
from States.state_manager import StateManager


class OptionsGenetic(BaseState):
    def __init__(self, state_manager: StateManager, ui_manager: UIManager):
        super().__init__("options_genetic", "quit", state_manager)

        self.ui_manager = ui_manager

        self.button_run_genetic = None
        self.button_run_best_snake = None
        self.button_back = None
