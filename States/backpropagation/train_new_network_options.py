import pygame
import pygame_gui
from pygame_gui import UIManager
from pygame_gui.elements import UILabel, UIButton, UITextEntryLine, UIDropDownMenu

from States.base_state import BaseState
from States.state_manager import StateManager
from constants import State
from settings import SnakeSettings, BoardSettings, NNSettings


# TODO Dynamic Direction
class BackpropagationTrainNewNetworkOptions(BaseState):
    def __init__(self, state_manager: StateManager, ui_manager: UIManager):
        super().__init__(State.BACKPROPAGATION_TRAIN_NEW_NETWORK_OPTIONS, state_manager)

        self.ui_manager = ui_manager

        self.title_label = None
        self.button_back = None

        self.dropdown_input_direction_count = None
        self.dropdown_input_direction_count_label = None

        self.dropdown_vision_line_return_type = None
        self.dropdown_vision_line_return_type_label = None

        self.board_size_entry = None
        self.board_size_entry_label = None

        self.starting_snake_size_entry = None
        self.starting_snake_size_entry_label = None

        self.button_run = None

    def start(self):
        self.title_label = UILabel(pygame.Rect((87, 40), (800, 25)), "Training Genetic Network", self.ui_manager, object_id="#window_label")
        self.button_back = UIButton(pygame.Rect((25, 725), (125, 35)), "BACK", self.ui_manager)

        self.dropdown_input_direction_count = UIDropDownMenu(NNSettings.AVAILABLE_INPUT_DIRECTIONS, NNSettings.AVAILABLE_INPUT_DIRECTIONS[0], pygame.Rect((200, 350), (75, 30)), self.ui_manager)
        self.dropdown_input_direction_count_label = UILabel(pygame.Rect((115, 300), (250, 35)), "Input Direction Count", self.ui_manager)

        self.dropdown_vision_line_return_type = UIDropDownMenu(NNSettings.AVAILABLE_VISION_LINES_RETURN_TYPE, NNSettings.AVAILABLE_VISION_LINES_RETURN_TYPE[0], pygame.Rect((650, 350), (100, 30)), self.ui_manager)
        self.dropdown_vision_line_return_type_label = UILabel(pygame.Rect((565, 300), (250, 35)), "Vision Line Return Type", self.ui_manager)

        self.starting_snake_size_entry = UITextEntryLine(pygame.Rect((200, 150), (75, 30)), self.ui_manager)
        self.starting_snake_size_entry_label = UILabel(pygame.Rect((115, 100), (250, 35)), "Starting Snake Size", self.ui_manager)
        self.starting_snake_size_entry.set_text(str(SnakeSettings.START_SNAKE_SIZE))

        self.board_size_entry = UITextEntryLine(pygame.Rect((650, 150), (75, 30)), self.ui_manager)
        self.board_size_entry_label = UILabel(pygame.Rect((565, 100), (250, 35)), "Board Size", self.ui_manager)
        self.board_size_entry.set_text(str(BoardSettings.BOARD_SIZE))

        self.button_run = UIButton(pygame.Rect((450, 700), (75, 35)), "RUN", self.ui_manager)

    def end(self):
        self.title_label.kill()
        self.button_back.kill()

        self.dropdown_input_direction_count.kill()
        self.dropdown_input_direction_count_label.kill()

        self.dropdown_vision_line_return_type.kill()
        self.dropdown_vision_line_return_type_label.kill()

        self.starting_snake_size_entry.kill()
        self.starting_snake_size_entry_label.kill()

        self.board_size_entry.kill()
        self.board_size_entry_label.kill()

        self.button_run.kill()

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
                    self.set_target_state_name(State.BACKPROPAGATION_MENU)
                    self.trigger_transition()
                if event.ui_element == self.button_run:
                    self.set_target_state_name(State.BACKPROPAGATION_TRAIN_NEW_NETWORK)
                    self.trigger_transition()

        self.ui_manager.update(time_delta)

        self.ui_manager.draw_ui(surface)
