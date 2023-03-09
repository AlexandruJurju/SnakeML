import pygame
import pygame_gui
from pygame_gui import UIManager
from pygame_gui.elements import UILabel, UIButton, UIDropDownMenu, UITextEntryLine

from States.base_state import BaseState
from States.state_manager import StateManager
from game_config import *


# noinspection PyTypeChecker
# TODO options for activation functions
class GeneticTrainNetworkOptions(BaseState):
    def __init__(self, state_manager: StateManager, ui_manager: UIManager):
        super().__init__(State.GENETIC_TRAIN_NETWORK_OPTIONS, state_manager)

        self.ui_manager = ui_manager

        self.title_label = None
        self.button_back = None

        self.dropdown_input_direction_count: UIDropDownMenu = None
        self.dropdown_input_direction_count_label: UILabel = None

        self.dropdown_vision_line_return_type: UIDropDownMenu = None
        self.dropdown_vision_line_return_type_label: UILabel = None

        self.population_count_entry: UITextEntryLine = None
        self.population_count_entry_label: UILabel = None

        self.mutation_rate_entry: UITextEntryLine = None
        self.mutation_rate_entry_label: UILabel = None

        self.board_size_entry: UITextEntryLine = None
        self.board_size_entry_label: UILabel = None

        self.starting_snake_size_entry: UITextEntryLine = None
        self.starting_snake_size_entry_label: UILabel = None

        self.file_name_entry: UITextEntryLine = None
        self.file_name_label: UILabel = None

        self.button_run = None

    def start(self):
        self.title_label = UILabel(pygame.Rect(ViewConsts.TITLE_LABEL_POSITION, ViewConsts.TITLE_LABEL_DIMENSION), "Genetic Network Options", self.ui_manager, object_id="#window_label")
        self.button_back = UIButton(pygame.Rect(ViewConsts.BUTTON_BACK_POSITION, ViewConsts.BUTTON_BACK_DIMENSION), "BACK", self.ui_manager)

        # 0-0
        self.starting_snake_size_entry = UITextEntryLine(pygame.Rect((ViewConsts.X_SECOND - 75 // 2 - 250, 150), (75, 30)), self.ui_manager)
        self.starting_snake_size_entry_label = UILabel(pygame.Rect((ViewConsts.X_SECOND - 250 // 2 - 250, 100), (250, 35)), "Starting Snake Size", self.ui_manager)
        self.starting_snake_size_entry.set_text(str(SnakeSettings.INITIAL_SNAKE_SIZE))

        # 0-1
        self.board_size_entry = UITextEntryLine(pygame.Rect((ViewConsts.X_SECOND - 75 // 2, 150), (75, 30)), self.ui_manager)
        self.board_size_entry_label = UILabel(pygame.Rect((ViewConsts.X_SECOND - 250 // 2, 100), (250, 35)), "Board Size", self.ui_manager)
        self.board_size_entry.set_text(str(BoardSettings.INITIAL_BOARD_SIZE))

        # 1-0
        self.dropdown_input_direction_count = UIDropDownMenu(NNSettings.AVAILABLE_INPUT_DIRECTIONS, NNSettings.AVAILABLE_INPUT_DIRECTIONS[0], pygame.Rect((ViewConsts.X_SECOND - 75 // 2 - 250, 350), (75, 30)), self.ui_manager)
        self.dropdown_input_direction_count_label = UILabel(pygame.Rect((ViewConsts.X_SECOND - 250 // 2 - 250, 300), (250, 35)), "Input Direction Count", self.ui_manager)

        # 1-1
        self.file_name_entry = UITextEntryLine(pygame.Rect((ViewConsts.X_SECOND - 125 // 2, 350), (125, 30)), self.ui_manager)
        self.file_name_label = UILabel(pygame.Rect((ViewConsts.X_SECOND - 125 // 2, 300), (125, 35)), "Network name", self.ui_manager)
        self.file_name_entry.set_text("Default")

        # 1-2
        self.dropdown_vision_line_return_type = UIDropDownMenu(NNSettings.AVAILABLE_VISION_LINES_RETURN_TYPE, NNSettings.AVAILABLE_VISION_LINES_RETURN_TYPE[0],
                                                               pygame.Rect((ViewConsts.X_SECOND - 125 // 2 + 250, 350), (125, 30)), self.ui_manager)
        self.dropdown_vision_line_return_type_label = UILabel(pygame.Rect((ViewConsts.X_SECOND - 250 // 2 + 250, 300), (250, 35)), "Vision Line Return Type", self.ui_manager)

        # 2-0
        self.population_count_entry = UITextEntryLine(pygame.Rect((ViewConsts.X_SECOND - 75 // 2 - 250, 550), (75, 30)), self.ui_manager)
        self.population_count_entry_label = UILabel(pygame.Rect((ViewConsts.X_SECOND - 200 // 2 - 250, 500), (200, 35)), "Individuals in Population", self.ui_manager)
        self.population_count_entry.set_text(str(GeneticSettings.POPULATION_COUNT))

        # 2-1
        self.mutation_rate_entry = UITextEntryLine(pygame.Rect((ViewConsts.X_SECOND - 75 // 2, 550), (75, 30)), self.ui_manager)
        self.mutation_rate_entry_label = UILabel(pygame.Rect((ViewConsts.X_SECOND - 200 // 2, 500), (200, 35)), "Mutation Rate", self.ui_manager)
        self.mutation_rate_entry.set_text(str(GeneticSettings.MUTATION_CHANCE))

        self.button_run = UIButton(pygame.Rect((ViewConsts.X_SECOND - 75 // 2, 650), (75, 35)), "RUN", self.ui_manager)

    def end(self):
        self.title_label.kill()
        self.button_back.kill()

        self.dropdown_input_direction_count.kill()
        self.dropdown_input_direction_count_label.kill()

        self.dropdown_vision_line_return_type.kill()
        self.dropdown_vision_line_return_type_label.kill()

        self.population_count_entry.kill()
        self.population_count_entry_label.kill()

        self.mutation_rate_entry.kill()
        self.mutation_rate_entry_label.kill()

        self.starting_snake_size_entry.kill()
        self.starting_snake_size_entry_label.kill()

        self.board_size_entry.kill()
        self.board_size_entry_label.kill()

        self.file_name_entry.kill()
        self.file_name_label.kill()

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
                    self.set_target_state_name(State.GENETIC_MENU)
                    self.trigger_transition()

                if event.ui_element == self.button_run:
                    self.set_target_state_name(State.GENETIC_TRAIN_NEW_NETWORK)
                    self.data_to_send = {
                        "input_direction_count": int(self.dropdown_input_direction_count.selected_option),
                        "vision_return_type": self.dropdown_vision_line_return_type.selected_option,
                        "file_name": self.file_name_entry.text,
                        "population_count": int(self.population_count_entry.text),
                        "mutation_rate": float(self.mutation_rate_entry.text),
                        "starting_snake_size": int(self.starting_snake_size_entry.text),
                        "board_size": int(self.board_size_entry.text)
                    }
                    self.trigger_transition()

        self.ui_manager.update(time_delta)

        self.ui_manager.draw_ui(surface)
