from typing import List, Tuple

import pygame
import pygame_gui
from pygame_gui import UIManager
from pygame_gui.elements import UILabel, UIButton, UIDropDownMenu, UITextEntryLine

from States.base_state import BaseState
from States.state_manager import StateManager
from game_config import *


# noinspection PyTypeChecker
class Options(BaseState):
    def __init__(self, state_manager: StateManager, ui_manager: UIManager):
        super().__init__(State.OPTIONS, state_manager)

        self.ui_manager = ui_manager
        self.options_target = None

        self.snake_options_list: [] = []
        self.genetic_options_list: [] = []
        self.neural_network_options_list: [] = []

        self.title_label = None
        self.button_back: UIButton = None

        self.button_run = None
        self.options_done = False

        self.button_snake_options: UIButton = None
        self.button_neuronal_network_options: UIButton = None
        self.button_genetic_options: UIButton = None

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
        self.file_name_entry_label: UILabel = None

        self.dropdown_activation_function_hidden: UIDropDownMenu = None
        self.dropdown_activation_function_hidden_label: UILabel = None

        self.dropdown_activation_function_output: UIDropDownMenu = None
        self.dropdown_activation_function_output_label: UILabel = None

        self.crossover_operators = None
        self.crossover_operators_label = None

        self.selection_operators = None
        self.selection_operators_label = None

        self.mutation_operators = None
        self.mutation_operators_label = None

        self.hidden_layer_count_entry: UITextEntryLine = None
        self.hidden_layer_count_entry_label: UILabel = None
        self.neural_network_layers_entries: List[Tuple[UITextEntryLine, UILabel]] = []

    def start(self):
        self.options_target = self.data_received["state"]

        self.button_snake_options = UIButton(pygame.Rect((ViewSettings.X_SECOND - 625, 150), (225, 40)), "Snake Options", self.ui_manager)
        self.button_neuronal_network_options = UIButton(pygame.Rect((ViewSettings.X_SECOND - 625, 300), (225, 40)), "Neural Network Options", self.ui_manager)
        self.button_genetic_options = UIButton(pygame.Rect((ViewSettings.X_SECOND - 625, 450), (225, 40)), "Genetic Algorithm Options", self.ui_manager)

        self.title_label = UILabel(pygame.Rect(ViewSettings.TITLE_LABEL_POSITION, ViewSettings.TITLE_LABEL_DIMENSION), "", self.ui_manager, object_id="#window_label")
        self.button_back = UIButton(pygame.Rect(ViewSettings.BUTTON_BACK_POSITION, ViewSettings.BUTTON_BACK_DIMENSION), "BACK", self.ui_manager)

        if self.options_target == "genetic":
            self.title_label.set_text("Genetic Options")
        else:
            self.title_label.set_text("Backpropagation Options")

        self.starting_snake_size_entry = UITextEntryLine(pygame.Rect((ViewSettings.X_SECOND - 75 // 2 - 250, 150), (75, 30)), self.ui_manager)
        self.starting_snake_size_entry_label = UILabel(pygame.Rect((ViewSettings.X_SECOND - 250 // 2 - 250, 100), (250, 35)), "Starting Snake Size", self.ui_manager)
        self.starting_snake_size_entry.set_text(str(GameSettings.INITIAL_SNAKE_SIZE))

        self.board_size_entry = UITextEntryLine(pygame.Rect((ViewSettings.X_SECOND - 75 // 2, 150), (75, 30)), self.ui_manager)
        self.board_size_entry_label = UILabel(pygame.Rect((ViewSettings.X_SECOND - 250 // 2, 100), (250, 35)), "Board Size", self.ui_manager)
        self.board_size_entry.set_text(str(GameSettings.INITIAL_BOARD_SIZE))

        self.dropdown_input_direction_count = UIDropDownMenu(GameSettings.AVAILABLE_INPUT_DIRECTIONS, GameSettings.AVAILABLE_INPUT_DIRECTIONS[0], pygame.Rect((ViewSettings.X_SECOND - 75 // 2 - 250, 350), (75, 30)),
                                                             self.ui_manager)
        self.dropdown_input_direction_count_label = UILabel(pygame.Rect((ViewSettings.X_SECOND - 250 // 2 - 250, 300), (250, 35)), "Input Direction Count", self.ui_manager)

        self.file_name_entry = UITextEntryLine(pygame.Rect((ViewSettings.X_SECOND - 175 // 2, 350), (175, 30)), self.ui_manager)
        self.file_name_entry_label = UILabel(pygame.Rect((ViewSettings.X_SECOND - 125 // 2, 300), (125, 35)), "Network name", self.ui_manager)
        self.file_name_entry.set_text("Default")
        self.file_name_entry.hide()
        self.file_name_entry_label.hide()

        self.dropdown_vision_line_return_type = UIDropDownMenu(GameSettings.AVAILABLE_VISION_LINES_RETURN_TYPE, GameSettings.AVAILABLE_VISION_LINES_RETURN_TYPE[0],
                                                               pygame.Rect((ViewSettings.X_SECOND - 125 // 2 + 250, 150), (125, 30)), self.ui_manager)
        self.dropdown_vision_line_return_type_label = UILabel(pygame.Rect((ViewSettings.X_SECOND - 250 // 2 + 250, 100), (250, 35)), "Vision Line Return Type", self.ui_manager)

        self.population_count_entry = UITextEntryLine(pygame.Rect((ViewSettings.X_SECOND - 75 // 2, 150), (75, 30)), self.ui_manager)
        self.population_count_entry_label = UILabel(pygame.Rect((ViewSettings.X_SECOND - 200 // 2, 100), (200, 35)), "Individuals in Population", self.ui_manager)
        self.population_count_entry.set_text(str(GameSettings.POPULATION_COUNT))

        self.mutation_rate_entry = UITextEntryLine(pygame.Rect((ViewSettings.X_SECOND - 75 // 2 + 250, 550), (75, 30)), self.ui_manager)
        self.mutation_rate_entry_label = UILabel(pygame.Rect((ViewSettings.X_SECOND - 200 // 2 + 250, 500), (200, 35)), "Mutation Rate", self.ui_manager)
        self.mutation_rate_entry.set_text(str(GameSettings.MUTATION_CHANCE))

        self.dropdown_activation_function_output_label = UILabel(pygame.Rect((ViewSettings.X_SECOND - 250 // 2, 100), (250, 35)), "Output Activation Function", self.ui_manager)
        self.dropdown_activation_function_output = UIDropDownMenu(GameSettings.AVAILABLE_ACTIVATION_FUNCTIONS, GameSettings.AVAILABLE_ACTIVATION_FUNCTIONS[0],
                                                                  pygame.Rect((ViewSettings.X_SECOND - 125 // 2, 150), (125, 30)), self.ui_manager)

        self.dropdown_activation_function_hidden_label = UILabel(pygame.Rect((ViewSettings.X_SECOND - 250 // 2 - 250, 100), (250, 35)), "Hidden Activation Function", self.ui_manager)
        self.dropdown_activation_function_hidden = UIDropDownMenu(GameSettings.AVAILABLE_ACTIVATION_FUNCTIONS, GameSettings.AVAILABLE_ACTIVATION_FUNCTIONS[0],
                                                                  pygame.Rect((ViewSettings.X_SECOND - 125 // 2 - 250, 150), (125, 30)), self.ui_manager)

        self.hidden_layer_count_entry = UITextEntryLine(pygame.Rect((ViewSettings.X_SECOND - 75 // 2, 350), (75, 30)), self.ui_manager)
        self.hidden_layer_count_entry_label = UILabel(pygame.Rect((ViewSettings.X_SECOND - 250 // 2, 300), (250, 35)), "Hidden Layer Count", self.ui_manager)
        self.hidden_layer_count_entry.set_text("1")

        input_neuron_count = int(self.dropdown_input_direction_count.selected_option) * 3 + 4
        input_layer = UITextEntryLine(pygame.Rect((ViewSettings.X_SECOND - 75 // 2 - 250, 550), (75, 30)), self.ui_manager)
        input_layer_label = UILabel(pygame.Rect((ViewSettings.X_SECOND - 125 // 2 - 250, 500), (125, 30)), "Input Layer", self.ui_manager)
        input_layer.set_text(str(input_neuron_count))
        # input_layer.disable()

        output_neuron_count = 4 if int(self.dropdown_input_direction_count.selected_option) == 4 or int(self.dropdown_input_direction_count.selected_option) == 8 else 3
        output_layer = UITextEntryLine(pygame.Rect((ViewSettings.X_SECOND - 75 // 2 + 250, 550), (75, 30)), self.ui_manager)
        output_layer_label = UILabel(pygame.Rect((ViewSettings.X_SECOND - 125 // 2 + 250, 500), (125, 30)), "Output Layer", self.ui_manager)
        output_layer.set_text(str(output_neuron_count))
        # output_layer.disable()

        first_hidden_layer_neuron_count = input_neuron_count + 8
        first_hidden_layer = UITextEntryLine(pygame.Rect((ViewSettings.X_SECOND - 75 // 2, 550), (75, 30)), self.ui_manager)
        first_hidden_layer_label = UILabel(pygame.Rect((ViewSettings.X_SECOND - 125 // 2, 500), (125, 30)), "Hidden Layer", self.ui_manager)
        first_hidden_layer.set_text(str(first_hidden_layer_neuron_count))

        self.neural_network_layers_entries.append([input_layer, input_layer_label])
        self.neural_network_layers_entries.append([first_hidden_layer, first_hidden_layer_label])
        self.neural_network_layers_entries.append([output_layer, output_layer_label])

        self.hide_layer_entries()

        # TODO make genetic options do something
        self.crossover_operators = UIDropDownMenu(GameSettings.AVAILABLE_CROSSOVER_OPERATORS, GameSettings.AVAILABLE_CROSSOVER_OPERATORS[0],
                                                  pygame.Rect((ViewSettings.X_SECOND - 125 // 2, 350), (125, 30)), self.ui_manager)
        self.crossover_operators_label = UILabel(pygame.Rect((ViewSettings.X_SECOND - 250 // 2, 300), (250, 35)), "Crossover Operators", self.ui_manager)

        self.selection_operators = UIDropDownMenu(GameSettings.AVAILABLE_SELECTION_OPERATORS, GameSettings.AVAILABLE_SELECTION_OPERATORS[0],
                                                  pygame.Rect((ViewSettings.X_SECOND - 150 // 2 - 250, 350), (150, 30)), self.ui_manager)
        self.selection_operators_label = UILabel(pygame.Rect((ViewSettings.X_SECOND - 250 // 2 - 250, 300), (250, 35)), "Selection Operators", self.ui_manager)

        self.mutation_operators = UIDropDownMenu(GameSettings.AVAILABLE_MUTATION_OPERATORS, GameSettings.AVAILABLE_MUTATION_OPERATORS[0],
                                                 pygame.Rect((ViewSettings.X_SECOND - 125 // 2 + 250, 350), (125, 30)), self.ui_manager)
        self.mutation_operators_label = UILabel(pygame.Rect((ViewSettings.X_SECOND - 250 // 2 + 250, 300), (250, 35)), "Mutation Operators", self.ui_manager)

        self.genetic_options_list = [self.mutation_rate_entry, self.mutation_rate_entry_label, self.population_count_entry, self.population_count_entry_label, self.crossover_operators, self.crossover_operators_label,
                                     self.selection_operators, self.selection_operators_label, self.mutation_operators, self.mutation_operators_label]

        self.snake_options_list = [self.starting_snake_size_entry, self.starting_snake_size_entry_label, self.board_size_entry, self.board_size_entry_label]

        self.neural_network_options_list = [self.dropdown_activation_function_output, self.dropdown_activation_function_output_label, self.dropdown_activation_function_hidden,
                                            self.dropdown_activation_function_hidden_label, self.dropdown_input_direction_count,
                                            self.dropdown_input_direction_count_label, self.dropdown_vision_line_return_type, self.dropdown_vision_line_return_type_label,
                                            self.hidden_layer_count_entry, self.hidden_layer_count_entry_label]

        for option in self.snake_options_list:
            option.hide()
        for option in self.genetic_options_list:
            option.hide()
        for option in self.neural_network_options_list:
            option.hide()

        if self.options_target == "backpropagation":
            self.button_genetic_options.hide()

        self.button_run = UIButton(pygame.Rect((ViewSettings.X_SECOND - 75 // 2, 675), (75, 40)), "RUN", self.ui_manager)

    def hide_layer_entries(self):
        for ui_element in self.neural_network_layers_entries:
            ui_element[0].hide()
            ui_element[1].hide()

    def show_layer_entries(self):
        for ui_element in self.neural_network_layers_entries:
            ui_element[0].show()
            ui_element[1].show()

    def kill_layer_entries(self):
        for ui_element in self.neural_network_layers_entries:
            ui_element[0].kill()
            ui_element[1].kill()

    def end(self):
        self.title_label.kill()
        self.button_back.kill()

        self.hidden_layer_count_entry.kill()
        self.hidden_layer_count_entry_label.kill()

        self.crossover_operators.kill()
        self.crossover_operators_label.kill()

        self.selection_operators.kill()
        self.selection_operators_label.kill()

        self.button_snake_options.kill()
        self.button_genetic_options.kill()
        self.button_neuronal_network_options.kill()

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
        self.file_name_entry_label.kill()

        self.dropdown_activation_function_hidden.kill()
        self.dropdown_activation_function_hidden_label.kill()

        self.dropdown_activation_function_output.kill()
        self.dropdown_activation_function_output_label.kill()

        self.kill_layer_entries()

        self.button_run.kill()

    def run(self, surface, time_delta):
        surface.fill(self.ui_manager.ui_theme.get_colour("dark_bg"))

        # TODO disable is buggy, breaks updating values, to fix put disable after updating
        # TODO to fix, use labels instead of entries for input and output
        self.neural_network_layers_entries[0][0].set_text(str(int(self.dropdown_input_direction_count.selected_option) * 3 + 4))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.set_target_state_name(State.QUIT)
                self.trigger_transition()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.set_target_state_name(State.QUIT)
                    self.trigger_transition()

            self.ui_manager.process_events(event)

            if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                if event.ui_element == self.dropdown_input_direction_count:
                    pass

            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.button_back:
                    if self.options_target == "genetic":
                        self.set_target_state_name(State.GENETIC_MENU)
                    else:
                        self.set_target_state_name(State.BACKPROPAGATION_MENU)
                    self.trigger_transition()

                if event.ui_element == self.button_snake_options:
                    for option in self.genetic_options_list:
                        option.hide()
                    for option in self.neural_network_options_list:
                        option.hide()
                    self.hide_layer_entries()

                    for option in self.snake_options_list:
                        option.show()

                if event.ui_element == self.button_genetic_options:
                    for option in self.snake_options_list:
                        option.hide()
                    for option in self.neural_network_options_list:
                        option.hide()
                    self.hide_layer_entries()

                    for option in self.genetic_options_list:
                        option.show()

                if event.ui_element == self.button_neuronal_network_options:
                    for option in self.snake_options_list:
                        option.hide()
                    for option in self.genetic_options_list:
                        option.hide()

                    for option in self.neural_network_options_list:
                        option.show()
                    self.show_layer_entries()

                if event.ui_element == self.button_run:
                    if self.options_done is True:
                        if self.options_target == "genetic":
                            self.set_target_state_name(State.GENETIC_TRAIN_NEW_NETWORK)
                        else:
                            self.set_target_state_name(State.BACKPROPAGATION_TRAIN_NEW_NETWORK)

                        self.data_to_send = {
                            "input_direction_count": int(self.dropdown_input_direction_count.selected_option),
                            "vision_return_type": self.dropdown_vision_line_return_type.selected_option,
                            "file_name": self.file_name_entry.text,
                            "hidden_activation": self.dropdown_activation_function_hidden.selected_option,
                            "output_activation": self.dropdown_activation_function_output.selected_option,
                            "population_count": int(self.population_count_entry.text),
                            "mutation_rate": float(self.mutation_rate_entry.text),
                            "initial_snake_size": int(self.starting_snake_size_entry.text),
                            "board_size": int(self.board_size_entry.text)
                        }
                        self.trigger_transition()
                    else:
                        for option in self.snake_options_list:
                            option.hide()
                        for option in self.genetic_options_list:
                            option.hide()
                        for option in self.neural_network_options_list:
                            option.hide()
                        self.button_snake_options.hide()
                        self.button_genetic_options.hide()
                        self.button_neuronal_network_options.hide()
                        self.hide_layer_entries()

                        self.file_name_entry.show()
                        self.file_name_entry_label.show()

                        self.options_done = True

        self.ui_manager.update(time_delta)

        self.ui_manager.draw_ui(surface)
