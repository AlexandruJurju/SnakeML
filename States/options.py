from typing import Tuple, Dict

import pygame
import pygame_gui
from pygame_gui import UIManager
from pygame_gui.elements import UILabel, UIButton, UIDropDownMenu, UITextEntryLine

from States.base_state import BaseState
from game_config import *


# noinspection PyTypeChecker
class Options(BaseState):
    def __init__(self, ui_manager: UIManager):
        super().__init__(State.OPTIONS)

        self.ui_manager = ui_manager
        self.options_target = None

        self.snake_options_list: [] = []
        self.genetic_options_list: [] = []
        self.neural_network_options_list: [] = []
        self.options_state = 0

        self.title_label = None
        self.button_back: UIButton = None

        self.button_next: UIButton = None
        self.options_done = False

        self.dropdown_input_direction_count: UIDropDownMenu = None
        self.dropdown_input_direction_count_label: UILabel = None

        self.dropdown_segment_return: UIDropDownMenu = None
        self.label_dropdown_segment_return: UILabel = None

        self.dropdown_apple_return: UIDropDownMenu = None
        self.label_dropdown_apple_return: UILabel = None

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

        self.dropdown_hidden_function: UIDropDownMenu = None
        self.dropdown_hidden_function_label: UILabel = None

        self.dropdown_activation_function_output: UILabel = None
        self.dropdown_activation_function_output_label: UILabel = None

        self.crossover_operators_dropdown: UIDropDownMenu = None
        self.crossover_operators_label = None

        self.selection_operators_dropdown: UIDropDownMenu = None
        self.selection_operators_label = None

        self.mutation_operators_dropdown: UIDropDownMenu = None
        self.mutation_operators_label = None

        self.distance_function: UILabel = None
        self.distance_function_label: UILabel = None

        self.hidden_layer_count_dropdown: UIDropDownMenu = None
        self.hidden_layer_count_dropdown_label: UILabel = None
        self.neural_network_layers_entries: Dict[Tuple[UITextEntryLine, UILabel]] = {}

    def start(self):
        self.options_target = self.data_received["state"]
        self.options_state = 0
        self.snake_options_list: [] = []
        self.genetic_options_list: [] = []
        self.neural_network_options_list: [] = []

        x_positions = {"left-left": ViewSettings.X_CENTER - 500,
                       "left-center": ViewSettings.X_CENTER - 250,
                       "center": ViewSettings.X_CENTER,
                       "right-center": ViewSettings.X_CENTER + 250,
                       "right-right": ViewSettings.X_CENTER + 500}
        y_positions = [ViewSettings.Y_CENTER - 250, ViewSettings.Y_CENTER - 50, ViewSettings.Y_CENTER + 150]
        x_positions_label = {"left-left": ViewSettings.X_CENTER - 500 - 125,
                             "left-center": ViewSettings.X_CENTER - 250 - 125,
                             "center": ViewSettings.X_CENTER - 125,
                             "right-center": ViewSettings.X_CENTER + 250 - 125,
                             "right-right": ViewSettings.X_CENTER + 500 - 125}
        y_positions_label = [ViewSettings.Y_CENTER - 250 - 50, ViewSettings.Y_CENTER - 50 - 50, ViewSettings.Y_CENTER + 150 - 50]
        self.title_label = UILabel(pygame.Rect(ViewSettings.TITLE_LABEL_POSITION, ViewSettings.TITLE_LABEL_DIMENSION), "", self.ui_manager)
        self.button_back = UIButton(pygame.Rect(ViewSettings.BUTTON_BACK_POSITION, ViewSettings.BUTTON_BACK_DIMENSION), "BACK", self.ui_manager)

        # ================================================
        self.starting_snake_size_entry = UITextEntryLine(pygame.Rect((x_positions["left-center"] - 75 // 2, y_positions[1]), (75, 30)), self.ui_manager)
        self.starting_snake_size_entry_label = UILabel(pygame.Rect((x_positions_label["left-center"], y_positions_label[1]), (250, 35)), "Starting Snake Size", self.ui_manager)
        self.starting_snake_size_entry.set_text(str(GameSettings.INITIAL_SNAKE_SIZE))

        self.board_size_entry = UITextEntryLine(pygame.Rect((x_positions["center"] - 75 // 2, y_positions[1]), (75, 30)), self.ui_manager)
        self.board_size_entry_label = UILabel(pygame.Rect((x_positions_label["center"], y_positions_label[1]), (250, 35)), "Board Size", self.ui_manager)
        self.board_size_entry.set_text(str(GameSettings.INITIAL_BOARD_SIZE))

        # ================================================
        self.dropdown_activation_function_output = UILabel(pygame.Rect((x_positions["left-center"] - 125 // 2, y_positions[0]), (125, 30)), GameSettings.AVAILABLE_ACTIVATION_FUNCTIONS[0], self.ui_manager)
        self.dropdown_activation_function_output_label = UILabel(pygame.Rect((x_positions_label["left-center"], y_positions_label[0]), (250, 35)), "Output Activation Function", self.ui_manager)

        self.dropdown_hidden_function = UIDropDownMenu(GameSettings.AVAILABLE_ACTIVATION_FUNCTIONS, GameSettings.AVAILABLE_ACTIVATION_FUNCTIONS[2],
                                                       pygame.Rect((x_positions["left-left"] - 125 // 2, y_positions[0]), (125, 30)), self.ui_manager)
        self.dropdown_hidden_function_label = UILabel(pygame.Rect((x_positions_label["left-left"], y_positions_label[0]), (250, 35)), "Hidden Activation Function", self.ui_manager)

        self.distance_function = UILabel(pygame.Rect((x_positions["right-right"] - 225 // 2, y_positions[1]), (225, 30)), "chebyshev", self.ui_manager)
        self.distance_function_label = UILabel(pygame.Rect((x_positions_label["right-right"], y_positions_label[1]), (250, 35)), "Distance Function", self.ui_manager)

        self.dropdown_input_direction_count = UIDropDownMenu(GameSettings.AVAILABLE_INPUT_DIRECTIONS, GameSettings.AVAILABLE_INPUT_DIRECTIONS[0], pygame.Rect((x_positions["left-left"] - 75 // 2, y_positions[1]), (75, 30)), self.ui_manager)
        self.dropdown_input_direction_count_label = UILabel(pygame.Rect((x_positions_label["left-left"], y_positions_label[1]), (250, 35)), "Input Direction Count", self.ui_manager)

        self.hidden_layer_count_dropdown = UIDropDownMenu(["1", "2", "3"], "1", pygame.Rect((x_positions["center"] - 75 // 2, y_positions[1]), (75, 30)), self.ui_manager)
        self.hidden_layer_count_dropdown_label = UILabel(pygame.Rect((x_positions_label["center"], y_positions_label[1]), (250, 35)), "Hidden Layer Count", self.ui_manager)

        available_returns = ["boolean", "distance"]
        if self.options_target == "backpropagation":
            available_returns = ["distance", "boolean"]
        self.dropdown_segment_return = UIDropDownMenu(available_returns, available_returns[0], pygame.Rect((x_positions["right-center"] - 125 // 2, y_positions[0]), (125, 30)), self.ui_manager)
        self.label_dropdown_segment_return = UILabel(pygame.Rect((x_positions_label["right-center"], y_positions_label[0]), (250, 35)), "Segment Return type", self.ui_manager)

        self.dropdown_apple_return = UIDropDownMenu(available_returns, available_returns[0], pygame.Rect((x_positions["right-right"] - 125 // 2, y_positions[0]), (125, 30)), self.ui_manager)
        self.label_dropdown_apple_return = UILabel(pygame.Rect((x_positions_label["right-right"], y_positions_label[0]), (250, 35)), "Apple Return type", self.ui_manager)

        input_neuron_count = int(self.dropdown_input_direction_count.selected_option) * 3 + 2
        input_layer = UILabel(pygame.Rect((x_positions["left-left"] - 75 // 2, y_positions[2]), (75, 30)), str(input_neuron_count), self.ui_manager)
        input_layer_label = UILabel(pygame.Rect((x_positions_label["left-left"], y_positions_label[2]), (250, 30)), "Input Layer", self.ui_manager)

        output_neuron_count = 4 if int(self.dropdown_input_direction_count.selected_option) == 4 or int(self.dropdown_input_direction_count.selected_option) == 8 else 3
        output_layer = UILabel(pygame.Rect((x_positions["right-right"] - 75 // 2, y_positions[2]), (75, 30)), str(output_neuron_count), self.ui_manager)
        output_layer_label = UILabel(pygame.Rect((x_positions_label["right-right"], y_positions_label[2]), (250, 30)), "Output Layer", self.ui_manager)

        first_hidden_layer_neuron_count = input_neuron_count + 4
        first_hidden_layer = UITextEntryLine(pygame.Rect((x_positions["left-center"] - 75 // 2, y_positions[2]), (75, 30)), self.ui_manager)
        first_hidden_layer_label = UILabel(pygame.Rect((x_positions_label["left-center"], y_positions_label[2]), (250, 30)), "Hidden Layer 1", self.ui_manager)
        first_hidden_layer.set_text(str(first_hidden_layer_neuron_count))

        second_hidden_layer = UITextEntryLine(pygame.Rect((x_positions["center"] - 75 // 2, y_positions[2]), (75, 30)), self.ui_manager)
        second_hidden_layer_label = UILabel(pygame.Rect((x_positions_label["center"], y_positions_label[2]), (250, 30)), "Hidden Layer 2", self.ui_manager)
        second_hidden_layer.set_text(str(first_hidden_layer_neuron_count))

        third_hidden_layer = UITextEntryLine(pygame.Rect((x_positions["right-center"] - 75 // 2, y_positions[2]), (75, 30)), self.ui_manager)
        third_hidden_layer_label = UILabel(pygame.Rect((x_positions_label["right-center"], y_positions_label[2]), (250, 30)), "Hidden Layer 3", self.ui_manager)
        third_hidden_layer.set_text(str(first_hidden_layer_neuron_count))

        self.neural_network_layers_entries["input"] = ([input_layer, input_layer_label])
        self.neural_network_layers_entries["first"] = ([first_hidden_layer, first_hidden_layer_label])
        self.neural_network_layers_entries["second"] = ([second_hidden_layer, second_hidden_layer_label])
        self.neural_network_layers_entries["third"] = ([third_hidden_layer, third_hidden_layer_label])
        self.neural_network_layers_entries["output"] = ([output_layer, output_layer_label])

        # ================================================
        self.population_count_entry = UITextEntryLine(pygame.Rect((x_positions["center"] - 75 // 2, y_positions[0]), (75, 30)), self.ui_manager)
        self.population_count_entry_label = UILabel(pygame.Rect((x_positions_label["center"], y_positions_label[0]), (250, 35)), "Individuals in Population", self.ui_manager)
        self.population_count_entry.set_text(str(GameSettings.POPULATION_COUNT))

        self.crossover_operators_dropdown = UIDropDownMenu(GameSettings.AVAILABLE_CROSSOVER_OPERATORS, GameSettings.AVAILABLE_CROSSOVER_OPERATORS[0], pygame.Rect((x_positions["center"] - 225 // 2, y_positions[1]), (225, 30)), self.ui_manager)
        self.crossover_operators_label = UILabel(pygame.Rect((x_positions_label["center"], y_positions_label[1]), (250, 35)), "Crossover Operators", self.ui_manager)

        self.selection_operators_dropdown = UIDropDownMenu(GameSettings.AVAILABLE_SELECTION_OPERATORS, GameSettings.AVAILABLE_SELECTION_OPERATORS[0], pygame.Rect((x_positions["left-left"] - 225 // 2, y_positions[1]), (225, 30)), self.ui_manager)
        self.selection_operators_label = UILabel(pygame.Rect((x_positions_label["left-left"], y_positions_label[1]), (250, 35)), "Selection Operators", self.ui_manager)

        self.mutation_operators_dropdown = UIDropDownMenu(GameSettings.AVAILABLE_MUTATION_OPERATORS, GameSettings.AVAILABLE_MUTATION_OPERATORS[0], pygame.Rect((x_positions["right-right"] - 225 // 2, y_positions[1]), (225, 30)), self.ui_manager)
        self.mutation_operators_label = UILabel(pygame.Rect((x_positions_label["right-right"], y_positions_label[1]), (250, 35)), "Mutation Operators", self.ui_manager)

        self.mutation_rate_entry = UITextEntryLine(pygame.Rect((x_positions["right-right"] - 75 // 2, y_positions[2]), (75, 30)), self.ui_manager)
        self.mutation_rate_entry_label = UILabel(pygame.Rect((x_positions_label["right-right"], y_positions_label[2]), (250, 35)), "Mutation Rate", self.ui_manager)
        self.mutation_rate_entry.set_text(str(GameSettings.MUTATION_CHANCE))

        # ================================================
        self.genetic_options_list = [self.mutation_rate_entry, self.mutation_rate_entry_label,
                                     self.population_count_entry, self.population_count_entry_label,
                                     self.crossover_operators_dropdown, self.crossover_operators_label,
                                     self.selection_operators_dropdown, self.selection_operators_label,
                                     self.mutation_operators_dropdown, self.mutation_operators_label]

        self.snake_options_list = [self.starting_snake_size_entry, self.starting_snake_size_entry_label,
                                   self.board_size_entry, self.board_size_entry_label]

        self.neural_network_options_list = [self.dropdown_activation_function_output, self.dropdown_activation_function_output_label,
                                            self.dropdown_hidden_function, self.dropdown_hidden_function_label,
                                            self.dropdown_input_direction_count, self.dropdown_input_direction_count_label,
                                            self.dropdown_segment_return, self.label_dropdown_segment_return,
                                            self.dropdown_apple_return, self.label_dropdown_apple_return,
                                            self.hidden_layer_count_dropdown, self.hidden_layer_count_dropdown_label,
                                            self.distance_function, self.distance_function_label]

        # ================================================
        self.file_name_entry = UITextEntryLine(pygame.Rect((x_positions["center"] - 175 // 2, y_positions[1]), (175, 30)), self.ui_manager)
        self.file_name_entry_label = UILabel(pygame.Rect((x_positions_label["center"], y_positions_label[1]), (250, 35)), "Network name", self.ui_manager)
        self.file_name_entry.set_text("Default")
        self.file_name_entry.hide()
        self.file_name_entry_label.hide()

        self.button_next = UIButton(pygame.Rect((ViewSettings.X_CENTER - 75 // 2, ViewSettings.HEIGHT - 75), (75, 40)), "Next", self.ui_manager)

    def hide_layer_entries(self):
        for key in self.neural_network_layers_entries:
            self.neural_network_layers_entries[key][0].hide()
            self.neural_network_layers_entries[key][1].hide()

    def show_layer_entries(self):
        self.neural_network_layers_entries["input"][0].show()
        self.neural_network_layers_entries["input"][1].show()

        self.neural_network_layers_entries["output"][0].show()
        self.neural_network_layers_entries["output"][1].show()

        if self.hidden_layer_count_dropdown.selected_option == "1":
            self.neural_network_layers_entries["first"][0].show()
            self.neural_network_layers_entries["first"][1].show()

        if self.hidden_layer_count_dropdown.selected_option == "2":
            self.neural_network_layers_entries["first"][0].show()
            self.neural_network_layers_entries["first"][1].show()

            self.neural_network_layers_entries["second"][0].show()
            self.neural_network_layers_entries["second"][1].show()

        if self.hidden_layer_count_dropdown.selected_option == "3":
            self.neural_network_layers_entries["first"][0].show()
            self.neural_network_layers_entries["first"][1].show()

            self.neural_network_layers_entries["second"][0].show()
            self.neural_network_layers_entries["second"][1].show()

            self.neural_network_layers_entries["third"][0].show()
            self.neural_network_layers_entries["third"][1].show()

    def end(self):
        self.ui_manager.clear_and_reset()

    def hide_all(self):
        for option in self.snake_options_list:
            option.hide()
        for option in self.neural_network_options_list:
            option.hide()
        for option in self.genetic_options_list:
            option.hide()
        self.hide_layer_entries()
        self.file_name_entry.hide()
        self.file_name_entry_label.hide()

    def draw_options(self):
        match self.options_state:
            case -1:
                self.set_target_state_name(State.MAIN_MENU)
                self.trigger_transition()

            case 0:
                self.hide_all()
                for option in self.snake_options_list:
                    option.show()
                self.title_label.set_text("Snake Game Options")

            case 1:
                for option in self.snake_options_list:
                    option.hide()
                for option in self.genetic_options_list:
                    option.hide()
                self.file_name_entry.hide()
                self.file_name_entry_label.hide()

                self.show_layer_entries()
                for option in self.neural_network_options_list:
                    option.show()
                self.title_label.set_text("Neural Network Options")

            case 2:
                self.hide_layer_entries()
                for option in self.neural_network_options_list:
                    option.hide()
                self.file_name_entry.hide()
                self.file_name_entry_label.hide()

                for option in self.genetic_options_list:
                    option.show()
                self.title_label.set_text("Genetic Algorithm Options")

            case 3:
                for option in self.genetic_options_list:
                    option.hide()
                for option in self.neural_network_options_list:
                    option.hide()
                self.hide_layer_entries()

                self.file_name_entry.show()
                self.file_name_entry_label.show()
                self.button_next.set_text("RUN")

            case 4:
                self.data_to_send = {
                    "input_direction_count": int(self.dropdown_input_direction_count.selected_option),
                    "segment_return_type": self.dropdown_segment_return.selected_option,
                    "apple_return_type": self.dropdown_apple_return.selected_option,
                    "distance_function": self.distance_function.text,
                    "file_name": self.file_name_entry.text,
                    "hidden_activation": self.dropdown_hidden_function.selected_option,
                    "output_activation": self.dropdown_activation_function_output.text,
                    "input_layer_neurons": int(self.neural_network_layers_entries["input"][0].text),
                    "hidden_layer_neurons": int(self.neural_network_layers_entries["first"][0].text),
                    "output_layer_neurons": int(self.neural_network_layers_entries["output"][0].text),
                    "initial_snake_size": int(self.starting_snake_size_entry.text),
                    "board_size": int(self.board_size_entry.text)
                }

                if self.options_target == "genetic":
                    self.set_target_state_name(State.GENETIC_TRAIN_NEW_NETWORK)
                    self.data_to_send.update(
                        {
                            "hidden_layer_count": int(self.hidden_layer_count_dropdown.selected_option),
                            "hidden_layer1_neuron_count": int(self.neural_network_layers_entries["first"][0].text),
                            "hidden_layer2_neuron_count": int(self.neural_network_layers_entries["second"][0].text),
                            "hidden_layer3_neuron_count": int(self.neural_network_layers_entries["third"][0].text),
                            "population_count": int(self.population_count_entry.text),
                            "selection_operator": self.selection_operators_dropdown.selected_option,
                            "crossover_operator": self.crossover_operators_dropdown.selected_option,
                            "mutation_operator": self.mutation_operators_dropdown.selected_option,
                            "mutation_rate": float(self.mutation_rate_entry.text)
                        }
                    )
                else:
                    self.set_target_state_name(State.BACKPROPAGATION_TRAIN_NEW_NETWORK)

                self.trigger_transition()

    def draw_hidden_layer_inputs(self):
        if self.hidden_layer_count_dropdown.selected_option == "1":
            pass
        if self.hidden_layer_count_dropdown.selected_option == "2":
            pass
        if self.hidden_layer_count_dropdown.selected_option == "3":
            pass

    def run(self, surface, time_delta):
        surface.fill(self.ui_manager.ui_theme.get_colour("main_bg"))

        self.neural_network_layers_entries["input"][0].set_text(str(int(self.dropdown_input_direction_count.selected_option) * 3 + 2))
        if self.dropdown_input_direction_count.selected_option == "4":
            self.distance_function.set_text(GameSettings.AVAILABLE_DISTANCES[0])
        else:
            self.distance_function.set_text(GameSettings.AVAILABLE_DISTANCES[1])
        self.draw_options()

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
                    if self.options_state == 3 and self.options_target == "backpropagation":
                        self.options_state -= 2
                    else:
                        self.options_state -= 1

                if event.ui_element == self.button_next:
                    if self.options_state == 1 and self.options_target == "backpropagation":
                        self.options_state += 2
                    else:
                        self.options_state += 1

        self.ui_manager.update(time_delta)
        self.ui_manager.draw_ui(surface)
