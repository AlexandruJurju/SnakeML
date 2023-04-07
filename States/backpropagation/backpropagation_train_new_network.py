import copy

import pygame_gui
from pygame_gui import UIManager
from pygame_gui.elements import UILabel, UIButton

import neural_network
import vision
from States.base_state import BaseState
from file_operations import TrainingExample, save_neural_network_to_json, read_training_data_and_train, write_examples_to_json_4d
from game_config import State, GameSettings
from neural_network import *
from view import *


class BackpropagationTrainNewNetwork(BaseState):
    def __init__(self, ui_manager: UIManager):
        super().__init__(State.BACKPROPAGATION_TRAIN_NEW_NETWORK)

        self.training = None
        self.initial_board_size = None
        self.initial_snake_size = None
        self.input_direction_count = None
        self.segment_return_type = None
        self.apple_return_type = None
        self.segment_return_type = None
        self.distance_function = None
        self.file_name = None
        self.ui_manager = ui_manager
        self.model = None

        self.training_examples: List[TrainingExample] = []
        self.evaluated: List[TrainingExample] = []

        self.title_label = None
        self.button_back = None

    def start(self):
        self.title_label = UILabel(pygame.Rect(ViewSettings.TITLE_LABEL_POSITION, ViewSettings.TITLE_LABEL_DIMENSION), "Backpropagation Train New Network", self.ui_manager)
        self.button_back = UIButton(pygame.Rect(ViewSettings.BUTTON_BACK_POSITION, ViewSettings.BUTTON_BACK_DIMENSION), "BACK", self.ui_manager)

        self.initial_board_size = self.data_received["board_size"]
        self.initial_snake_size = self.data_received["initial_snake_size"]
        self.input_direction_count = self.data_received["input_direction_count"]
        self.segment_return_type = self.data_received["segment_return_type"]
        self.apple_return_type = self.data_received["apple_return_type"]
        self.distance_function = getattr(vision, self.data_received["distance_function"])
        self.file_name = self.data_received["file_name"]

        input_neuron_count = self.data_received["input_layer_neurons"]
        hidden_neuron_count = self.data_received["hidden_layer_neurons"]
        output_neuron_count = self.data_received["output_layer_neurons"]

        hidden_activation = getattr(neural_network, self.data_received["hidden_activation"])
        output_activation = getattr(neural_network, self.data_received["output_activation"])
        hidden_activation_prime = getattr(neural_network, self.data_received["hidden_activation"] + "_prime")
        output_activation_prime = getattr(neural_network, self.data_received["output_activation"] + "_prime")

        net = NeuralNetwork()
        net.add_layer(Dense(input_neuron_count, hidden_neuron_count))
        net.add_layer(Activation(hidden_activation, hidden_activation_prime))
        net.add_layer(Dense(hidden_neuron_count, output_neuron_count))
        net.add_layer(Activation(output_activation, output_activation_prime))

        self.model = Model(self.initial_board_size, self.initial_snake_size, False, net)
        self.training = False

    def end(self):
        self.title_label.kill()
        self.button_back.kill()

    def is_example_in_evaluated(self, example: TrainingExample):
        for eval_example in self.evaluated:
            if eval_example.vision_lines == example.vision_lines:
                return True
        return False

    def execute(self, surface):
        vision_lines = vision.get_vision_lines_snake_head(self.model.board, self.model.snake.body[0], self.input_direction_count,
                                                          max_dist=0, apple_return_type=self.apple_return_type, segment_return_type=self.segment_return_type, distance_function=self.distance_function)
        nn_output = self.model.get_nn_output(vision_lines)

        example_output = np.where(nn_output == np.max(nn_output), 1, 0)
        example = TrainingExample(copy.deepcopy(self.model.board), self.model.snake.direction, vision_lines, example_output.ravel().tolist())

        # print(len(self.evaluated))
        if len(self.evaluated) == 0:
            self.training_examples.append(example)
        else:
            if not self.is_example_in_evaluated(example):
                self.training_examples.append(example)

        draw_board(surface, self.model.board, ViewSettings.BOARD_POSITION[0], ViewSettings.BOARD_POSITION[1])

        next_direction = self.model.get_nn_output_4directions(nn_output)
        is_alive = self.model.move(next_direction)
        if not is_alive:
            self.training = True

    def wait_for_key(self) -> str:
        while True:
            event = pygame.event.wait()
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.button_back:
                    self.set_target_state_name(State.OPTIONS)
                    self.data_to_send["state"] = "backpropagation"
                    self.trigger_transition()
                    break

            if event.type == pygame.KEYDOWN:
                match event.key:
                    case pygame.K_w:
                        return "W"
                    case pygame.K_s:
                        return "S"
                    case pygame.K_a:
                        return "A"
                    case pygame.K_d:
                        return "D"
                    case pygame.K_RETURN:
                        return ""
                    case pygame.K_x:
                        return "X"
                    case pygame.K_ESCAPE:
                        self.set_target_state_name(State.QUIT)
                        self.trigger_transition()
                        break

    def train_backpropagation(self, surface, time_delta):
        current_example = self.training_examples[0]
        self.training_examples.pop(0)

        draw_board(surface, current_example.board, ViewSettings.BOARD_POSITION[0], ViewSettings.BOARD_POSITION[1])
        draw_next_snake_direction(surface, current_example.board, self.model.get_nn_output_4directions(current_example.predictions), ViewSettings.BOARD_POSITION[0], ViewSettings.BOARD_POSITION[1])

        snake_head = find_snake_head_poz(current_example.board)
        vision_lines = vision.get_vision_lines_snake_head(current_example.board, snake_head, self.input_direction_count, max_dist=0, apple_return_type=self.apple_return_type,
                                                          segment_return_type=self.segment_return_type, distance_function=self.distance_function)
        draw_vision_lines(surface, snake_head, vision_lines, ViewSettings.BOARD_POSITION[0], ViewSettings.BOARD_POSITION[1])
        write_controls(surface, 300, 300)

        self.ui_manager.update(time_delta)
        self.ui_manager.draw_ui(surface)
        pygame.display.flip()

        input_string = self.wait_for_key()
        skip = False

        if input_string == "X":
            skip = True
        else:
            if input_string == "":
                target_output = current_example.predictions
            else:
                target_output = [0.0, 0.0, 0.0, 0.0]
                if input_string == "W":
                    target_output[0] = 1.0
                if input_string == "S":
                    target_output[1] = 1.0
                if input_string == "A":
                    target_output[2] = 1.0
                if input_string == "D":
                    target_output[3] = 1.0

            self.evaluated.append(TrainingExample(current_example.board, current_example.current_direction, current_example.vision_lines, target_output))

        if len(self.training_examples) == 0 or skip:
            self.training_examples.clear()
            file_path = "Backpropagation_Training/" + str(self.input_direction_count) + "_in_directions_" + str(self.data_received["output_layer_neurons"]) + "_out_directions.json"

            write_examples_to_json_4d(self.evaluated, file_path)

            # self.evaluated.clear()

            self.model.snake.brain.reinit_weights_and_biases()
            self.model = Model(self.initial_board_size, self.initial_snake_size, False, self.model.snake.brain)
            read_training_data_and_train(self.model.snake.brain, file_path)
            self.training = False

    def run(self, surface, time_delta):
        surface.fill(self.ui_manager.ui_theme.get_colour("dark_bg"))

        if not self.training:
            self.execute(surface)
        else:
            self.train_backpropagation(surface, time_delta)

            data_to_save = {
                "generation": -1,
                "initial_board_size": self.initial_board_size,
                "initial_snake_size": self.initial_snake_size,
                "input_direction_count": self.input_direction_count,
                "apple_return_type": self.apple_return_type,
                "segment_return_type": self.segment_return_type,
                "distance_function": self.data_received["distance_function"]
            }

            save_neural_network_to_json(data_to_save,
                                        self.model.snake.brain,
                                        GameSettings.BACKPROPAGATION_NETWORK_FOLDER + self.data_received["file_name"])

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
                        self.set_target_state_name(State.OPTIONS)
                        self.data_to_send = {
                            "state": "backpropagation"
                        }
                        self.trigger_transition()

            self.ui_manager.update(time_delta)

            self.ui_manager.draw_ui(surface)
