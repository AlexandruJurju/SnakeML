import copy

import pygame_gui
from pygame_gui import UIManager
from pygame_gui.elements import UILabel, UIButton

from States.base_state import BaseState
from States.state_manager import StateManager
from constants import State
from neural_network import *
from settings import NNSettings
from train_network import TrainingExample, write_examples_to_json_4d, train_network, save_neural_network_to_json
from view import *
from vision import get_vision_lines, VisionLine


class BackpropTrainNewNetwork(BaseState):
    def __init__(self, state_manager: StateManager, ui_manager: UIManager):
        super().__init__(State.BACKPROPAGATION_TRAIN_NEW_NETWORK, state_manager)

        self.ui_manager = ui_manager
        self.model = None

        self.training_examples: List[TrainingExample] = []
        self.evaluated: List[TrainingExample] = []

        self.training = False

        self.title_label = None
        self.button_back = None

    def start(self):
        self.title_label = UILabel(pygame.Rect(ViewConsts.TITLE_LABEL_POSITION, ViewConsts.TITLE_LABEL_DIMENSION), "Backpropagation Train New Network", self.ui_manager, object_id="#window_label")
        self.button_back = UIButton(pygame.Rect(ViewConsts.BUTTON_BACK_POSITION, ViewConsts.BUTTON_BACK_DIMENSION), "BACK", self.ui_manager)

        input_direction_count = self.data_received["input_direction_count"]
        input_neuron_count = input_direction_count * 3 + 4
        hidden_neuron_count = 24
        output_neuron_count = 4 if input_direction_count == 4 or input_direction_count == 8 else 3

        net = NeuralNetwork()
        net.add_layer(Dense(input_neuron_count, hidden_neuron_count))
        net.add_layer(Activation(tanh, tanh_prime))
        net.add_layer(Dense(hidden_neuron_count, output_neuron_count))
        net.add_layer(Activation(sigmoid, sigmoid_prime))

        self.model = Model(self.data_received["board_size"], self.data_received["initial_snake_size"], False, net)

    def end(self):
        self.title_label.kill()
        self.button_back.kill()

    def print_vision_line(self, vision_line: VisionLine):
        print(f"{vision_line.wall_coord} {vision_line.wall_distance} || {vision_line.apple_coord} {vision_line.apple_distance} || {vision_line.segment_coord} {vision_line.segment_distance} ")

    def print_all_vision_lines(self, vision_lines: List[VisionLine]):
        for line in vision_lines:
            self.print_vision_line(line)

    def is_example_in_evaluated(self, example: TrainingExample):
        for eval_example in self.evaluated:
            if eval_example.vision_lines == example.vision_lines:
                return True
        return False

    def execute(self, surface):
        vision_lines = get_vision_lines(self.model.board, self.data_received["input_direction_count"], self.data_received["vision_return_type"])
        nn_output = self.model.get_nn_output(vision_lines)

        example_output = np.where(nn_output == np.max(nn_output), 1, 0)
        example = TrainingExample(copy.deepcopy(self.model.board), self.model.snake.direction, vision_lines, example_output.ravel().tolist())

        # print(len(self.evaluated))
        if len(self.evaluated) == 0:
            self.training_examples.append(example)
        else:
            if not self.is_example_in_evaluated(example):
                self.training_examples.append(example)

        draw_board(surface, self.model.board, ViewConsts.BOARD_POSITION[0], ViewConsts.BOARD_POSITION[1])
        # draw_vision_lines(surface, self.model, vision_lines, ViewConsts.BOARD_POSITION[0], ViewConsts.BOARD_POSITION[1])
        # draw_neural_network_complete(surface, self.model, vision_lines, ViewConsts.NN_POSITION[0], ViewConsts.NN_POSITION[1])

        next_direction = self.model.get_nn_output_4directions(nn_output)
        is_alive = self.model.move_in_direction(next_direction)
        if not is_alive:
            self.training = True

    def wait_for_key(self) -> str:
        while True:
            event = pygame.event.wait()
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.button_back:
                    self.set_target_state_name(State.BACKPROPAGATION_TRAIN_NEW_NETWORK_OPTIONS)
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

        draw_board(surface, current_example.board, ViewConsts.BOARD_POSITION[0], ViewConsts.BOARD_POSITION[1])
        draw_next_snake_direction(surface, current_example.board, self.model.get_nn_output_4directions(current_example.predictions), ViewConsts.BOARD_POSITION[0], ViewConsts.BOARD_POSITION[1])
        self.ui_manager.update(time_delta)

        self.ui_manager.draw_ui(surface)

        pygame.display.flip()

        # print(f"Model \n {np.matrix(current_example.board)} \n")
        # print(f"Current Direction : {current_example.current_direction} \n")
        # print(f"Prediction UP : {current_example.predictions[0]}")
        # print(f"Prediction DOWN : {current_example.predictions[1]}")
        # print(f"Prediction LEFT : {current_example.predictions[2]}")
        # print(f"Prediction RIGHT : {current_example.predictions[3]}")
        # print()

        # print("Enter target outputs for neural network in form")
        # print("UP=W DOWN=S LEFT=A RIGHT=D")

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

            # print(target_output)
            # print()
            self.evaluated.append(TrainingExample(current_example.board, current_example.current_direction, current_example.vision_lines, target_output))

        if len(self.training_examples) == 0 or skip:
            self.training_examples.clear()

            output_neuron_count = 4 if self.data_received["input_direction_count"] == 4 or self.data_received["input_direction_count"] == 8 else 3
            file_path = "Backpropagation_Training/" + str(self.data_received["input_direction_count"]) + "_in_directions_" + str(output_neuron_count) + "_out_directions.json"

            write_examples_to_json_4d(self.evaluated, file_path)

            # self.evaluated.clear()

            self.model.snake.brain.reinit_weights_and_biases()
            self.model = Model(self.data_received["board_size"], self.data_received["initial_snake_size"], False, self.model.snake.brain)

            train_network(self.model.snake.brain, file_path)

            self.training = False

    def run(self, surface, time_delta):
        surface.fill(self.ui_manager.ui_theme.get_colour("dark_bg"))

        if not self.training:
            self.execute(surface)
        else:
            self.train_backpropagation(surface, time_delta)
            save_neural_network_to_json(-1, -1,
                                        self.data_received["board_size"],
                                        self.data_received["initial_snake_size"],
                                        self.data_received["input_direction_count"],
                                        self.data_received["vision_return_type"],
                                        self.model.snake.brain,
                                        NNSettings.BACKPROPAGATION_NETWORK_FOLDER + self.data_received["file_name"])

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
                    self.set_target_state_name(State.BACKPROPAGATION_TRAIN_NEW_NETWORK_OPTIONS)
                    self.trigger_transition()

        self.ui_manager.update(time_delta)

        self.ui_manager.draw_ui(surface)
