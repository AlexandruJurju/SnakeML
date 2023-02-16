import copy

import pygame
import pygame_gui
from pygame_gui import UIManager
from pygame_gui.elements import UILabel, UIButton

from States.base_state import BaseState
from States.state_manager import StateManager
from constants import State
from model import Model
from neural_network import *
from settings import NNSettings, BoardSettings, SnakeSettings
from train_network import TrainingExample, write_examples_to_json_4d, train_network
from view import draw_board, draw_next_snake_direction
from vision import get_vision_lines


class BackpropagationTrainNewNetwork(BaseState):
    def __init__(self, state_manager: StateManager, ui_manager: UIManager):
        super().__init__(State.BACKPROPAGATION_TRAIN_NEW_NETWORK, state_manager)

        self.ui_manager = ui_manager
        self.model = None

        self.training_examples = []
        self.evaluated = []

        self.training = False

        self.title_label = None
        self.button_back = None

    def start(self):
        self.title_label = UILabel(pygame.Rect((87, 40), (800, 25)), "Training Genetic Network", self.ui_manager, object_id="#window_label")
        self.button_back = UIButton(pygame.Rect((25, 725), (125, 35)), "BACK", self.ui_manager)
        net = NeuralNetwork()
        net.add_layer(Dense(NNSettings.INPUT_NEURON_COUNT, NNSettings.HIDDEN_NEURON_COUNT))
        net.add_layer(Activation(tanh, tanh_prime))
        net.add_layer(Dense(NNSettings.HIDDEN_NEURON_COUNT, NNSettings.OUTPUT_NEURON_COUNT))
        net.add_layer(Activation(sigmoid, sigmoid_prime))

        self.model = Model(BoardSettings.BOARD_SIZE, SnakeSettings.START_SNAKE_SIZE, net)

    def end(self):
        self.title_label.kill()
        self.button_back.kill()

    def execute(self, surface):
        vision_lines = get_vision_lines(self.model.board)
        nn_output = self.model.get_nn_output(vision_lines)

        example_output = np.where(nn_output == np.max(nn_output), 1, 0)
        example = TrainingExample(copy.deepcopy(self.model.board), self.model.snake.direction, vision_lines, example_output.ravel().tolist())
        self.training_examples.append(example)

        draw_board(surface, self.model.board, 500, 300)
        # self.draw_vision_lines(self.model, vision_lines)
        # self.draw_neural_network(self.model, vision_lines)
        # self.write_ttl(self.model.snake.ttl)
        # self.write_score(self.model.snake.score)

        next_direction = self.model.get_nn_output_4directions(nn_output)
        is_alive = self.model.move_in_direction(next_direction)
        if not is_alive:
            self.training = True

    @staticmethod
    def wait_for_key() -> str:
        while True:
            for event in pygame.event.get():
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

    def train_backpropagation(self, surface):
        current_example = self.training_examples[0]
        self.training_examples.pop(0)

        draw_board(surface, current_example.board, 500, 300)
        # TODO draw next direction doesnt work for other offsets
        draw_next_snake_direction(surface, current_example.board, self.model.get_nn_output_4directions(current_example.predictions), 500, 300)
        pygame.display.flip()

        print(f"Model \n {np.matrix(current_example.board)} \n")
        print(f"Current Direction : {current_example.current_direction} \n")
        print(f"Prediction UP : {current_example.predictions[0]}")
        print(f"Prediction DOWN : {current_example.predictions[1]}")
        print(f"Prediction LEFT : {current_example.predictions[2]}")
        print(f"Prediction RIGHT : {current_example.predictions[3]}")
        print()

        print("Enter target outputs for neural network in form")
        print("UP=W DOWN=S LEFT=A RIGHT=D")

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

            print(target_output)
            print()
            self.evaluated.append(TrainingExample(copy.deepcopy(current_example.board), current_example.current_direction, current_example.vision_lines, target_output))

        if len(self.training_examples) == 0 or skip:
            self.training_examples.clear()
            write_examples_to_json_4d(self.evaluated)

            self.evaluated.clear()

            # TODO BAD REINIT
            self.model.snake.brain.reinit_weights_and_biases()
            # TODO add reinit function in model
            self.model = Model(BoardSettings.BOARD_SIZE, SnakeSettings.START_SNAKE_SIZE, self.model.snake.brain)

            train_network(self.model.snake.brain)

            self.training = False

    def run(self, surface, time_delta):
        surface.fill(self.ui_manager.ui_theme.get_colour("dark_bg"))

        if not self.training:
            self.execute(surface)
        else:
            self.train_backpropagation(surface)

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
