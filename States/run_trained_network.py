import pygame
import pygame_gui
from pygame_gui import UIManager
from pygame_gui.elements import UILabel, UIButton

from States.base_state import BaseState
from States.state_manager import StateManager
from constants import BoardConsts
from model import Model
from neural_network import NeuralNetwork, Dense, Activation, tanh, tanh_prime, sigmoid_prime, sigmoid
from settings import SnakeSettings, NNSettings
from train_network import read_neural_network_from_json
from view import draw_board
from vision import get_vision_lines


class RunTrainedGeneticNetwork(BaseState):
    def __init__(self, state_manager: StateManager, ui_manager: UIManager):
        super().__init__("run_trained_genetic_network", "quit", state_manager)

        # TODO BAD model initialization
        net = NeuralNetwork()
        net.add_layer(Dense(NNSettings.INPUT_NEURON_COUNT, NNSettings.HIDDEN_NEURON_COUNT))
        net.add_layer(Activation(tanh, tanh_prime))
        net.add_layer(Dense(NNSettings.HIDDEN_NEURON_COUNT, NNSettings.OUTPUT_NEURON_COUNT))
        net.add_layer(Activation(sigmoid, sigmoid_prime))

        model = Model(BoardConsts.BOARD_SIZE, SnakeSettings.START_SNAKE_SIZE, net)

        self.model = model

        self.ui_manager = ui_manager

        self.title_label = None
        self.button_back = None

    def start(self):
        self.title_label = UILabel(pygame.Rect((87, 40), (800, 50)), "Trained Genetic Network", self.ui_manager, object_id="#window_label")
        self.button_back = UIButton(pygame.Rect((50, 100), (150, 35)), "BACK", self.ui_manager)

    def end(self):
        self.title_label.kill()
        self.button_back.kill()

    def run(self, surface, time_delta):
        surface.fill(self.ui_manager.ui_theme.get_colour("dark_bg"))

        self.model.snake.brain = read_neural_network_from_json()

        vision_lines = get_vision_lines(self.model.board)
        neural_net_prediction = self.model.get_nn_output(vision_lines)

        draw_board(surface, self.model.board, 500, 300)

        next_direction = self.model.get_nn_output_4directions(neural_net_prediction)
        is_alive = self.model.move_in_direction(next_direction)

        if not is_alive:
            self.model = Model(BoardConsts.BOARD_SIZE, SnakeSettings.START_SNAKE_SIZE, self.model.snake.brain)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.set_target_state_name("quit")
                self.trigger_transition()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.set_target_state_name("quit")
                    self.trigger_transition()

            self.ui_manager.process_events(event)

            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.button_back:
                    self.set_target_state_name("menu_genetic")
                    self.trigger_transition()

        self.ui_manager.update(time_delta)

        self.ui_manager.draw_ui(surface)
