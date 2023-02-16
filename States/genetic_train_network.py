from typing import List

import numpy as np
import pygame
import pygame_gui
from pygame_gui import UIManager
from pygame_gui.elements import UILabel, UIButton

from States.base_state import BaseState
from States.state_manager import StateManager
from constants import State
from genetic_operators import elitist_selection, roulette_selection, full_mutation, full_crossover
from model import Snake, Model
from neural_network import NeuralNetwork, Dense, Activation, tanh, tanh_prime, sigmoid, sigmoid_prime
from settings import BoardSettings, SnakeSettings, GeneticSettings, NNSettings
from train_network import save_neural_network_to_json
from vision import get_vision_lines


class GeneticTrainNetwork(BaseState):
    def __init__(self, state_manager: StateManager, ui_manager: UIManager):
        super().__init__(State.GENETIC_TRAIN_NETWORK, state_manager)

        self.ui_manager = ui_manager
        self.model = None
        self.generation = None
        self.parent_list: List[Snake] = []
        self.offspring_list: List[NeuralNetwork] = []

        self.title_label = None
        self.button_back = None

    def start(self):
        self.title_label = UILabel(pygame.Rect((87, 40), (800, 25)), "Training Genetic Network", self.ui_manager, object_id="#window_label")
        self.button_back = UIButton(pygame.Rect((25, 725), (125, 35)), "BACK", self.ui_manager)

        self.generation = 0
        self.parent_list: List[Snake] = []
        self.offspring_list: List[NeuralNetwork] = []

        net = NeuralNetwork()
        net.add_layer(Dense(NNSettings.INPUT_NEURON_COUNT, NNSettings.HIDDEN_NEURON_COUNT))
        net.add_layer(Activation(tanh, tanh_prime))
        net.add_layer(Dense(NNSettings.HIDDEN_NEURON_COUNT, NNSettings.OUTPUT_NEURON_COUNT))
        net.add_layer(Activation(sigmoid, sigmoid_prime))

        model = Model(BoardSettings.BOARD_SIZE, SnakeSettings.START_SNAKE_SIZE, net)

        self.model = model

    def end(self):
        self.title_label.kill()
        self.button_back.kill()

    def run_genetic(self, surface):
        # self.window.fill(ViewConsts.COLOR_BACKGROUND)

        vision_lines = get_vision_lines(self.model.board)
        neural_net_prediction = self.model.get_nn_output(vision_lines)

        # draw_board(surface, self.model.board, 350, 100)
        # self.draw_vision_lines(self.model, vision_lines)
        # self.draw_neural_network(self.model, vision_lines, nn_input, neural_net_prediction)
        # self.write_ttl(self.model.snake.ttl)
        # self.write_score(self.model.snake.score)

        next_direction = self.model.get_nn_output_4directions(neural_net_prediction)
        is_alive = self.model.move_in_direction(next_direction)

        if not is_alive:
            self.model.snake.calculate_fitness()
            self.parent_list.append(self.model.snake)

            if self.generation == 0:
                self.model.snake.brain.reinit_weights_and_biases()
                self.model = Model(BoardSettings.BOARD_SIZE, SnakeSettings.START_SNAKE_SIZE, self.model.snake.brain)
            else:
                self.model = Model(BoardSettings.BOARD_SIZE, SnakeSettings.START_SNAKE_SIZE, self.offspring_list[len(self.parent_list) - 1])

            if len(self.parent_list) == GeneticSettings.POPULATION_COUNT:
                self.offspring_list.clear()
                self.next_generation()

    def next_generation(self):
        self.offspring_list.clear()

        # total_fitness = sum(individual.fitness for individual in self.parent_list)
        best_individual = max(self.parent_list, key=lambda individual: individual.fitness)

        save_neural_network_to_json(self.generation, best_individual.fitness, best_individual.brain)

        print(f"GEN {self.generation + 1}   BEST FITNESS : {best_individual.fitness}")

        parents_for_mating = elitist_selection(self.parent_list, 500)
        np.random.shuffle(parents_for_mating)

        while len(self.offspring_list) < GeneticSettings.POPULATION_COUNT:
            parent1, parent2 = roulette_selection(parents_for_mating, 2)
            child1, child2 = full_crossover(parent1.brain, parent1.brain)

            full_mutation(child1)
            full_mutation(child2)

            self.offspring_list.append(child1)
            self.offspring_list.append(child2)

        self.model.snake.brain.reinit_weights_and_biases()
        self.model = Model(BoardSettings.BOARD_SIZE, SnakeSettings.START_SNAKE_SIZE, self.offspring_list[0])

        self.generation += 1
        self.parent_list.clear()

    def run(self, surface, time_delta):
        surface.fill(self.ui_manager.ui_theme.get_colour("dark_bg"))

        self.run_genetic(surface)

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
                    self.set_target_state_name(State.GENETIC_TRAIN_NETWORK_OPTIONS)
                    self.trigger_transition()

        # self.ui_manager.update(time_delta)

        # self.ui_manager.draw_ui(surface)
