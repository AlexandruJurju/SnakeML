import pygame_gui
from pygame_gui import UIManager
from pygame_gui.elements import UILabel, UIButton

import neural_network
from States.base_state import BaseState
from States.state_manager import StateManager
from file_operations import save_neural_network_to_json
from game_config import GameSettings
from game_config import State
from genetic_operators import elitist_selection, roulette_selection, full_mutation, full_crossover
from model import Snake
from neural_network import NeuralNetwork, Activation
from view import *
from vision import get_vision_lines_snake_head


class GeneticTrainNewNetwork(BaseState):
    def __init__(self, state_manager: StateManager, ui_manager: UIManager):
        super().__init__(State.GENETIC_TRAIN_NEW_NETWORK, state_manager)

        self.mutation_rate = None
        self.population_count = None
        self.vision_return_type = None
        self.input_direction_count = None
        self.initial_snake_size = None
        self.initial_board_size = None
        self.ui_manager = ui_manager
        self.model = None
        self.generation = None
        self.parent_list: List[Snake] = []
        self.offspring_list: List[NeuralNetwork] = []

        self.x_points = []
        self.y_points = []

        self.title_label = None
        self.button_back = None
        self.generation_label = None
        self.individual_label = None

    def start(self):
        self.initial_board_size = self.data_received["board_size"]
        self.initial_snake_size = self.data_received["initial_snake_size"]
        self.input_direction_count = self.data_received["input_direction_count"]
        self.vision_return_type = self.data_received["vision_return_type"]
        self.population_count = self.data_received["population_count"]
        self.mutation_rate = self.data_received["mutation_rate"]

        self.title_label = UILabel(pygame.Rect(ViewSettings.TITLE_LABEL_POSITION, ViewSettings.TITLE_LABEL_DIMENSION), "Genetic Train New Network", self.ui_manager, object_id="#window_label")
        self.button_back = UIButton(pygame.Rect(ViewSettings.BUTTON_BACK_POSITION, ViewSettings.BUTTON_BACK_DIMENSION), "BACK", self.ui_manager)

        self.generation_label = UILabel(pygame.Rect((45, 50), (150, 25)), "Population :", self.ui_manager)
        self.individual_label = UILabel(pygame.Rect((50, 100), (200, 25)), "Individual :", self.ui_manager)

        self.generation = 0
        self.parent_list: List[Snake] = []
        self.offspring_list: List[NeuralNetwork] = []

        input_neuron_count = self.data_received["input_layer_neurons"]
        hidden_neuron_count = self.data_received["hidden_layer_neurons"]
        output_neuron_count = self.data_received["output_layer_neurons"]

        hidden_activation = getattr(neural_network, self.data_received["hidden_activation"])
        output_activation = getattr(neural_network, self.data_received["output_activation"])

        # activation prime doesn't matter in feedforward, use base activation functions to avoid error
        net = NeuralNetwork()
        net.add_layer(Dense(input_neuron_count, hidden_neuron_count))
        net.add_layer(Activation(hidden_activation, hidden_activation))
        net.add_layer(Dense(hidden_neuron_count, output_neuron_count))
        net.add_layer(Activation(output_activation, output_activation))

        self.model = Model(self.initial_board_size, self.initial_snake_size, True, net)

    def end(self):
        self.title_label.kill()
        self.button_back.kill()
        self.generation_label.kill()
        self.individual_label.kill()

    def run_genetic(self, surface):
        vision_lines = get_vision_lines_snake_head(self.model.board, self.model.snake.body[0], self.input_direction_count, self.vision_return_type)
        neural_net_prediction = self.model.get_nn_output(vision_lines)

        if ViewSettings.DRAW:
            draw_board(surface, self.model.board, ViewSettings.BOARD_POSITION[0], ViewSettings.BOARD_POSITION[1])
            draw_vision_lines(surface, self.model, vision_lines, ViewSettings.BOARD_POSITION[0], ViewSettings.BOARD_POSITION[1])
            draw_neural_network_complete(surface, self.model, vision_lines, ViewSettings.NN_POSITION[0], ViewSettings.NN_POSITION[1])

        next_direction = self.model.get_nn_output_4directions(neural_net_prediction)
        is_alive = self.model.move(next_direction)

        if not is_alive:
            self.model.snake.calculate_fitness(self.model.max_score)
            self.parent_list.append(self.model.snake)

            if self.generation == 0:
                self.model.snake.brain.reinit_weights_and_biases()
                self.model = Model(self.initial_board_size, self.initial_snake_size, True, self.model.snake.brain)
            else:
                self.model = Model(self.initial_board_size, self.initial_snake_size, True, self.offspring_list[len(self.parent_list) - 1])

            if len(self.parent_list) == self.population_count:
                self.offspring_list.clear()
                self.next_generation()

    # TODO decaying mutation rate
    # TODO write console output in a txt file to store generation information
    def next_generation(self):
        self.offspring_list.clear()

        total_fitness = sum(individual.fitness for individual in self.parent_list)
        best_individual = max(self.parent_list, key=lambda individual: individual.fitness)

        counts = {'won': 0, 'apple_count': 0, 'too_old': 0, 'steps_taken': 0}
        for individual in self.parent_list:
            counts['apple_count'] += individual.score
            counts['steps_taken'] += individual.steps_taken
            if individual.won:
                counts['won'] += 1
            if individual.TTL == 0:
                counts['too_old'] += 1

        won_count = counts['won']
        apple_count = counts['apple_count']
        too_old = counts['too_old']
        steps_taken = counts['steps_taken']

        save_neural_network_to_json(self.generation,
                                    best_individual.fitness,
                                    self.initial_board_size,
                                    self.initial_snake_size,
                                    self.input_direction_count,
                                    self.vision_return_type,
                                    best_individual.brain,
                                    GameSettings.GENETIC_NETWORK_FOLDER + "/" + self.data_received["file_name"] + "/" + str(self.generation % 5))

        # print(f"GEN {self.generation + 1}   BEST FITNESS : {best_individual.fitness}")

        print(f"GEN {self.generation + 1:<5}   MEAN FITNESS : {total_fitness / self.population_count:<25}\t WON : {won_count:^5}\t TOO_OLD : {too_old}\t MEAN_SCORE {apple_count / len(self.parent_list):<25}\t "
              f"RATIO {(apple_count / len(self.parent_list)) / (steps_taken / len(self.parent_list))}")

        # self.x_points.append(self.generation)
        # self.y_points.append(best_individual.fitness)

        parents_for_mating = elitist_selection(self.parent_list, 100)
        for parent in parents_for_mating[:100]:
            self.offspring_list.append(parent.brain)

        # np.random.shuffle(parents_for_mating)
        np.random.shuffle(self.parent_list)

        while len(self.offspring_list) < self.population_count:
            parent1, parent2 = roulette_selection(self.parent_list, 2)
            child1, child2 = full_crossover(parent1.brain, parent1.brain)

            full_mutation(child1, self.mutation_rate)
            full_mutation(child2, self.mutation_rate)

            self.offspring_list.append(child1)
            self.offspring_list.append(child2)

        self.model.snake.brain.reinit_weights_and_biases()
        self.model = Model(self.initial_board_size, self.initial_snake_size, True, self.offspring_list[0])

        self.generation += 1
        self.parent_list.clear()

        # plt.plot(self.x_points, self.y_points)
        # plt.xlabel("Generation")
        # plt.ylabel("Fitness")
        # plt.show()

    def run(self, surface, time_delta):
        # FILL TAKES ALOT OF TIME
        if ViewSettings.DRAW:
            surface.fill(self.ui_manager.ui_theme.get_colour("dark_bg"))

        self.run_genetic(surface)
        self.generation_label.set_text("Generation : " + str(self.generation))
        self.individual_label.set_text("Individual : " + str(len(self.parent_list)) + " / " + str(self.population_count))

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
                        "state": "genetic"
                    }
                    self.trigger_transition()

        if ViewSettings.DRAW:
            self.ui_manager.update(time_delta)
            self.ui_manager.draw_ui(surface)
