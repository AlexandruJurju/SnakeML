import pygame_gui
from matplotlib import pyplot as plt
from pygame_gui import UIManager
from pygame_gui.elements import UILabel, UIButton

import genetic_operators
import neural_network
import vision
from States.base_state import BaseState
from file_operations import save_neural_network_to_json, write_genetic_training
from game_config import GameSettings
from game_config import State
from genetic_operators import elitist_selection, full_mutation, full_crossover
from model import Snake
from neural_network import NeuralNetwork, Activation
from view import *
from vision import get_vision_lines_snake_head


# TODO make hidden layers count
class GeneticTrainNewNetwork(BaseState):
    def __init__(self, ui_manager: UIManager):
        super().__init__(State.GENETIC_TRAIN_NEW_NETWORK)

        self.crossover_operator = None
        self.distance_function = None
        self.apple_return_type = None
        self.segment_return_type = None
        self.selection_operator = None
        self.mutation_operator = None
        self.file_name = None
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
        self.max_distance = None

        self.x_generations = []
        self.y_best_individual_fitness = []
        self.y_best_individual_score = []
        self.y_average_score = []
        self.y_best_ratio = []
        self.networks = []
        self.training_data = ""

        self.title_label = None
        self.button_back = None
        self.generation_label = None
        self.individual_label = None

        self.button_draw_network = None
        self.draw_network = False
        self.rect_draw_network = None

        self.button_draw_vision_lines = None
        self.draw_vision_lines = True
        self.rect_draw_vision_lines = None

        self.button_stop_drawing = None
        self.draw_switch = True

    def start(self):
        self.initial_board_size = self.data_received["board_size"]
        self.initial_snake_size = self.data_received["initial_snake_size"]
        self.input_direction_count = self.data_received["input_direction_count"]
        self.segment_return_type = self.data_received["segment_return_type"]
        self.apple_return_type = self.data_received["apple_return_type"]
        self.distance_function = getattr(vision, self.data_received["distance_function"])
        self.population_count = self.data_received["population_count"]
        self.selection_operator = getattr(genetic_operators, self.data_received["selection_operator"])
        self.crossover_operator = getattr(genetic_operators, self.data_received["crossover_operator"])
        self.mutation_operator = getattr(genetic_operators, self.data_received["mutation_operator"])
        self.mutation_rate = self.data_received["mutation_rate"]
        self.file_name = self.data_received["file_name"]

        hidden_layer_count = self.data_received["hidden_layer_count"]
        input_neuron_count = self.data_received["input_layer_neurons"]
        hidden_layer1_neuron_count = self.data_received["hidden_layer1_neuron_count"]
        hidden_layer2_neuron_count = self.data_received["hidden_layer2_neuron_count"]
        hidden_layer3_neuron_count = self.data_received["hidden_layer3_neuron_count"]
        output_neuron_count = self.data_received["output_layer_neurons"]

        hidden_activation = getattr(neural_network, self.data_received["hidden_activation"])
        output_activation = getattr(neural_network, self.data_received["output_activation"])

        net = NeuralNetwork()
        net.add_layer(Dense(input_neuron_count, hidden_layer1_neuron_count))
        net.add_layer(Activation(hidden_activation, hidden_activation))

        hidden_neurons = [hidden_layer1_neuron_count, hidden_layer2_neuron_count, hidden_layer3_neuron_count]

        for i in range(hidden_layer_count):
            if i == hidden_layer_count - 1:
                net.add_layer(Dense(hidden_neurons[i], output_neuron_count))
                net.add_layer(Activation(output_activation, output_activation))
            else:
                net.add_layer(Dense(hidden_neurons[i], hidden_neurons[i + 1]))
                net.add_layer(Activation(hidden_activation, hidden_activation))

        self.x_generations = []
        self.y_best_individual_fitness = []
        self.y_best_individual_score = []
        self.y_average_score = []
        self.y_best_ratio = []
        self.networks = []
        self.training_data = ""

        self.title_label = UILabel(pygame.Rect(ViewSettings.TITLE_LABEL_POSITION, ViewSettings.TITLE_LABEL_DIMENSION), "Genetic Train New Network", self.ui_manager)
        self.button_back = UIButton(pygame.Rect(ViewSettings.BUTTON_BACK_POSITION, ViewSettings.BUTTON_BACK_DIMENSION), "BACK", self.ui_manager)

        self.button_draw_network = UIButton(pygame.Rect((50, 175), (175, 30)), "Draw Network", self.ui_manager)
        self.rect_draw_network = pygame.Rect((250, 175), (30, 30))

        self.button_draw_vision_lines = UIButton(pygame.Rect((50, 250), (175, 30)), "Draw Vision Lines", self.ui_manager)
        self.rect_draw_vision_lines = pygame.Rect((250, 250), (30, 30))

        self.button_stop_drawing = UIButton(pygame.Rect((50, 450), (175, 30)), "Stop Drawing", self.ui_manager)

        self.generation_label = UILabel(pygame.Rect((50, 50), (150, 25)), "Population: ", self.ui_manager)
        self.individual_label = UILabel(pygame.Rect((50, 100), (200, 25)), "Individual: ", self.ui_manager)

        self.generation = 0
        self.parent_list: List[Snake] = []
        self.offspring_list: List[NeuralNetwork] = []

        self.model = Model(self.initial_board_size, self.initial_snake_size, True, net)

        self.max_distance = self.distance_function((0, 1), (self.initial_board_size, 1))
        print(self.max_distance)

    def end(self):
        self.ui_manager.clear_and_reset()

    def run_genetic(self, surface):
        vision_lines = get_vision_lines_snake_head(self.model.board, self.model.snake.body[0], self.input_direction_count,
                                                   max_dist=self.max_distance, apple_return_type=self.apple_return_type, segment_return_type=self.segment_return_type, distance_function=self.distance_function)
        neural_net_prediction = self.model.get_nn_output(vision_lines)

        if ViewSettings.DRAW:
            draw_board(surface, self.model.board, ViewSettings.BOARD_POSITION[0], ViewSettings.BOARD_POSITION[1])

            if self.draw_vision_lines:
                draw_vision_lines(surface, self.model.snake.body[0], vision_lines, ViewSettings.BOARD_POSITION[0], ViewSettings.BOARD_POSITION[1])

            if self.draw_network:
                draw_neural_network_complete(surface, self.model, vision_lines, ViewSettings.NN_POSITION[0], ViewSettings.NN_POSITION[1])

        next_direction = self.model.get_nn_output_4directions(neural_net_prediction)
        is_alive = self.model.move(next_direction)

        if not is_alive:
            self.model.snake.calculate_fitness()
            self.parent_list.append(self.model.snake)

            if self.generation == 0:
                self.model.snake.brain.reinit_weights_and_biases()
                self.model = Model(self.initial_board_size, self.initial_snake_size, True, self.model.snake.brain)
            else:
                self.model = Model(self.initial_board_size, self.initial_snake_size, True, self.offspring_list[len(self.parent_list) - 1])

    def next_generation(self):
        self.offspring_list = []

        total_fitness = sum(individual.fitness for individual in self.parent_list)
        # best_individual = max(self.parent_list, key=lambda individual: (individual.fitness, individual.score / individual.steps_taken))
        # best_individual = max(self.parent_list, key=lambda individual: (individual.fitness, individual.score / individual.steps_taken if individual.steps_taken != 0 else 0))
        best_individual = max(self.parent_list, key=lambda individual: (individual.score, individual.score / individual.steps_taken if individual.steps_taken != 0 else 0))

        # TODO change order
        # best_individual = max(self.parent_list, key=lambda individual: (individual.fitness, individual.score / individual.steps_taken))

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
        average_score = apple_count / len(self.parent_list)
        average_fitness = total_fitness / self.population_count

        name = "Generation" + str(self.generation)

        data_to_save = {
            "generation": self.generation,
            "initial_board_size": self.initial_board_size,
            "initial_snake_size": self.initial_snake_size,
            "input_direction_count": self.input_direction_count,
            "apple_return_type": self.apple_return_type,
            "segment_return_type": self.segment_return_type,
            "distance_function": self.data_received["distance_function"]
        }

        self.networks.append([data_to_save, best_individual.brain, GameSettings.GENETIC_NETWORK_FOLDER + "/" + self.file_name + "/" + name])

        training_data = (f"GEN: {self.generation + 1:<5} "
                         f"AVG FITNESS: {average_fitness:<25}\t"
                         f"AVG SCORE: {average_score:<10}\t"
                         f"AVG RATIO: {(apple_count / len(self.parent_list)) / (steps_taken / len(self.parent_list)):<25}\t"
                         f"BEST FITNESS: {best_individual.fitness:<25}\t"
                         f"BEST SCORE: {best_individual.score:<5}\t"
                         f"BEST RATIO: {best_individual.score / best_individual.steps_taken if best_individual.steps_taken > 0 else 0:<25}"
                         f"TOO_OLD: {too_old:<8}\t"
                         f"WON: {won_count:<5}\t"
                         )
        print(training_data)
        self.training_data += training_data + "\n"

        self.x_generations.append(self.generation)
        self.y_best_individual_fitness.append(best_individual.fitness)
        self.y_best_individual_score.append(best_individual.score)
        self.y_average_score.append(average_score)
        self.y_best_ratio.append(best_individual.score / best_individual.steps_taken)

        parents_for_mating = elitist_selection(self.parent_list, 100)
        for parent in parents_for_mating[:100]:
            self.offspring_list.append(parent.brain)

        # np.random.shuffle(parents_for_mating)
        # np.random.shuffle(self.parent_list)

        while len(self.offspring_list) < self.population_count:
            parent1, parent2 = self.selection_operator(self.parent_list, 2)
            child1, child2 = full_crossover(parent1.brain, parent1.brain, self.crossover_operator)

            full_mutation(child1, self.mutation_rate, self.mutation_operator)
            full_mutation(child2, self.mutation_rate, self.mutation_operator)

            self.offspring_list.append(child1)
            self.offspring_list.append(child2)

        # self.model.snake.brain.reinit_weights_and_biases()
        self.model = Model(self.initial_board_size, self.initial_snake_size, True, self.offspring_list[0])

        self.generation += 1
        self.parent_list = []

    def run(self, surface, time_delta):
        # FILL TAKES ALOT OF TIME
        if ViewSettings.DRAW:
            surface.fill(self.ui_manager.ui_theme.get_colour("dark_bg"))
            pygame.draw.rect(surface, ViewSettings.COLOR_GREEN if self.draw_network else ViewSettings.COLOR_RED, self.rect_draw_network)
            pygame.draw.rect(surface, ViewSettings.COLOR_GREEN if self.draw_vision_lines else ViewSettings.COLOR_RED, self.rect_draw_vision_lines)
            self.generation_label.set_text("Generation : " + str(self.generation))
            self.individual_label.set_text("Individual : " + str(len(self.parent_list)) + " / " + str(self.population_count))

        if len(self.parent_list) == self.population_count:
            self.next_generation()
        else:
            self.run_genetic(surface)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.set_target_state_name(State.QUIT)
                self.trigger_transition()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    for network in self.networks[-50:]:
                        save_neural_network_to_json(
                            network[0],
                            network[1],
                            network[2]
                        )

                    write_genetic_training(self.training_data, GameSettings.GENETIC_NETWORK_FOLDER + "/" + self.file_name, True if self.generation == 0 else False)

                    fig1 = plt.figure(figsize=(16, 9))
                    plt.plot(self.x_generations, self.y_best_individual_score, "b", label="Best Individual Score")
                    plt.plot(self.x_generations, self.y_average_score, "r", label="Generation Mean Score")
                    plt.legend(loc="upper left")
                    plt.xlabel("Generation")
                    plt.ylabel("Score")
                    plt.title("Score Comparison")
                    plt.savefig(GameSettings.GENETIC_NETWORK_FOLDER + "/" + self.file_name + "/" + "plot.pdf")
                    plt.show()

                    fig2 = plt.figure(figsize=(16, 9))
                    plt.plot(self.x_generations, self.y_best_ratio, "b")
                    plt.xlabel("Generations")
                    plt.ylabel("Best Individual Ratio")
                    plt.title("Ratio Progression")
                    plt.savefig(GameSettings.GENETIC_NETWORK_FOLDER + "/" + self.file_name + "/" + "best_ratio.pdf")
                    plt.show()

                    self.set_target_state_name(State.QUIT)
                    self.trigger_transition()

                if event.key == pygame.K_RETURN:
                    ViewSettings.DRAW = True

            self.ui_manager.process_events(event)

            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.button_back:
                    self.set_target_state_name(State.OPTIONS)
                    self.data_to_send = {
                        "state": "genetic"
                    }
                    self.trigger_transition()

                if event.ui_element == self.button_draw_network:
                    self.draw_network = not self.draw_network

                if event.ui_element == self.button_draw_vision_lines:
                    self.draw_vision_lines = not self.draw_vision_lines

                if event.ui_element == self.button_stop_drawing:
                    surface.fill(self.ui_manager.ui_theme.get_colour("dark_bg"))

                    font = pygame.font.SysFont("Arial", 20)
                    text_line = "PRESS ENTER TO TURN ON DRAWING"
                    text_surface = font.render(text_line, True, (255, 255, 255))
                    text_rect = text_surface.get_rect()
                    text_rect.center = (ViewSettings.X_CENTER, ViewSettings.Y_CENTER)
                    surface.blit(text_surface, text_rect)

                    ViewSettings.DRAW = False
                    self.ui_manager.update(time_delta)
                    self.ui_manager.draw_ui(surface)
                    pygame.display.flip()

        if ViewSettings.DRAW:
            self.ui_manager.update(time_delta)
            self.ui_manager.draw_ui(surface)
