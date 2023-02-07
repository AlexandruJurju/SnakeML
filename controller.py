import pygame

from Neural.train_model import *
from Neural.train_model import train_network
from model import *
from view import View


class Controller:
    def __init__(self, model: Model, view: View):
        self.running = True
        self.model = model
        self.view = view

    def run(self):
        training_examples = []
        while self.running:
            self.view.clear_window()

            vision_lines = Vision.get_dynamic_vision_lines(self.model.board, self.model.snake.direction)
            self.view.draw_board(self.model)
            self.view.draw_vision_lines(self.model, vision_lines)

            neural_net_prediction = self.model.get_nn_output(vision_lines)
            nn_input = Vision.get_parameters_in_nn_input_form(vision_lines, self.model.snake.direction)
            self.view.draw_neural_network(self.model, vision_lines, nn_input, neural_net_prediction)

            example = TrainingExample(copy.deepcopy(self.model.board), neural_net_prediction, self.model.snake.direction)
            training_examples.append(example)

            next_direction = self.model.get_nn_output_3directions_dynamic(neural_net_prediction)
            self.running = self.model.move_in_direction(next_direction)

            if not self.running:
                self.view.draw_dead(self.model)
                pygame.display.update()

                evaluate_live_examples(training_examples)
                training_examples = []

                # TODO BAD REINIT, TO BE REMOVED
                # TODO train data , search file like a dictionary to find if there are conflicting data
                self.model.snake.brain.reinit_weights_and_biases()
                train_network(self.model.snake.brain)

                self.model = Model(10, 3, self.model.snake.brain)

                self.running = True

            self.view.update_window()
