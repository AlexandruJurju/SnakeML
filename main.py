import os
from typing import List

import pygame
from pygame_gui import UIManager
from scipy.spatial.distance import chebyshev

from States.backpropagation.backpropagation_menu import BackpropagationMenu
from States.backpropagation.backpropagation_train_new_network import BackpropagationTrainNewNetwork
from States.genetic.genetic_menu import GeneticMenu
from States.genetic.genetic_train_new_network import GeneticTrainNewNetwork
from States.main_menu import MainMenu
from States.options import Options
from States.run_pretrained import RunPretrained
from States.state_manager import StateManager
from game_config import ViewSettings, State
from model import Model
from neural_network import NeuralNetwork, Dense, Activation, sigmoid, relu
from vision import get_vision_lines_snake_head, VisionLine


def main():
    pygame.init()
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pygame.key.set_repeat()
    pygame.display.set_caption('Snake AI')
    screen = pygame.display.set_mode((ViewSettings.WIDTH, ViewSettings.HEIGHT))

    ui_manager = UIManager(screen.get_size(), "data/themes/ui_theme.json")
    # ui_manager.add_font_paths("jetbrainsmono", "data/fonts/JetBrainsMono-Regular.ttf", "data/fonts/JetBrainsMono-Bold.ttf", "data/fonts/JetBrainsMono-Italic.ttf", "data/fonts/JetBrainsMono-BoldItalic.ttf")
    # ui_manager.preload_fonts([{'name': 'jetbrainsmono', 'point_size': 14, 'style': 'regular'}])

    # ui_manager.add_font_paths("Montserrat",
    #                           "data/fonts/Montserrat-Regular.ttf",
    #                           "data/fonts/Montserrat-Bold.ttf",
    #                           "data/fonts/Montserrat-Italic.ttf",
    #                           "data/fonts/Montserrat-BoldItalic.ttf")
    #
    # ui_manager.preload_fonts([{'name': 'Montserrat', 'html_size': 4.5, 'style': 'bold'},
    #                           {'name': 'Montserrat', 'html_size': 4.5, 'style': 'regular'},
    #                           {'name': 'Montserrat', 'html_size': 2, 'style': 'regular'},
    #                           {'name': 'Montserrat', 'html_size': 2, 'style': 'italic'},
    #                           {'name': 'Montserrat', 'html_size': 6, 'style': 'bold'},
    #                           {'name': 'Montserrat', 'html_size': 6, 'style': 'regular'},
    #                           {'name': 'Montserrat', 'html_size': 6, 'style': 'bold_italic'},
    #                           {'name': 'Montserrat', 'html_size': 4, 'style': 'bold'},
    #                           {'name': 'Montserrat', 'html_size': 4, 'style': 'regular'},
    #                           {'name': 'Montserrat', 'html_size': 4, 'style': 'italic'}
    #                           ])

    state_manager = StateManager()
    state_manager.add_state(MainMenu(ui_manager))
    state_manager.add_state(Options(ui_manager))
    state_manager.add_state(RunPretrained(ui_manager))
    state_manager.add_state(GeneticMenu(ui_manager))
    state_manager.add_state(GeneticTrainNewNetwork(ui_manager))
    state_manager.add_state(BackpropagationMenu(ui_manager))
    state_manager.add_state(BackpropagationTrainNewNetwork(ui_manager))

    state_manager.set_initial_state(State.MAIN_MENU)

    running = True

    while running:
        running = state_manager.execute_state(screen)
    pygame.quit()


def print_vision_line(vision_line: VisionLine):
    print(f" {vision_line.direction} w_c {vision_line.wall_coord} w_d {vision_line.wall_distance} || a_c {vision_line.apple_coord} a_d {vision_line.apple_distance} || s_c {vision_line.segment_coord} s_d {vision_line.segment_distance} ")


def print_all_vision_lines(vision_lines: List[VisionLine]):
    for line in vision_lines:
        print_vision_line(line)


if __name__ == '__main__':
    net = NeuralNetwork()
    net.add_layer(Dense(16, 24))
    net.add_layer(Activation(relu, relu))
    net.add_layer(Dense(24, 4))
    net.add_layer(Activation(sigmoid, sigmoid))

    model = Model(10, 3, False, net)
    vision_lines = get_vision_lines_snake_head(model.board, model.snake.body[0], 4,
                                               max_dist=-1, apple_return_type="boolean", segment_return_type="boolean", distance_function=chebyshev)

    print(model.board)
    print_all_vision_lines(vision_lines)

    main()

    # net = NeuralNetwork()
    # net.add_layer(Dense(5, 10))
    # net.add_layer(Activation(sigmoid, sigmoid))
    #
    # weights_1 = net.get_dense_layers()[0].weights
    #
    # net.reinit_weights_and_biases()
    #
    # weights_2 = net.get_dense_layers()[0].weights
    #
    # print(weights_1)
    # print("\n")
    # print(weights_2)
    # print("========")
    #
    # child1 = gaussian_mutation(weights_1, 0.05)
    # print(child1)
    # print("\n")
