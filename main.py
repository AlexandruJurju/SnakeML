import os

import pygame
from pygame_gui import UIManager

from States.backpropagation.backpropagation_menu import BackpropagationMenu
from States.backpropagation.backpropagation_train_new_network import BackpropagationTrainNewNetwork
from States.genetic.genetic_menu import GeneticMenu
from States.genetic.genetic_train_new_network import GeneticTrainNewNetwork
from States.main_menu import MainMenu
from States.options import Options
from States.run_pretrained import RunPretrained
from States.state_manager import StateManager
from game_config import ViewSettings, State
from genetic_operators import two_point_crossover
from neural_network import NeuralNetwork, Dense, Activation, sigmoid


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
    state_manager.add_state(MainMenu(state_manager, ui_manager))
    state_manager.add_state(Options(state_manager, ui_manager))
    state_manager.add_state(RunPretrained(state_manager, ui_manager))
    state_manager.add_state(GeneticMenu(state_manager, ui_manager))
    state_manager.add_state(GeneticTrainNewNetwork(state_manager, ui_manager))
    state_manager.add_state(BackpropagationMenu(state_manager, ui_manager))
    state_manager.add_state(BackpropagationTrainNewNetwork(state_manager, ui_manager))

    state_manager.set_initial_state(State.MAIN_MENU)

    running = True

    while running:
        running = state_manager.run(screen)
    pygame.quit()


if __name__ == '__main__':
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
    # child1, child2 = two_point_crossover(weights_1, weights_2)
    # print(child1)
    # print("\n")
    # print(child2)
    # print("===========")
