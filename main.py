import os

import pygame
from pygame_gui import UIManager

from States.backpropagation.backprop_menu import BackpropMenu
from States.backpropagation.backprop_pretrained_network import BackpropPretrainedNetwork
from States.backpropagation.backprop_train_new_network import BackpropTrainNewNetwork
from States.backpropagation.backprop_train_new_network_options import BackpropTrainNewNetworkOptions
from States.genetic.genetic_menu import MenuGenetic
from States.genetic.genetic_pretrained_network import GeneticPretrainedNetwork
from States.genetic.genetic_train_new_network import GeneticTrainNewNetwork
from States.genetic.genetic_train_new_network_options import GeneticTrainNetworkOptions
from States.main_menu import MainMenu
from States.state_manager import StateManager
from game_config import ViewConsts, State


def main():
    pygame.init()
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pygame.key.set_repeat()
    pygame.display.set_caption('Snake AI')
    screen = pygame.display.set_mode((ViewConsts.WIDTH, ViewConsts.HEIGHT))

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
    state_manager.add_state(MenuGenetic(state_manager, ui_manager))
    state_manager.add_state(GeneticPretrainedNetwork(state_manager, ui_manager))
    state_manager.add_state(GeneticTrainNetworkOptions(state_manager, ui_manager))
    state_manager.add_state(GeneticTrainNewNetwork(state_manager, ui_manager))
    state_manager.add_state(BackpropMenu(state_manager, ui_manager))
    state_manager.add_state(BackpropPretrainedNetwork(state_manager, ui_manager))
    state_manager.add_state(BackpropTrainNewNetworkOptions(state_manager, ui_manager))
    state_manager.add_state(BackpropTrainNewNetwork(state_manager, ui_manager))

    state_manager.set_initial_state(State.GENETIC_TRAIN_NETWORK_OPTIONS)

    clock = pygame.time.Clock()
    running = True

    while running:
        time_delta = clock.tick(ViewConsts.MAX_FPS) / 1000.0
        running = state_manager.run(screen, time_delta)
    pygame.quit()


if __name__ == '__main__':
    main()
