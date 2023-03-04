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
from constants import ViewConsts, State


def main():
    pygame.init()
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pygame.key.set_repeat()
    pygame.display.set_caption('Snake AI')
    screen = pygame.display.set_mode((ViewConsts.WIDTH, ViewConsts.HEIGHT))

    ui_manager = UIManager(screen.get_size(), "data/themes/ui_theme.json")

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

    state_manager.set_initial_state(State.BACKPROPAGATION_TRAIN_NEW_NETWORK_OPTIONS)

    clock = pygame.time.Clock()
    running = True

    while running:
        time_delta = clock.tick(ViewConsts.MAX_FPS) / 1000.0
        running = state_manager.run(screen, time_delta)
    pygame.quit()


if __name__ == '__main__':
    main()
