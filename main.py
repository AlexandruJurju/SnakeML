import os

import pygame
from pygame_gui import UIManager

from States.backpropagation_train_network import BackpropagationTrainNetwork
from States.genetic_train_network import GeneticTrainNetwork
from States.main_menu import MainMenu
from States.options import Options
from States.run_network import RunTrained
from States.state_manager import StateManager
from game_config import ViewSettings, State


def main():
    pygame.init()
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pygame.key.set_repeat()
    pygame.display.set_caption('Snake AI')
    screen = pygame.display.set_mode((ViewSettings.WIDTH, ViewSettings.HEIGHT))

    ui_manager = UIManager(screen.get_size(), "data/themes/light.json")
    ui_manager.add_font_paths('jetbrains', "data/fonts/JetBrainsMono-Regular.ttf")
    ui_manager.preload_fonts([{'name': 'jetbrains', 'point_size': 18, 'style': 'regular'}])

    state_manager = StateManager()
    state_manager.add_state(MainMenu(ui_manager))
    state_manager.add_state(Options(ui_manager))
    state_manager.add_state(RunTrained(ui_manager))
    state_manager.add_state(GeneticTrainNetwork(ui_manager))
    state_manager.add_state(BackpropagationTrainNetwork(ui_manager))

    state_manager.set_initial_state(State.MAIN_MENU)

    running = True

    while running:
        running = state_manager.execute_state(screen)
    pygame.quit()


if __name__ == '__main__':

    main()
