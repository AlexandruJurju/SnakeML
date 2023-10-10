import os
import sys

import pygame
from pygame_gui import UIManager

from States.backpropagation_train_network import BackpropagationTrainNetwork
from States.genetic_train_network import GeneticTrainNetwork
from States.main_menu import MainMenu
from States.options import Options
from States.run_network import RunTrained
from States.state_manager import StateManager
from game_config import ViewSettings, State, GameSettings


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller's 'onefile' mode """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def create_folder_if_not_there(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def init_folders():
    create_folder_if_not_there(GameSettings.GENETIC_NETWORK_FOLDER)
    create_folder_if_not_there(GameSettings.BACKPROPAGATION_NETWORK_FOLDER)
    create_folder_if_not_there(GameSettings.BACKPROPAGATION_TRAINING_DATA)


def main():
    init_folders()
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


# def print_all_vision_lines(lines: List[vision.VisionLine]):
#     for line in lines:
#         print(f"Wall D: {line.wall_distance} Apple D: {line.apple_distance} Segment D: {line.segment_distance}")


if __name__ == '__main__':
    main()
