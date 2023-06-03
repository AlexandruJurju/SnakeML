import os
from typing import List

import pygame
from pygame_gui import UIManager

import vision
from States.backpropagation_train_network import BackpropagationTrainNetwork
from States.genetic_train_network import GeneticTrainNetwork
from States.main_menu import MainMenu
from States.options import Options
from States.run_network import RunTrained
from States.state_manager import StateManager
from game_config import ViewSettings, State
from model import Model
from neural_network import NeuralNetwork, Dense, Activation, sigmoid, relu


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


def print_all_vision_lines(lines: List[vision.VisionLine]):
    for line in lines:
        print(f"Wall D: {line.wall_distance} Apple D: {line.apple_distance} Segment D: {line.segment_distance}")


if __name__ == '__main__':
    net = NeuralNetwork()
    net.add_layer(Dense(14, 16))
    net.add_layer(Activation(relu, relu))
    net.add_layer(Dense(16, 4))
    net.add_layer(Activation(sigmoid, sigmoid))

    model = Model(10, 3, net)
    vision.put_distances(model.board, model.snake.body[0])
    # vision_lines = vision.get_vision_lines(model.board, model.snake.body[0], 4, apple_return_type="distance", segment_return_type="distance")
    #
    # np.set_printoptions(precision=4, suppress=True, linewidth=10000)
    # print(net.get_dense_layers()[0].weights)

    # main()
