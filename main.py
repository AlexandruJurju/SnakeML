import os

import pygame
from pygame_gui import UIManager

from States.backpropagation_train_new_network import BackpropagationTrainNewNetwork
from States.genetic_train_new_network import GeneticTrainNewNetwork
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
    state_manager.add_state(GeneticTrainNewNetwork(ui_manager))
    state_manager.add_state(BackpropagationTrainNewNetwork(ui_manager))

    state_manager.set_initial_state(State.MAIN_MENU)

    running = True

    while running:
        running = state_manager.execute_state(screen)
    pygame.quit()


if __name__ == '__main__':
    # net = NeuralNetwork()
    # net.add_layer(Dense(14, 16))
    # net.add_layer(Activation(relu, relu))
    # net.add_layer(Dense(16, 4))
    # net.add_layer(Activation(sigmoid, sigmoid))

    # model = Model(10, 3, False, net)
    # vision_lines = get_vision_lines_snake_head(model.board, model.snake.body[0], 4,
    #                                            max_dist=-1, apple_return_type="boolean", segment_return_type="boolean", distance_function=chebyshev)
    # print(model.board)
    # print_all_vision_lines(vision_lines)

    # model = Model(10, 3, False, net)
    # vision.put_distances(model.board, model.snake.body[0])
    # print(model.board)

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
