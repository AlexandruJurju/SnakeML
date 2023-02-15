import os

import pygame
from pygame_gui import UIManager

from States.genetic_menu import MenuGenetic
from States.genetic_run_trained_network import GeneticRunTrainedNetwork
from States.main_menu import MainMenu
from States.state_manager import StateManager
from constants import ViewConsts, State

if __name__ == '__main__':
    # net = NeuralNetwork()
    # net.add_layer(Dense(NNSettings.INPUT_NEURON_COUNT, NNSettings.HIDDEN_NEURON_COUNT))
    # net.add_layer(Activation(tanh, tanh_prime))
    # net.add_layer(Dense(NNSettings.HIDDEN_NEURON_COUNT, NNSettings.OUTPUT_NEURON_COUNT))
    # net.add_layer(Activation(sigmoid, sigmoid_prime))
    #
    # model = Model(BoardConsts.BOARD_SIZE, SnakeSettings.START_SNAKE_SIZE, net)
    #
    # # train_network(model.snake.brain)
    #
    # game = Game(model, State.MAIN_MENU)
    # game.state_machine()

    pygame.init()
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pygame.key.set_repeat()
    x_screen_size = 1024
    y_screen_size = 600
    pygame.display.set_caption('Snake AI')
    screen = pygame.display.set_mode((ViewConsts.WIDTH, ViewConsts.HEIGHT))

    ui_manager = UIManager(screen.get_size())

    state_manager = StateManager()
    state_manager.add_state(MainMenu(state_manager, ui_manager))
    state_manager.add_state(MenuGenetic(state_manager, ui_manager))
    state_manager.add_state(GeneticRunTrainedNetwork(state_manager, ui_manager))
    state_manager.set_initial_state(State.MAIN_MENU)

    clock = pygame.time.Clock()
    running = True

    while running:
        time_delta = clock.tick(ViewConsts.MAX_FPS) / 1000.0

        running = state_manager.run(screen, time_delta)

        pygame.display.flip()
    pygame.quit()
