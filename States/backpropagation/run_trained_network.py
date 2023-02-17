import pygame
import pygame_gui
from pygame_gui import UIManager
from pygame_gui.core.utility import create_resource_path
from pygame_gui.elements import UILabel, UIButton
from pygame_gui.windows import UIFileDialog

from States.base_state import BaseState
from States.state_manager import StateManager
from constants import State
from model import Model
from settings import BoardSettings, SnakeSettings, NNSettings
from train_network import read_neural_network_from_json
from view import draw_board
from vision import get_vision_lines


class BackpropagationTrainedNetwork(BaseState):
    def __init__(self, state_manager: StateManager, ui_manager: UIManager):
        super().__init__(State.BACKPROPAGATION_TRAINED_NETWORK, state_manager)

        self.ui_manager = ui_manager

        self.model = None
        self.execute_network = False

        self.title_label = None
        self.button_back = None
        self.score_counter = None
        self.button_run = None
        self.button_load = None
        self.file_dialog = None

    def start(self):
        self.title_label = UILabel(pygame.Rect((87, 40), (800, 25)), "Trained Backpropagation Network", self.ui_manager, object_id="#window_label")
        self.button_back = UIButton(pygame.Rect((25, 725), (125, 35)), "BACK", self.ui_manager)
        self.score_counter = UILabel(pygame.Rect((150, 100), (150, 35)), "Score: ", self.ui_manager)
        self.button_load = UIButton(pygame.Rect((25, 100), (125, 35)), "Load Network", self.ui_manager)
        self.button_run = UIButton(pygame.Rect((25, 150), (125, 35)), "Run Network", self.ui_manager)
        self.button_run.disable()

    def end(self):
        self.title_label.kill()
        self.button_back.kill()
        self.score_counter.kill()
        self.button_load.kill()
        self.button_run.kill()

    def run_network(self, surface):

        vision_lines = get_vision_lines(self.model.board)
        nn_output = self.model.get_nn_output(vision_lines)

        draw_board(surface, self.model.board, 500, 300)
        # self.draw_vision_lines(self.model, vision_lines)
        # self.draw_neural_network(self.model, vision_lines)
        # self.write_ttl(self.model.snake.ttl)
        # self.write_score(self.model.snake.score)

        next_direction = self.model.get_nn_output_4directions(nn_output)
        is_alive = self.model.move_in_direction(next_direction)
        self.score_counter.set_text("Score : " + str(self.model.snake.score))

        if not is_alive:
            self.model = Model(BoardSettings.BOARD_SIZE, SnakeSettings.START_SNAKE_SIZE, self.model.snake.brain)

    def run(self, surface, time_delta):
        surface.fill(self.ui_manager.ui_theme.get_colour("dark_bg"))

        if self.execute_network:
            self.run_network(surface)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.set_target_state_name(State.QUIT)
                self.trigger_transition()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.set_target_state_name(State.QUIT)
                    self.trigger_transition()

            self.ui_manager.process_events(event)

            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.button_back:
                    self.set_target_state_name(State.BACKPROPAGATION_MENU)
                    self.trigger_transition()
                if event.ui_element == self.button_run:
                    self.execute_network = True
                if event.ui_element == self.button_load:
                    self.execute_network = False
                    self.button_run.disable()
                    self.file_dialog = UIFileDialog(pygame.Rect((150, 50), (450, 450)), self.ui_manager, window_title="Load Network", initial_file_path=NNSettings.BACKPROPAGATION_NETWORK_FOLDER, allow_picking_directories=False,
                                                    allow_existing_files_only=True)
                    self.button_load.disable()
            if event.type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED:
                try:
                    file_path = create_resource_path(event.text)
                    network = read_neural_network_from_json(file_path)
                    self.model = Model(BoardSettings.BOARD_SIZE, SnakeSettings.START_SNAKE_SIZE, network)
                    self.button_load.enable()
                    self.button_run.enable()
                except pygame.error:
                    pass

        self.ui_manager.update(time_delta)

        self.ui_manager.draw_ui(surface)
