import pygame
import pygame_gui
from pygame_gui import UIManager
from pygame_gui.core.utility import create_resource_path
from pygame_gui.elements import UILabel, UIButton, UITextEntryLine
from pygame_gui.windows import UIFileDialog

from States.base_state import BaseState
from States.state_manager import StateManager
from constants import State
from model import Model
from settings import NNSettings
from train_network import read_all_from_json
from view import draw_board
from vision import get_vision_lines


class BackpropTrainedNetwork(BaseState):
    def __init__(self, state_manager: StateManager, ui_manager: UIManager):
        super().__init__(State.BACKPROPAGATION_TRAINED_NETWORK, state_manager)

        self.network = None
        self.vision_return_type = None
        self.input_direction_count = None
        self.ui_manager = ui_manager

        self.model = None
        self.execute_network = False

        self.title_label = None
        self.button_back = None
        self.score_counter = None
        self.button_run = None
        self.button_load = None
        self.file_dialog = None

        self.board_size_entry = None
        self.board_size_label = None

        self.snake_size_entry = None
        self.snake_size_label = None

    def start(self):
        self.title_label = UILabel(pygame.Rect((87, 40), (800, 25)), "Trained Backpropagation Network", self.ui_manager, object_id="#window_label")
        self.button_back = UIButton(pygame.Rect((25, 725), (125, 35)), "BACK", self.ui_manager)
        self.score_counter = UILabel(pygame.Rect((150, 100), (150, 35)), "Score: ", self.ui_manager)
        self.button_load = UIButton(pygame.Rect((25, 100), (125, 35)), "Load Network", self.ui_manager)
        self.button_run = UIButton(pygame.Rect((25, 150), (125, 35)), "Run Network", self.ui_manager)
        self.button_run.disable()

        self.board_size_label = UILabel(pygame.Rect((25, 250), (125, 35)), "Board Size", self.ui_manager)
        self.board_size_entry = UITextEntryLine(pygame.Rect((25, 300), (125, 35)), self.ui_manager)

        self.snake_size_label = UILabel(pygame.Rect((175, 250), (125, 35)), "Snake Size", self.ui_manager)
        self.snake_size_entry = UITextEntryLine(pygame.Rect((175, 300), (125, 35)), self.ui_manager)

    def end(self):
        self.title_label.kill()
        self.button_back.kill()
        self.score_counter.kill()
        self.button_load.kill()
        self.button_run.kill()
        self.board_size_entry.kill()
        self.board_size_label.kill()
        self.snake_size_entry.kill()
        self.snake_size_label.kill()

    def run_network(self, surface):

        vision_lines = get_vision_lines(self.model.board, self.input_direction_count, self.vision_return_type)
        nn_output = self.model.get_nn_output(vision_lines)

        draw_board(surface, self.model.board, 500, 150)
        # self.draw_vision_lines(self.model, vision_lines)
        # self.draw_neural_network(self.model, vision_lines)

        next_direction = self.model.get_nn_output_4directions(nn_output)
        is_alive = self.model.move_in_direction(next_direction)
        self.score_counter.set_text("Score : " + str(self.model.snake.score))

        if not is_alive:
            self.model = Model(int(self.board_size_entry.text), int(self.snake_size_entry.text), self.model.snake.brain)

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
                    self.model = Model(int(self.board_size_entry.text), int(self.snake_size_entry.text), self.network)
                    self.execute_network = True
                if event.ui_element == self.button_load:
                    self.execute_network = False
                    self.button_run.disable()
                    self.file_dialog = UIFileDialog(pygame.Rect((150, 50), (450, 450)), self.ui_manager, window_title="Load Network", initial_file_path=NNSettings.BACKPROPAGATION_NETWORK_FOLDER,
                                                    allow_picking_directories=False,
                                                    allow_existing_files_only=True)
                    self.button_load.disable()
            if event.type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED:
                try:
                    file_path = create_resource_path(event.text)
                    config = read_all_from_json(file_path)
                    self.network = config["network"]
                    self.input_direction_count = config["input_direction_count"]
                    self.vision_return_type = config["vision_return_type"]
                    self.board_size_entry.set_text(str(config["board_size"]))
                    self.snake_size_entry.set_text(str(config["snake_size"]))
                    self.button_load.enable()
                    self.button_run.enable()
                except pygame.error:
                    pass

        self.ui_manager.update(time_delta)

        self.ui_manager.draw_ui(surface)
