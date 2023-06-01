import pygame_gui
from pygame_gui import UIManager
from pygame_gui.elements import UILabel, UIButton

import cvision
import neural_network
import view
from States.base_state import BaseState
from file_operations import TrainingExample, save_neural_network_to_json, read_training_data_and_train, write_examples_to_json_4d
from game_config import State, GameSettings
from neural_network import *
from view import *


class BackpropagationTrainNetwork(BaseState):
    def __init__(self, ui_manager: UIManager):
        super().__init__(State.BACKPROPAGATION_TRAIN_NETWORK)

        self.training = None
        self.initial_board_size = None
        self.initial_snake_size = None
        self.input_direction_count = None
        self.segment_return_type = None
        self.apple_return_type = None
        self.segment_return_type = None
        self.file_name = None
        self.ui_manager = ui_manager
        self.model = None

        self.training_examples: List[TrainingExample] = []

        self.title_label = None
        self.button_back = None

    def start(self):
        self.training_examples: List[TrainingExample] = []

        self.title_label = UILabel(pygame.Rect(ViewSettings.TITLE_LABEL_POSITION, ViewSettings.TITLE_LABEL_DIMENSION), "Backpropagation Train New Network", self.ui_manager)
        self.button_back = UIButton(pygame.Rect(ViewSettings.BUTTON_BACK_POSITION, ViewSettings.BUTTON_BACK_DIMENSION), "BACK", self.ui_manager)

        self.initial_board_size = self.data_received["board_size"]
        self.initial_snake_size = self.data_received["initial_snake_size"]
        self.input_direction_count = self.data_received["input_direction_count"]
        self.segment_return_type = self.data_received["segment_return_type"]
        self.apple_return_type = self.data_received["apple_return_type"]
        self.file_name = self.data_received["file_name"]

        input_neuron_count = self.data_received["input_layer_neurons"]
        hidden_neuron_count = self.data_received["hidden_layer_neurons"]
        output_neuron_count = self.data_received["output_layer_neurons"]

        hidden_activation = getattr(neural_network, self.data_received["hidden_activation"])
        output_activation = getattr(neural_network, self.data_received["output_activation"])
        hidden_activation_prime = getattr(neural_network, self.data_received["hidden_activation"] + "_prime")
        output_activation_prime = getattr(neural_network, self.data_received["output_activation"] + "_prime")

        net = NeuralNetwork()
        net.add_layer(Dense(input_neuron_count, hidden_neuron_count))
        net.add_layer(Activation(hidden_activation, hidden_activation_prime))
        net.add_layer(Dense(hidden_neuron_count, output_neuron_count))
        net.add_layer(Activation(output_activation, output_activation_prime))

        self.model = Model(self.initial_board_size, self.initial_snake_size, net)
        self.training = False

    def end(self):
        self.title_label.kill()
        self.button_back.kill()

    def check_if_already_seen(self, vision_lines) -> int:
        for i in range(len(self.training_examples)):
            if np.array_equal(self.training_examples[i].vision_lines, vision_lines):
                return i
        return None

    def play_game_manual(self, surface, time_delta):
        snake_head = np.asarray(self.model.snake.body[0], dtype=np.int32)
        vision_lines = cvision.get_vision_lines_snake_head(self.model.board, snake_head, self.input_direction_count, apple_return_type=self.apple_return_type, segment_return_type=self.segment_return_type)
        old_lines = vision.cvision_to_old_vision(vision_lines)
        example_index = self.check_if_already_seen(old_lines)

        if example_index is not None:
            predictions = self.training_examples[example_index].user_move
            direction = self.model.get_nn_output_4directions(predictions)

            draw_board(surface, self.model.board, ViewSettings.BOARD_POSITION[0], ViewSettings.BOARD_POSITION[1])
            view.draw_next_snake_direction(surface, self.model.snake.body[0], direction, ViewSettings.BOARD_POSITION[0], ViewSettings.BOARD_POSITION[1])
            write_controls(surface, 300, 300)
            self.ui_manager.update(time_delta)
            self.ui_manager.draw_ui(surface)
            pygame.display.flip()

            direction_to_move = None
            input_string = self.wait_for_key()
            if input_string == "":
                direction_to_move = direction
            else:
                target_output = [0.0, 0.0, 0.0, 0.0]
                if input_string == "W":
                    target_output[0] = 1.0
                    direction_to_move = Direction.UP
                if input_string == "S":
                    target_output[1] = 1.0
                    direction_to_move = Direction.DOWN
                if input_string == "A":
                    target_output[2] = 1.0
                    direction_to_move = Direction.LEFT
                if input_string == "D":
                    target_output[3] = 1.0
                    direction_to_move = Direction.RIGHT

                self.training_examples[example_index].user_move = target_output

            self.model.move(direction_to_move)

        else:
            draw_board(surface, self.model.board, ViewSettings.BOARD_POSITION[0], ViewSettings.BOARD_POSITION[1])
            draw_vision_lines(surface, snake_head, old_lines, ViewSettings.BOARD_POSITION[0], ViewSettings.BOARD_POSITION[1])
            write_controls(surface, 300, 300)
            self.ui_manager.update(time_delta)
            self.ui_manager.draw_ui(surface)
            pygame.display.flip()

            input_string = self.wait_for_key()
            target_output = [0.0, 0.0, 0.0, 0.0]
            direction_to_move = None
            if input_string == "W":
                target_output[0] = 1.0
                direction_to_move = Direction.UP
            if input_string == "S":
                target_output[1] = 1.0
                direction_to_move = Direction.DOWN
            if input_string == "A":
                target_output[2] = 1.0
                direction_to_move = Direction.LEFT
            if input_string == "D":
                target_output[3] = 1.0
                direction_to_move = Direction.RIGHT

            if input_string != "B":
                example = TrainingExample(self.model.snake.direction, old_lines, target_output)
                self.training_examples.append(example)
                is_alive = self.model.move(direction_to_move)
                if not is_alive:
                    self.model = Model(self.initial_board_size, self.initial_snake_size, self.model.snake.brain)

    def wait_for_key(self) -> str:
        while True:
            event = pygame.event.wait()
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.button_back:
                    self.set_target_state_name(State.OPTIONS)
                    self.data_to_send["state"] = "backpropagation"
                    self.trigger_transition()
                    break

            if event.type == pygame.KEYDOWN:
                match event.key:
                    case pygame.K_w:
                        return "W"
                    case pygame.K_s:
                        return "S"
                    case pygame.K_a:
                        return "A"
                    case pygame.K_d:
                        return "D"
                    case pygame.K_RETURN:
                        return ""
                    case pygame.K_x:
                        return "X"
                    case pygame.K_ESCAPE:
                        ViewSettings.DRAW = False
                        data_to_save = {
                            "generation": -1,
                            "initial_board_size": self.initial_board_size,
                            "initial_snake_size": self.initial_snake_size,
                            "input_direction_count": self.input_direction_count,
                            "apple_return_type": self.apple_return_type,
                            "segment_return_type": self.segment_return_type
                        }

                        file_path = "Backpropagation_Training/" + self.data_received["file_name"] + ".json"
                        write_examples_to_json_4d(self.training_examples, file_path)

                        self.model.snake.brain.reinit_weights_and_biases()
                        self.model = Model(self.initial_board_size, self.initial_snake_size, self.model.snake.brain)
                        read_training_data_and_train(self.model.snake.brain, file_path)

                        save_neural_network_to_json(data_to_save,
                                                    self.model.snake.brain,
                                                    GameSettings.BACKPROPAGATION_NETWORK_FOLDER + self.data_received["file_name"])

                        self.set_target_state_name(State.MAIN_MENU)
                        self.trigger_transition()
                        ViewSettings.DRAW = True
                        return "B"

    def run(self, surface, time_delta):
        if ViewSettings.DRAW:
            surface.fill(self.ui_manager.ui_theme.get_colour("main_bg"))
        # file_path = "Backpropagation_Training/" + self.data_received["file_name"] + ".json"
        # self.model.snake.brain.reinit_weights_and_biases()
        # self.model = Model(self.initial_board_size, self.initial_snake_size, self.model.snake.brain)
        # read_training_data_and_train(self.model.snake.brain, file_path)
        #
        # data_to_save = {
        #     "generation": -1,
        #     "initial_board_size": self.initial_board_size,
        #     "initial_snake_size": self.initial_snake_size,
        #     "input_direction_count": self.input_direction_count,
        #     "apple_return_type": self.apple_return_type,
        #     "segment_return_type": self.segment_return_type
        # }
        #
        # save_neural_network_to_json(data_to_save,
        #                             self.model.snake.brain,
        #                             GameSettings.BACKPROPAGATION_NETWORK_FOLDER + self.data_received["file_name"])

        self.play_game_manual(surface, time_delta)
        if ViewSettings.DRAW:
            self.ui_manager.update(time_delta)
            self.ui_manager.draw_ui(surface)
