import json
from typing import List, Tuple, Dict

import numpy as np

from Neural.neural_network import NeuralNetwork, mse, mse_prime
from constants import Direction
from settings import NNSettings
from vision import get_parameters_in_nn_input_form, VisionLine


class TrainingExample:
    def __init__(self, board: List[str], current_direction: Direction, vision_lines: List[VisionLine], predictions: List[float]):
        self.board = board
        self.current_direction = current_direction
        self.vision_lines = vision_lines
        self.predictions = predictions


def train_network(network: NeuralNetwork) -> None:
    x, y = read_training_data_json()

    # example for points
    # x is (10000,2) 10000 lines, 2 columns ; 10000 examples each with x coord and y coord
    # when using a single example x_test from x, x_test is (2,)
    # resizing can be done for the whole training data resize(10000,2,1)
    # or for just one example resize(2,1)
    x = np.reshape(x, (len(x), NNSettings.INPUT_NEURON_COUNT, 1))
    y = np.reshape(y, (len(y), NNSettings.OUTPUT_NEURON_COUNT, 1))

    network.train(mse, mse_prime, x, y, 0.5)

    # for x_test, y_test in zip(x, y):
    #     output = network.feed_forward(x_test)
    #     output_index = list(output).index(max(list(output)))
    #     target_index = list(y_test).index(max(list(y_test)))
    #     print(f"target = {target_index}, output = {output_index}")
    #     print("============================================")


def write_examples_to_json_4d(examples: List[TrainingExample]) -> None:
    dictionary_list: List[Dict] = []

    for example in examples:
        up = example.predictions[0]
        down = example.predictions[1]
        left = example.predictions[2]
        right = example.predictions[3]

        vision_lines = []
        for line in example.vision_lines:
            line_dict = {
                "direction": line.direction.name,
                "wall_coord": line.wall_coord,
                "wall_distance": line.wall_distance,
                "apple_coord": line.apple_coord,
                "apple_distance": line.apple_distance,
                "segment_coord": line.segment_coord,
                "segment_distance": line.segment_distance
            }
            vision_lines.append(line_dict)

        example_dictionary: Dict = {
            "board": example.board,
            "current_direction": example.current_direction.name,
            "vision_lines": vision_lines,
            "up": up,
            "down": down,
            "left": left,
            "right": right
        }
        dictionary_list.append(example_dictionary)

    output_file = open(NNSettings.TRAIN_DATA_FILE_LOCATION, "w")
    json.dump(dictionary_list, output_file)
    output_file.close()


def read_training_data_json() -> Tuple[List, List]:
    json_file = open(NNSettings.TRAIN_DATA_FILE_LOCATION, "r")
    json_object = json.load(json_file)

    x = []
    y = []
    if json_object:
        for example in json_object:
            real_direction = Direction[example["current_direction"]]

            vision_lines = []
            for line in example["vision_lines"]:
                line = VisionLine(line["wall_coord"], line["wall_distance"], line["apple_coord"], line["apple_distance"], line["segment_coord"], line["segment_distance"], Direction[line["direction"]])
                vision_lines.append(line)

            x.append(get_parameters_in_nn_input_form(vision_lines, real_direction))

            outputs = [example["up"], example["down"], example["left"], example["right"]]
            y.append(outputs)

    json_file.close()

    return x, y

# def evaluate_live_examples_4d(examples: List[TrainingExample]) -> None:
#     evaluated = []
#
#     for example in examples:
#         print(f"Model \n {np.matrix(example.board)} \n")
#         print(f"Current Direction : {example.current_direction} \n")
#         print(f"Prediction UP : {example.predictions[0]}")
#         print(f"Prediction DOWN : {example.predictions[1]}")
#         print(f"Prediction LEFT : {example.predictions[2]}")
#         print(f"Prediction RIGHT : {example.predictions[3]}")
#         print()
#
#         # if ViewVars.DRAW:
#         #     self.view.clear_window()
#         #     self.view.draw_board(example.model)
#         #     self.view.update_window()
#
#         print("Enter target outputs for neural network in form")
#         print("UP=W DOWN=S LEFT=A RIGHT=D")
#         target_string = input("")
#
#         if target_string == "":
#             target_output = example.predictions
#         elif target_string == "x":
#             break
#         else:
#             target_output = [0.0, 0.0, 0.0, 0.0]
#             if target_string.__contains__("w"):
#                 target_output[0] = 1.0
#             if target_string.__contains__("s"):
#                 target_output[1] = 1.0
#             if target_string.__contains__("a"):
#                 target_output[2] = 1.0
#             if target_string.__contains__("d"):
#                 target_output[3] = 1.0
#
#         print(target_output)
#         print()
#         evaluated.append(TrainingExample(copy.deepcopy(example.board), target_output, example.current_direction))
#
#     write_examples_to_csv_4d(evaluated)
