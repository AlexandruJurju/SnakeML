import copy
from typing import List
import numpy as np
from constants import Direction
import csv
from termcolor import colored, cprint
import csv
from Neural.neural_network import *
from constants import *
from vision import Vision

from vision import Vision


def read_training_models():
    file = open("Neural/train_data.csv")
    csvreader = csv.reader(file)

    rows = []
    for row in csvreader:
        rows.append(row)

    x = []
    y = []

    for row in rows:
        model_string = row[0]
        model_string = model_string.replace("[", "")
        model_string = model_string.replace("]", "")
        model_string = model_string.replace("'", "")
        row_list = model_string.split("\n")

        # TODO make it dynamic
        temp_board = np.empty((10 + 2, 10 + 2), dtype=object)
        for i, model_row in enumerate(row_list):
            values_in_row = model_row.split(" ")
            for j, model_column in enumerate(values_in_row):
                temp_board[i, j] = model_column

        # direction is saved as Direction.UP, but direction.name is just UP, use split to get second part
        direction_string = row[1].split(".")[1]
        real_direction = None
        for direction in MAIN_DIRECTIONS:
            direction_enum_name = direction.name
            if direction_string == direction_enum_name:
                real_direction = direction
                break

        vision_lines = Vision.get_dynamic_vision_lines(temp_board, real_direction)

        x.append(Vision.get_parameters_in_nn_input_form(vision_lines, real_direction))

        outputs_string_list = row[2].split(" ")
        outputs = []
        for tuple_string in outputs_string_list:
            if tuple_string != "":
                outputs.append(float(tuple_string))
        y.append(outputs)

    return x, y


def train_network(network: NeuralNetwork):
    x, y = read_training_models()

    # example for points
    # x is (10000,2) 10000 lines, 2 columns ; 10000 examples each with x coord and y coord
    # when using a single example x_test from x, x_test is (2,)
    # resizing can be done for the whole training data resize(10000,2,1)
    # or for just one example resize(2,1)
    # TODO reshape based on network layers
    x = np.reshape(x, (len(x), VISION_LINES_COUNT * 3 + 4, 1))
    y = np.reshape(y, (len(y), 3, 1))

    network.train(mse, mse_prime, x, y, 0.5)

    # for x_test, y_test in zip(x, y):
    #     output = network.feed_forward(x_test)
    #     output_index = list(output).index(max(list(output)))
    #     target_index = list(y_test).index(max(list(y_test)))
    #     print(f"target = {target_index}, output = {output_index}")
    #     print("============================================")


class TrainingExample:
    def __init__(self, model, prediction, current_direction):
        self.model = model
        self.prediction = prediction
        self.current_direction = current_direction


def write_model_predictions(model: [[]], prediction: np.ndarray) -> None:
    model_string = str(model)
    model_string = model_string.replace("[[", "\"[")
    model_string = model_string.replace("]]", "]\"")
    model_string = model_string.replace(" [", "[")

    prediction_string = str(np.reshape(prediction, (1, 3)))
    prediction_string = prediction_string.replace("[[", "")
    prediction_string = prediction_string.replace("]]", "")

    output = (model_string + "," + prediction_string)


def write_examples_to_csv(examples: List[TrainingExample]):
    file = open("Neural/train_data.csv", "w", newline='')
    writer = csv.writer(file)

    correct_examples = []
    for example in examples:
        model_string = str(example.model)
        model_string = model_string.replace("[[", "[")
        model_string = model_string.replace("]]", "]")
        model_string = model_string.replace(" [", "[")

        direction_string = str(example.current_direction)
        direction_string = direction_string.replace('\'', "")
        direction_string = direction_string.strip()

        prediction_string = str(np.reshape(example.prediction, (1, 3)))
        prediction_string = prediction_string.replace("[[", "")
        prediction_string = prediction_string.replace("]]", "")
        prediction_string = prediction_string.strip()

        correct_examples.append([model_string, direction_string, prediction_string])

    writer.writerows(correct_examples)
    file.close()


def rotate_90_clockwise(matrix) -> [[]]:
    size = len(matrix[0])
    for i in range(size // 2):
        for j in range(i, size - i - 1):
            temp = matrix[i][j]
            matrix[i][j] = matrix[size - 1 - j][i]
            matrix[size - 1 - j][i] = matrix[size - 1 - i][size - 1 - j]
            matrix[size - 1 - i][size - 1 - j] = matrix[j][size - 1 - i]
            matrix[j][size - 1 - i] = temp
    return matrix


def rotate_90_counterclockwise(matrix) -> [[]]:
    matrix = np.rot90(matrix)
    return matrix


def reorient_board(board, current_direction: Direction) -> [[]]:
    match current_direction:
        case Direction.UP:
            pass
        case Direction.LEFT:
            board = rotate_90_clockwise(board)
        case Direction.RIGHT:
            board = rotate_90_counterclockwise(board)
        case Direction.DOWN:
            board = rotate_90_clockwise(board)
            board = rotate_90_clockwise(board)
    return board


def print_example_smart(model, current_direction: Direction):
    head = Vision.find_snake_head_poz(model)
    color_positions = []

    match current_direction:
        case Direction.UP:
            color_positions.append([head[0] - 1, head[1]])
            color_positions.append([head[0], head[1] - 1])
            color_positions.append([head[0], head[1] + 1])


def evaluate_live_examples(examples: List[TrainingExample]):
    evaluated = []
    np.set_printoptions(suppress=True)

    for example in examples:
        print(example.model)
        print(example.prediction)
        print(example.current_direction)
        print()
        target_string = input("S L R : ")

        if target_string == "":
            target_output = example.prediction
        elif target_string == "x":
            break
        else:
            target_output = [0.0, 0.0, 0.0]
            if target_string.__contains__("s"):
                target_output[0] = 1.0
            if target_string.__contains__("l"):
                target_output[1] = 1.0
            if target_string.__contains__("r"):
                target_output[2] = 1.0

        print(target_output)
        print()
        evaluated.append(TrainingExample(copy.deepcopy(example.model), target_output, example.current_direction))

    write_examples_to_csv(evaluated)
