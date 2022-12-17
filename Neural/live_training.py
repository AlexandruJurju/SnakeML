import copy
from typing import List
import numpy as np
from constants import Direction
import csv
from termcolor import colored, cprint

from vision import Vision


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
    file = open("Neural/train_data2.csv", "w", newline='')
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
