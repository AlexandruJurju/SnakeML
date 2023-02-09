import copy
import csv
from typing import List, Tuple
from Neural.neural_network import *
from constants import *
from vision import get_parameters_in_nn_input_form, get_vision_lines


def read_training_models() -> Tuple:
    file = open(TRAIN_DATA_FILE_LOCATION)
    csvreader = csv.reader(file)

    data = []
    for row in csvreader:
        data.append(row)

    x = []
    y = []

    if len(data) != 0:
        for row in data:
            board = eval(row[0])

            # direction is saved as Direction.UP, but direction.name is just UP, use split to get second part
            direction_string = row[1].split(".")[1]
            real_direction = None
            for direction in MAIN_DIRECTIONS:
                direction_enum_name = direction.name
                if direction_string == direction_enum_name:
                    real_direction = direction
                    break

            vision_lines = get_vision_lines(board)

            x.append(get_parameters_in_nn_input_form(vision_lines, real_direction))

            # dynamic loop over columns in csv, skips board and current direction
            outputs = []
            for i in range(2, len(row)):
                print(row[i])
                outputs.append(float(row[i]))
            y.append(outputs)

    return x, y


def train_network(network: NeuralNetwork) -> None:
    x, y = read_training_models()

    # example for points
    # x is (10000,2) 10000 lines, 2 columns ; 10000 examples each with x coord and y coord
    # when using a single example x_test from x, x_test is (2,)
    # resizing can be done for the whole training data resize(10000,2,1)
    # or for just one example resize(2,1)
    x = np.reshape(x, (len(x), NN_INPUT_NEURON_COUNT, 1))
    y = np.reshape(y, (len(y), NN_OUTPUT_NEURON_COUNT, 1))

    network.train(mse, mse_prime, x, y, 0.5)

    # for x_test, y_test in zip(x, y):
    #     output = network.feed_forward(x_test)
    #     output_index = list(output).index(max(list(output)))
    #     target_index = list(y_test).index(max(list(y_test)))
    #     print(f"target = {target_index}, output = {output_index}")
    #     print("============================================")


class TrainingExample:
    def __init__(self, model: List[str], predictions: List[float], current_direction: Direction):
        self.model = model
        self.predictions = predictions
        self.current_direction = current_direction


def write_examples_to_csv_4d(examples: List[TrainingExample]) -> None:
    file = open(TRAIN_DATA_FILE_LOCATION, "w+", newline='')
    writer = csv.writer(file)

    training_examples = []
    for example in examples:
        up = example.predictions[0]
        down = example.predictions[1]
        left = example.predictions[2]
        right = example.predictions[3]

        training_examples.append([example.model, example.current_direction, up, down, left, right])

    writer.writerows(training_examples)
    file.close()


def evaluate_live_examples_4d(examples: List[TrainingExample]) -> None:
    evaluated = []

    for example in examples:
        print(f"Model \n {np.matrix(example.model)} \n")
        print(f"Current Direction : {example.current_direction} \n")
        print(f"Prediction UP : {example.predictions[0]}")
        print(f"Prediction DOWN : {example.predictions[1]}")
        print(f"Prediction LEFT : {example.predictions[2]}")
        print(f"Prediction RIGHT : {example.predictions[3]}")
        print()

        print("Enter target outputs for neural network in form")
        print("UP=W DOWN=S LEFT=A RIGHT=D")
        target_string = input("")

        if target_string == "":
            target_output = example.predictions
        elif target_string == "x":
            break
        else:
            target_output = [0.0, 0.0, 0.0, 0.0]
            if target_string.__contains__("w"):
                target_output[0] = 1.0
            if target_string.__contains__("s"):
                target_output[1] = 1.0
            if target_string.__contains__("a"):
                target_output[2] = 1.0
            if target_string.__contains__("d"):
                target_output[3] = 1.0

        print(target_output)
        print()
        evaluated.append(TrainingExample(copy.deepcopy(example.model), target_output, example.current_direction))

    write_examples_to_csv_4d(evaluated)

# TODO remake to look like 4d
# def write_examples_to_csv(examples: List[TrainingExample]) -> None:
#     file = open("Neural/train_data_3_output_directions.csv", "w", newline='')
#     writer = csv.writer(file)
#
#     correct_examples = []
#     for example in examples:
#         model_string = str(example.model)
#         model_string = model_string.replace("[[", "[")
#         model_string = model_string.replace("]]", "]")
#         model_string = model_string.replace(" [", "[")
#
#         direction_string = str(example.current_direction)
#         direction_string = direction_string.replace('\'', "")
#         direction_string = direction_string.strip()
#
#         prediction_string = str(np.reshape(example.predictions, (1, 3)))
#         prediction_string = prediction_string.replace("[[", "")
#         prediction_string = prediction_string.replace("]]", "")
#         prediction_string = prediction_string.strip()
#         straight = prediction_string.split(' ')[0]
#         left = prediction_string.split(' ')[1]
#         right = prediction_string.split(' ')[2]
#
#         correct_examples.append([model_string, direction_string, straight, left, right])
#
#     writer.writerows(correct_examples)
#     file.close()
#
#
# def evaluate_live_examples(examples: List[TrainingExample]) -> None:
#     evaluated = []
#     np.set_printoptions(suppress=True)
#
#     for example in examples:
#         print(f"Model \n {example.model} \n")
#         print(f"Current Direction : {example.current_direction} \n")
#         print(f"Prediction Straight : {example.predictions[0]}")
#         print(f"Prediction Left : {example.predictions[1]}")
#         print(f"Prediction Right : {example.predictions[2]}")
#         print()
#
#         print("Enter target outputs for neural network in form")
#         print("Straight Left Right ")
#         target_string = input("")
#
#         if target_string == "":
#             target_output = example.predictions
#         elif target_string == "x":
#             break
#         else:
#             target_output = [0.0, 0.0, 0.0]
#             if target_string.__contains__("s"):
#                 target_output[0] = 1.0
#             if target_string.__contains__("l"):
#                 target_output[1] = 1.0
#             if target_string.__contains__("r"):
#                 target_output[2] = 1.0
#
#         print(target_output)
#         print()
#         evaluated.append(TrainingExample(copy.deepcopy(example.model), target_output, example.current_direction))
#
#     write_examples_to_csv(evaluated)
