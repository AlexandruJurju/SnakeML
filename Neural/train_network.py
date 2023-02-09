import copy
import csv
from typing import List, Tuple
from Neural.neural_network import *
from constants import *
from vision import get_parameters_in_nn_input_form, get_vision_lines




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
