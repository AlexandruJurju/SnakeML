import json
from typing import Tuple, Dict

from constants import Direction
from neural_network import *
from vision import get_parameters_in_nn_input_form, VisionLine


class TrainingExample:
    def __init__(self, board: List[List[str]], current_direction: Direction, vision_lines: List[VisionLine], predictions: List[float]):
        self.board = board
        self.current_direction = current_direction
        self.vision_lines = vision_lines
        self.predictions = predictions


def train_network(network: NeuralNetwork, file_path: str) -> None:
    x, y = read_training_data_json(file_path)

    # example for points
    # x is (10000,2) 10000 lines, 2 columns ; 10000 examples each with x coord and y coord
    # when using a single example x_test from x, x_test is (2,)
    # resizing can be done for the whole training data resize(10000,2,1)
    # or for just one example resize(2,1)

    input_neuron_count = network.get_dense_layers()[0].input_size
    output_neuron_count = network.get_dense_layers()[-1].output_size

    x = np.reshape(x, (len(x), input_neuron_count, 1))
    y = np.reshape(y, (len(y), output_neuron_count, 1))

    network.train(mse, mse_prime, x, y, 0.5)

    # for x_test, y_test in zip(x, y):
    #     output = network.feed_forward(x_test)
    #     output_index = list(output).index(max(list(output)))
    #     target_index = list(y_test).index(max(list(y_test)))
    #     print(f"target = {target_index}, output = {output_index}")
    #     print("============================================")


def write_examples_to_json_4d(examples: List[TrainingExample], output_file_location: str) -> None:
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

    output_file = open(output_file_location, "w")
    json.dump(dictionary_list, output_file)
    output_file.close()


def read_training_data_json(file_location) -> Tuple[List, List]:
    json_file = open(file_location, "r")
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


def save_neural_network_to_json(generation: int, fitness: int, input_direction_count, vision_return_type, network: NeuralNetwork, path: str) -> None:
    network_dict = []
    for i, layer in enumerate(network.layers):
        if type(layer) is Activation:
            layer_dict = {
                "layer": "activation",
                "activation": layer.activation.__name__,
                "activation_prime": layer.activation_prime.__name__
            }
        else:
            layer_dict = {
                "layer": "dense",
                "input_size": layer.input_size,
                "output_size": layer.output_size,
                "weights": layer.weights.tolist(),
                "bias": layer.bias.tolist()
            }
        network_dict.append(layer_dict)

    generation_network = {"generation": generation,
                          "fitness": fitness,
                          "input_direction_count": input_direction_count,
                          "vision_return_type": vision_return_type,
                          "network": network_dict}

    network_file = open(path + ".json", "w")
    json.dump(generation_network, network_file)
    network_file.close()


def read_all_from_json(path: str) -> Dict:
    json_file = open(path, "r")
    json_object = json.load(json_file)

    output_network = NeuralNetwork()
    if json_object:
        for layer in json_object["network"]:
            if layer["layer"] == "dense":
                input_size = layer["input_size"]
                output_size = layer["output_size"]
                weights = np.reshape(layer["weights"], (layer["output_size"], layer["input_size"]))
                bias = np.reshape(layer["bias"], (layer["output_size"], 1))
                dense_layer = Dense(input_size, output_size)
                dense_layer.weights = weights
                dense_layer.bias = bias
                output_network.add_layer(dense_layer)
            else:
                activation_str = layer["activation"]
                activation_prime_str = layer["activation_prime"]

                if activation_str == "tanh":
                    activation = tanh
                else:
                    activation = sigmoid

                if activation_prime_str == "tanh_prime":
                    activation_prime = tanh_prime
                else:
                    activation_prime = sigmoid_prime

                activation_layer = Activation(activation, activation_prime)
                output_network.add_layer(activation_layer)

    output = json_object
    output["network"] = output_network
    return output


# def read_neural_network_from_json(path: str) -> NeuralNetwork:
#     json_file = open(path, "r")
#     json_object = json.load(json_file)
#
#     output_network = NeuralNetwork()
#     if json_object:
#         for layer in json_object["network"]:
#             if layer["layer"] == "dense":
#                 input_size = layer["input_size"]
#                 output_size = layer["output_size"]
#                 weights = np.reshape(layer["weights"], (layer["output_size"], layer["input_size"]))
#                 bias = np.reshape(layer["bias"], (layer["output_size"], 1))
#                 dense_layer = Dense(input_size, output_size)
#                 dense_layer.weights = weights
#                 dense_layer.bias = bias
#                 output_network.add_layer(dense_layer)
#             else:
#                 activation_str = layer["activation"]
#                 activation_prime_str = layer["activation_prime"]
#
#                 if activation_str == "tanh":
#                     activation = tanh
#                 else:
#                     activation = sigmoid
#
#                 if activation_prime_str == "tanh_prime":
#                     activation_prime = tanh_prime
#                 else:
#                     activation_prime = sigmoid_prime
#
#                 activation_layer = Activation(activation, activation_prime)
#                 output_network.add_layer(activation_layer)
#
#     return output_network
