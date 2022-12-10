import csv
from Neural.neural_network import *
from constants import *
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

    network.train(mse, mse_prime, x, y, 0.1)

    # for x_test, y_test in zip(x, y):
    #     output = network.feed_forward(x_test)
    #     output_index = list(output).index(max(list(output)))
    #     target_index = list(y_test).index(max(list(y_test)))
    #     print(f"target = {target_index}, output = {output_index}")
    #     print("============================================")
