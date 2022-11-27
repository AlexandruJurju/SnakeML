from numpy import ndarray
import csv
from game import *


def find_head_coord(board):
    for i in range(0, len(board)):
        for j in range(0, len(board)):
            if board[i, j] == "H":
                return i, j


def look_in_direction(board, direction: Direction) -> {}:
    apple_distance = np.inf
    segment_distance = np.inf

    apple_coord = None
    segment_coord = None

    head = find_head_coord(board)
    current_block = [head[0] + direction.value[0], head[1] + direction.value[1]]

    apple_found = False
    segment_found = False

    while board[current_block[0], current_block[1]] != "W":
        if board[current_block[0], current_block[1]] == "A" and apple_found == False:
            apple_distance = math.dist(head, current_block)
            apple_coord = current_block
            apple_found = True
        elif board[current_block[0], current_block[1]] == "S" and segment_found == False:
            segment_distance = math.dist(head, current_block)
            segment_coord = current_block
            segment_found = True
        current_block = [current_block[0] + direction.value[0], current_block[1] + direction.value[1]]

    wall_distance = math.dist(head, current_block)
    wall_coord = current_block

    wall_distance_output = 1 / wall_distance
    apple_boolean = 1.0 if apple_found else 0.0
    segment_boolean = 1.0 if segment_found else 0.0

    # vision = {
    #     "W": [wall_coord, wall_distance_output],
    #     "A": [apple_coord, apple_boolean],
    #     "S": [segment_coord, segment_boolean]
    # }

    return VisionLine(wall_coord, wall_distance, apple_coord, apple_boolean, segment_coord, segment_boolean)


def get_vision_lines(board) -> {VisionLine}:
    return {
        "+X": look_in_direction(board, Direction.RIGHT),
        "-X": look_in_direction(board, Direction.LEFT),
        "-Y": look_in_direction(board, Direction.DOWN),
        "+Y": look_in_direction(board, Direction.UP),
        "Q1": look_in_direction(board, Direction.Q1),
        "Q2": look_in_direction(board, Direction.Q2),
        "Q3": look_in_direction(board, Direction.Q3),
        "Q4": look_in_direction(board, Direction.Q4)
    }


def make_board(size) -> ndarray:
    board = np.empty((size, size), dtype=object)

    for i in range(0, size):
        for j in range(0, size):
            # place walls on the borders and nothing inside
            if i == 0 or i == size - 1 or j == 0 or j == size - 1:
                board[i, j] = "W"
            else:
                board[i, j] = "X"

    return board


def put_snake_on_board(board, snake_positions: []) -> np.ndarray:
    for i, position in enumerate(snake_positions):
        if i == 0:
            board[position[0], position[1]] = "H"
        else:
            board[position[0], position[1]] = "S"
    return board


def board_to_nn_input(board):
    nn_input = []
    vision_lines = get_vision_lines(board)

    for line in vision_lines:
        nn_input.append(vision_lines[line].wall_distance)
        nn_input.append(vision_lines[line].apple_distance)
        nn_input.append(vision_lines[line].segment_distance)

    return np.reshape(nn_input, (len(nn_input), 1))


def read_training_models():
    file = open("train.csv")
    csvreader = csv.reader(file)

    rows = []
    for row in csvreader:
        rows.append(row)

    x = []
    y = []

    for row in rows:
        temp_board = np.empty((10 + 2, 10 + 2), dtype=object)
        model_string = row[0]
        model_string = model_string.replace("[", "")
        model_string = model_string.replace("]", "")
        model_string = model_string.replace("'", "")
        row_list = model_string.split("\n")

        for i, model_row in enumerate(row_list):
            values_in_row = model_row.split(" ")
            for j, model_column in enumerate(values_in_row):
                temp_board[i, j] = model_column
        x.append(temp_board)

        outputs_string_list = row[1].split(" ")
        outputs = []
        for tuple_string in outputs_string_list:
            outputs.append(float(tuple_string))
        y.append(outputs)

    x_train = []
    for config in x:
        print(config)
        print(board_to_nn_input(config))
        print()
        x_train.append(board_to_nn_input(config))

    return x_train, y


if __name__ == "__main__":
    net = NeuralNetwork()
    net.add(Dense(24, 16))
    net.add(Activation(tanh, tanh_prime))
    net.add(Dense(16, 3))
    net.add(Activation(sigmoid, sigmoid_prime))

    board_size = 10 + 2
    board = make_board(board_size)

    X, Y = read_training_models()

    # TODO split in 70% train 30% test
    # TODO make a test loop
    X = np.reshape(X, (len(X), 24, 1))
    Y = np.reshape(Y, (len(Y), 3, 1))

    net.train(mse, mse_prime, X, Y, 250, 0.1)

    for x_test, y_test in zip(X, Y):
        output = net.feed_forward(x_test)
        output_index = list(output).index(max(list(output)))
        target_index = list(y_test).index(max(list(y_test)))
        print(f"target = {target_index}, output = {output_index}")
        print("============================================")

    output = net.feed_forward(X[0])
    output_index = list(output).index(max(list(output)))
    print(output_index)

    game = Game(10, 3, net)
    game.run()
