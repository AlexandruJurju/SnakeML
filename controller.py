import copy
import csv

from Neural.neural_network import mse, mse_prime
from model import *
from view import View


class TrainingExample:
    def __init__(self, model: List[str], predictions: List[float], current_direction: Direction):
        self.model = model
        self.predictions = predictions
        self.current_direction = current_direction


# TODO add options for using different neural networks
# TODO add options for using different directions 4,8,16
class Controller:
    def __init__(self, model: Model, view: View):
        self.running = True
        self.model = model
        self.view = view

        self.train_network(self.model.snake.brain)

    def run(self) -> None:
        training_examples = []
        while self.running:
            if ViewConsts.DRAW:
                self.view.clear_window()

            vision_lines = get_vision_lines(self.model.board)

            neural_net_prediction = self.model.get_nn_output(vision_lines)
            nn_input = get_parameters_in_nn_input_form(vision_lines, self.model.snake.direction)

            # max maximum in neural net output 1, others 0
            example_prediction = np.where(neural_net_prediction == np.max(neural_net_prediction), 1, 0)
            example = TrainingExample(copy.deepcopy(self.model.board), example_prediction.ravel().tolist(), self.model.snake.direction)
            training_examples.append(example)

            if ViewConsts.DRAW:
                self.view.draw_board(self.model.board)
                self.view.draw_vision_lines(self.model, vision_lines)
                self.view.draw_neural_network(self.model, vision_lines, nn_input, neural_net_prediction)
                self.view.draw_score(self.model.snake.score)
                self.view.draw_ttl(self.model.snake.ttl)
                self.view.update_window()

            next_direction = self.model.get_nn_output_4directions(neural_net_prediction)
            self.running = self.model.move_in_direction(next_direction)

            if not self.running:
                if ViewConsts.DRAW:
                    self.view.draw_dead(self.model.board)

                self.evaluate_live_examples_4d(training_examples)
                training_examples.clear()

                # TODO BAD REINIT, TO BE REMOVED
                # TODO train data , search file like a dictionary to find if there are conflicting data
                self.model.snake.brain.reinit_weights_and_biases()
                self.train_network(self.model.snake.brain)

                # TODO add reinit function in model
                self.model = Model(BOARD_SIZE, START_SNAKE_SIZE, self.model.snake.brain)

                self.running = True

    def read_training_models(self) -> Tuple:
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

    def train_network(self, network: NeuralNetwork) -> None:
        x, y = self.read_training_models()

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

    def write_examples_to_csv_4d(self, examples: List[TrainingExample]) -> None:
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

    def evaluate_live_examples_4d(self, examples: List[TrainingExample]) -> None:
        evaluated = []

        for example in examples:
            print(f"Model \n {np.matrix(example.model)} \n")
            print(f"Current Direction : {example.current_direction} \n")
            print(f"Prediction UP : {example.predictions[0]}")
            print(f"Prediction DOWN : {example.predictions[1]}")
            print(f"Prediction LEFT : {example.predictions[2]}")
            print(f"Prediction RIGHT : {example.predictions[3]}")
            print()

            if ViewConsts.DRAW:
                self.view.clear_window()
                self.view.draw_board(example.model)
                self.view.update_window()

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

        self.write_examples_to_csv_4d(evaluated)
