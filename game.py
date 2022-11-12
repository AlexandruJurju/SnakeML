import pygame

from constants import *
from model import *
from Neural.neural_network_utils import *


class Game:
    def __init__(self, model_size: int, snake_size: int):
        pygame.init()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.fps_clock = pygame.time.Clock()

        net = KerasNetwork()
        net.add(Dense(28, 16))
        net.add(Activation(relu, relu))
        net.add(Dense(16, 4))
        net.add(Activation(softmax, softmax))

        self.running = True
        self.model = Model(model_size, snake_size, net)
        self.direction = Direction.UP

    def __draw_network(self):
        nn_font = pygame.font.SysFont("arial", 15)
        board_end = SQUARE_SIZE * self.model.size + OFFSET_BOARD_X

        input_label_offset_x = board_end + 50
        input_label_offset_y = 20
        label_distance_between = 30

        neuron_width_between = 200
        neuron_height_between = 30
        neuron_offset_x = input_label_offset_x + 50
        neuron_offset_y = 20
        neuron_radius = 10
        nn_inputs = self.model.get_parameters_in_nn_input_form()

        # for drawing vision lines names next to input neurons
        count = 0
        param_type = ["W", "A", "S"]
        for line in self.model.get_vision_lines(8, "boolean"):
            for param in param_type:
                line_label = nn_font.render(line + param, True, (255, 255, 255))
                self.window.blit(line_label, [input_label_offset_x, label_distance_between * count + input_label_offset_y - 10])
                count += 1

        # max distance is used to center the neurons in the next layers, formula for new yOffset is (yLenghtPrevius - yLenghtCurrent) / 2
        max_y_distance = 0

        # line start and line end are lists that contain the positions of the neurons
        # the lists are used for drawing the lines between neurons
        line_start = []
        line_end = []

        dense_layers = []
        for layer in self.model.snake.brain.layers:
            if type(layer) is Dense:
                dense_layers.append(layer)

        inputs = self.model.get_parameters_in_nn_input_form()
        outputs = self.model.get_nn_output()

        # draw neurons
        for i, layer in enumerate(dense_layers):
            # if it's the first layer, draw neurons using input
            if i == 0:
                for j in range(layer.input_size):
                    # draw the neuron
                    pygame.draw.circle(self.window, COLOR_WHITE, (neuron_offset_x, neuron_height_between * j + neuron_offset_y), neuron_radius, width=1)

                    # calculate neuron green color intensity using input parameters
                    activation_color = (0, round(255 * inputs[j][0]), 0)

                    # draw green circle inside neuron with activation color
                    pygame.draw.circle(self.window, activation_color, (neuron_offset_x, neuron_height_between * j + neuron_offset_y), neuron_radius - 1)

                    # append neuron position to start list
                    line_start.append([neuron_offset_x, neuron_height_between * j + neuron_offset_y])

                # increment current offset to obtain OX offset for next layer
                neuron_offset_x += neuron_width_between

                # calculate maxYDistance for centering neurons of next layer
                max_y_distance = neuron_height_between * layer.input_size

            # calculate current y distance for centering neurons
            current_y_distance = layer.output_size * neuron_height_between

            # calculate offset using distance of prev layer and distance of current layer
            hidden_offset_y = (max_y_distance - current_y_distance) // 2

            for j in range(layer.output_size):
                # if current layer is the last layer in NN then neurons are colored using NN output results
                if i == len(dense_layers) - 1:
                    pygame.draw.circle(self.window, (0, 255 * outputs[j], 0), (neuron_offset_x, neuron_height_between * j + neuron_offset_y + hidden_offset_y),
                                       neuron_radius - 1)
                    pygame.draw.circle(self.window, COLOR_WHITE, (neuron_offset_x, neuron_height_between * j + neuron_offset_y + hidden_offset_y), neuron_radius - 1, width=1)

                    # write direction name in output
                    # TODO direction output matches NN output
                    match j:
                        case 0:
                            direction = "UP"
                        case 1:
                            direction = "DOWN"
                        case 2:
                            direction = "LEFT"
                        case 3:
                            direction = "RIGHT"

                    line_label = nn_font.render(direction, True, (255, 255, 255))
                    self.window.blit(line_label, [neuron_offset_x + 10, neuron_height_between * j + neuron_offset_y + hidden_offset_y - 5])
                else:
                    # hidden neuron activation color
                    if self.model.snake.brain.layers[i + 1].output[j] <= 0:
                        color = (0, 0, 0)
                    else:
                        color = (0, 255, 0)
                    pygame.draw.circle(self.window, color, (neuron_offset_x, neuron_height_between * j + neuron_offset_y + hidden_offset_y), neuron_radius - 1)
                    pygame.draw.circle(self.window, COLOR_WHITE, (neuron_offset_x, neuron_height_between * j + neuron_offset_y + hidden_offset_y), neuron_radius - 1, width=1)

                # line end for drawing lines
                line_end.append([neuron_offset_x, neuron_height_between * j + neuron_offset_y + hidden_offset_y])
            neuron_offset_x += neuron_width_between

            # self.__draw_colored_lines_between_neurons(layer, line_end, line_start)
            self.__draw_lines_between_neurons(line_end, line_start)

            line_start = line_end
            line_end = []

    def __draw_lines_between_neurons(self, line_end, line_start):
        for k in range(len(line_end)):
            for l in range(len(line_start)):
                pygame.draw.line(self.window, COLOR_WHITE, line_start[l], line_end[k], width=1)

    def __draw_colored_lines_between_neurons(self, layer, line_end, line_start):
        for k in range(len(line_end)):
            for l in range(len(line_start)):
                if layer.weights[k][l] < 0:
                    color = (-(255 * layer.weights[k][l]), 0, 0)
                else:
                    color = (0, 100 * layer.weights[k][l], 255 * layer.weights[k][l])
                pygame.draw.line(self.window, color, line_start[l], line_end[k], width=1)

    def __draw_board(self):
        # use y,x for index in board instead of x,y because of changed logic
        # x is line y is column ; drawing x is column and y is line
        for x in range(self.model.size):
            for y in range(self.model.size):
                x_position = x * SQUARE_SIZE + OFFSET_BOARD_X
                y_position = y * SQUARE_SIZE + OFFSET_BOARD_Y

                match self.model.board[y, x]:
                    case "S":
                        pygame.draw.rect(self.window, COLOR_SNAKE, pygame.Rect(x_position, y_position, SQUARE_SIZE, SQUARE_SIZE))
                    case "W":
                        pygame.draw.rect(self.window, COLOR_WHITE, pygame.Rect(x_position, y_position, SQUARE_SIZE, SQUARE_SIZE))
                    case "A":
                        pygame.draw.rect(self.window, COLOR_APPLE, pygame.Rect(x_position, y_position, SQUARE_SIZE, SQUARE_SIZE))

                # draw snake head using another color
                head = self.model.snake.body[0]
                pygame.draw.rect(self.window, COLOR_SNAKE_HEAD,
                                 pygame.Rect(head[1] * SQUARE_SIZE + OFFSET_BOARD_X, head[0] * SQUARE_SIZE + OFFSET_BOARD_Y, SQUARE_SIZE, SQUARE_SIZE))

                # draw lines between squares
                pygame.draw.rect(self.window, COLOR_SQUARE_DELIMITER, pygame.Rect(x_position, y_position, SQUARE_SIZE, SQUARE_SIZE), width=1)

    def __draw_vision_lines(self):
        vision_lines = self.model.get_vision_lines(8, "boolean")
        font = pygame.font.SysFont("arial", 18)

        # loop over all lines in given vision lines
        for line in vision_lines:
            line_label = font.render(line, True, COLOR_BLACK)

            # render vision line text at wall position
            self.window.blit(line_label, [vision_lines[line].wall_coord[1] * SQUARE_SIZE + OFFSET_BOARD_X,
                                          vision_lines[line].wall_coord[0] * SQUARE_SIZE + OFFSET_BOARD_Y])

            # draw line from head to wall, draw before body and apple lines
            # drawing uses SQUARE_SIZE//2 so that lines go through the middle of the squares
            line_end_x = self.model.snake.body[0][1] * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_X
            line_end_y = self.model.snake.body[0][0] * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_Y

            # draw line form snake head until wall block
            pygame.draw.line(self.window, COLOR_APPLE,
                             ((vision_lines[line].wall_coord[1]) * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_X,
                              (vision_lines[line].wall_coord[0]) * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_Y),
                             (line_end_x, line_end_y), width=1)

            # draw another line from snake head to first segment found
            if vision_lines[line].segment_coord is not None:
                pygame.draw.line(self.window, COLOR_RED,
                                 (vision_lines[line].segment_coord[1] * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_X,
                                  vision_lines[line].segment_coord[0] * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_Y),
                                 (line_end_x, line_end_y), width=5)

            # draw another line from snake to apple if apple is found
            if vision_lines[line].apple_coord is not None:
                pygame.draw.line(self.window, COLOR_GREEN,
                                 (vision_lines[line].apple_coord[1] * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_X,
                                  vision_lines[line].apple_coord[0] * SQUARE_SIZE + SQUARE_SIZE // 2 + OFFSET_BOARD_Y),
                                 (line_end_x, line_end_y), width=5)

    def __manage_key_inputs(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                match event.key:
                    case pygame.K_ESCAPE:
                        self.running = False
                    case pygame.K_UP:
                        self.direction = Direction.UP
                    case pygame.K_DOWN:
                        self.direction = Direction.DOWN
                    case pygame.K_LEFT:
                        self.direction = Direction.LEFT
                    case pygame.K_RIGHT:
                        self.direction = Direction.RIGHT

    def run(self) -> None:
        while self.running:
            self.__manage_key_inputs()
            self.window.fill(COLOR_BACKGROUND)

            # next_direction = self.model.get_direction_from_nn_output()
            # self.running = self.model.move_in_direction(next_direction)

            self.model.move_random_direction()

            if self.running:
                self.__draw_board()
                # self.__draw_vision_lines()
                self.__draw_network()

                pygame.display.update()
                self.fps_clock.tick(MAX_FPS)
