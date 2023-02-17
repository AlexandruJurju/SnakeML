class SnakeSettings:
    START_SNAKE_SIZE = 3
    SNAKE_MAX_TTL = 100


class NNSettings:
    INPUT_DIRECTION_COUNT = 4
    # VISION_LINES_RETURN_TYPE = "boolean"

    # INPUT_NEURON_COUNT = INPUT_DIRECTION_COUNT * 3 + 4
    # HIDDEN_NEURON_COUNT = 24
    # OUTPUT_NEURON_COUNT = 4 if INPUT_DIRECTION_COUNT == 4 or INPUT_DIRECTION_COUNT == 8 else 3

    AVAILABLE_INPUT_DIRECTIONS = ["4", "8"]
    AVAILABLE_VISION_LINES_RETURN_TYPE = ["boolean", "distance"]

    # TRAIN_DATA_FILE_LOCATION = "Backpropagation_Training/" + str(INPUT_DIRECTION_COUNT) + "_in_directions_" + str(OUTPUT_NEURON_COUNT) + "_out_directions.json"
    GENETIC_NETWORK_FOLDER = "Genetic_Networks/"
    BACKPROPAGATION_NETWORK_FOLDER = "Backpropagation_Networks/"


class GeneticSettings:
    MUTATION_CHANCE = 0.05
    SBX_ETA = 100.0
    POPULATION_COUNT = 1000


class BoardSettings:
    BOARD_SIZE = 10
