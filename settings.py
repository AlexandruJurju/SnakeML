class SnakeSettings:
    START_SNAKE_SIZE = 3
    SNAKE_MAX_TTL = 150


class NNSettings:
    INPUT_DIRECTION_COUNT = 4
    VISION_LINES_RETURN_TYPE = "boolean"

    NN_INPUT_NEURON_COUNT = INPUT_DIRECTION_COUNT * 3 + 4
    NN_HIDDEN_NEURON_COUNT = 24
    NN_OUTPUT_NEURON_COUNT = 4 if INPUT_DIRECTION_COUNT == 4 or INPUT_DIRECTION_COUNT == 8 else 3
    TRAIN_DATA_FILE_LOCATION = "Neural/train_data_" + str(NN_OUTPUT_NEURON_COUNT) + "_output_directions.csv"


class Genetic:
    MUTATION_CHANCE = 0.05
