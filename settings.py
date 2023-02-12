class SnakeSettings:
    START_SNAKE_SIZE = 3
    SNAKE_MAX_TTL = 150


class NNSettings:
    INPUT_DIRECTION_COUNT = 4
    VISION_LINES_RETURN_TYPE = "boolean"

    INPUT_NEURON_COUNT = INPUT_DIRECTION_COUNT * 3 + 4
    HIDDEN_NEURON_COUNT = 24
    OUTPUT_NEURON_COUNT = 4 if INPUT_DIRECTION_COUNT == 4 or INPUT_DIRECTION_COUNT == 8 else 3
    TRAIN_DATA_FILE_LOCATION = "Neural/train_data_" + str(OUTPUT_NEURON_COUNT) + "_output_directions.json"


class GeneticSettings:
    MUTATION_CHANCE = 0.05
    SBX_ETA = 100.0
    POPULATION_COUNT = 1000
