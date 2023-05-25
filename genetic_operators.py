import copy
import random
from typing import Tuple

from model import Snake
from neural_network import *


# operators from pymoo
# https://github.com/anyoptimization/pymoo/tree/main/pymoo/operators

def roulette_selection(population: List[Snake], selection_count: int) -> List[Snake]:
    """
    In Roulette selection the chance for and individual to be selected is directly proportional with that individual's fitness
    :param population: list containing all individuals
    :param selection_count: number of individuals to be extracted from the population
    :return: list on individuals selected from the population
    """
    total_fitness = sum(individual.fitness for individual in population)
    probabilities = [individual.fitness / total_fitness for individual in population]
    selection = random.choices(population, probabilities, k=selection_count)
    return selection


def roulette_selection_negative(population: List[Snake], selection_count: int) -> List[Snake]:
    """
     In Roulette selection the chance for and individual to be selected is directly proportional with that individual's fitness
     :param population: list containing all individuals
     :param selection_count: number of individuals to be extracted from the population
     :return: list on individuals selected from the population
     """

    selected = []
    min_fitness = min(individual.fitness for individual in population)
    individuals_after_shift = [individual.fitness - min_fitness for individual in population]
    total_population_fitness = sum(individuals_after_shift)

    # Choose a random value between 0 and sum_pop_fitness
    # Loop over individuals in population and sum their fitness in current_fitness
    for i in range(selection_count):
        random_fitness = random.uniform(0, total_population_fitness)
        current_fitness = 0

        for individual, fitness in zip(population, individuals_after_shift):
            current_fitness += fitness
            if current_fitness >= random_fitness:
                selected.append(individual)
                break

    return selected


def tournament_selection(population: List[Snake], selection_count: int, tournament_size: int) -> List[Snake]:
    """
    The function selects random individuals from the population and returns the fittest one from the selected ones

    :param population: list containing all individuals
    :param selection_count: number of individuals to be extracted from the population
    :param tournament_size: number of random individuals selected to enter a tournament
    :return: list on individuals selected from the population
    """

    selected = []
    for i in range(selection_count):
        tournament = np.random.choice(population, tournament_size)
        winner = max(tournament, key=lambda individual: individual.fitness)
        selected.append(winner)
    return selected


def elitist_selection(population: List[Snake], selection_count: int) -> List[Snake]:
    """
    Elitist selection sorts the population by fitness in descending order and selects the individuals with the
    greatest fitness first

    :param population: list containing all individuals
    :param selection_count: number of individuals to be extracted from the population
    :return: list on individuals selected from the population
    """

    selected = sorted(population, key=lambda individual: individual.fitness, reverse=True)
    # return first SELECTION_COUNT individuals from the sorted population
    return selected[:selection_count]


def one_point_crossover(parent1_matrix: np.ndarray, parent2_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    In one point crossover a single index is used to separate genetic material from the parents.

    :param parent1_matrix: parent1 weight or bias numpy array
    :param parent2_matrix: parent 2 weight or bias numpy array
    :return Tuple with the generator offspring numpy array from the given parents arrays:
    """
    matrix_row, matrix_col = np.shape(parent1_matrix)

    child1_matrix = parent1_matrix.copy()
    child2_matrix = parent2_matrix.copy()

    crossover_point = (np.random.randint(0, matrix_row), np.random.randint(0, matrix_col))

    child1_matrix[:crossover_point[0], :] = parent2_matrix[:crossover_point[0], :]
    child1_matrix[crossover_point[0], :crossover_point[1]] = parent2_matrix[crossover_point[0], :crossover_point[1]]

    child2_matrix[:crossover_point[0], :] = parent1_matrix[:crossover_point[0], :]
    child2_matrix[crossover_point[0], :crossover_point[1]] = parent1_matrix[crossover_point[0], :crossover_point[1]]

    return child1_matrix, child2_matrix


# def one_point_crossover(parent1: NeuralNetwork, parent2: NeuralNetwork) -> Tuple[NeuralNetwork, NeuralNetwork]:
#     child1 = copy.deepcopy(parent1)
#     child2 = copy.deepcopy(parent2)
#
#     child1_dense_layers = child1.get_dense_layers()
#     child2_dense_layers = child2.get_dense_layers()
#
#     parent1_dense_layers = parent1.get_dense_layers()
#     parent2_dense_layers = parent2.get_dense_layers()
#
#     for i in range(len(parent1_dense_layers)):
#         matrix_rows, matrix_cols = np.shape(parent1_dense_layers[i].weights)
#         crossover_row = np.random.randint(0, matrix_rows)
#         crossover_col = np.random.randint(0, matrix_cols)
#
#         # this method interchanges all values until index
#         # x x x x   y y y y     x x x x
#         # x x x x   y y y y     x x y y
#         # x x x x   y y y y     y y y y
#         # x x x x   y y y y     y y y y
#         child1_dense_layers[i].weights[:crossover_row, :] = parent2_dense_layers[i].weights[:crossover_row, :]
#         child1_dense_layers[i].weights[crossover_row, :crossover_col] = parent2_dense_layers[i].weights[crossover_row, :crossover_col]
#
#         child2_dense_layers[i].weights[:crossover_row, :] = parent1_dense_layers[i].weights[:crossover_row, :]
#         child2_dense_layers[i].weights[crossover_row, :crossover_col] = parent1_dense_layers[i].weights[crossover_row, :crossover_col]
#
#         # This method interchanges just FULL rows
#         # child1_dense_layers[i].weights = np.concatenate((parent1_dense_layers[i].weights[:rand_row, :], parent2_dense_layers[i].weights[rand_row:, :]), axis=0)
#         # child2_dense_layers[i].weights = np.concatenate((parent2_dense_layers[i].weights[:rand_row, :], parent1_dense_layers[i].weights[rand_row:, :]), axis=0)
#
#         # this method interchanges both rows and cols, maybe bad because it interchanges blocks between parents, this is two point
#         # x x x x   y y y y     x x x x
#         # x x x x   y y y y     x y y x
#         # x x x x   y y y y     x y y x
#         # x x x x   y y y y     x x x x
#         # child1_dense_layers[i].weights[:crossover_row, :crossover_col] = parent1_dense_layers[i].weights[:crossover_row, :crossover_col]
#         # child1_dense_layers[i].weights[crossover_row:, crossover_col:] = parent2_dense_layers[i].weights[crossover_row:, crossover_col:]
#         # child2_dense_layers[i].weights[:crossover_row, :crossover_col] = parent2_dense_layers[i].weights[:crossover_row, :crossover_col]
#         # child2_dense_layers[i].weights[crossover_row:, crossover_col:] = parent1_dense_layers[i].weights[crossover_row:, crossover_col:]
#
#         matrix_rows, matrix_cols = np.shape(parent1_dense_layers[i].bias)
#         crossover_row = np.random.randint(0, matrix_rows)
#         crossover_col = np.random.randint(0, matrix_cols)
#
#         child1_dense_layers[i].bias[:crossover_row, :] = parent2_dense_layers[i].bias[:crossover_row, :]
#         child1_dense_layers[i].bias[crossover_row, :crossover_col] = parent2_dense_layers[i].bias[crossover_row, :crossover_col]
#
#         child2_dense_layers[i].bias[:crossover_row, :] = parent1_dense_layers[i].bias[:crossover_row, :]
#         child2_dense_layers[i].bias[crossover_row, :crossover_col] = parent1_dense_layers[i].bias[crossover_row, :crossover_col]
#
#     return child1, child2
#

def two_point_crossover(parent1_matrix: np.ndarray, parent2_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    matrix_row, matrix_col = np.shape(parent1_matrix)

    child1_matrix = np.empty((matrix_row, matrix_col))
    child2_matrix = np.empty((matrix_row, matrix_col))

    point1 = (np.random.randint(0, matrix_row), np.random.randint(0, matrix_col))
    point2 = (np.random.randint(0, matrix_row), np.random.randint(0, matrix_col))

    for i in range(matrix_row):
        for j in range(matrix_col):
            if point1 < (i, j) < point2:
                child1_matrix[i, j] = parent2_matrix[i, j]
                child2_matrix[i, j] = parent1_matrix[i, j]
            else:
                child1_matrix[i, j] = parent1_matrix[i, j]
                child2_matrix[i, j] = parent2_matrix[i, j]

    return child1_matrix, child2_matrix


def uniform_crossover(parent1_matrix: np.ndarray, parent2_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    child1_matrix = np.empty_like(parent1_matrix)
    child2_matrix = np.empty_like(parent2_matrix)

    probability_mask = np.random.uniform(0, 1, parent1_matrix.shape)

    child1_matrix[probability_mask <= 0.5] = parent1_matrix[probability_mask <= 0.5]
    child1_matrix[probability_mask > 0.5] = parent2_matrix[probability_mask > 0.5]

    child2_matrix[probability_mask <= 0.5] = parent2_matrix[probability_mask <= 0.5]
    child2_matrix[probability_mask > 0.5] = parent1_matrix[probability_mask > 0.5]

    return child1_matrix, child2_matrix


def calculate_bq(u: float, eta: float) -> float:
    """
    Calculate the simulated binary crossover (SBX) scaling factor 'bq' given a random factor 'u' and a distribution index 'eta'.

    :param u: A random scaling factor between 0 and 1.
    :param eta: The distribution index, which controls the offspring spread. A high eta value generates offspring
    closer to the parents, whereas a low eta value generates more diverse offspring.
    :return: The SBX scaling factor 'bq'.
    """
    if u <= 0.5:
        bq = (2 * u) ** (1 / (eta + 1))
    else:
        bq = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
    return bq


def sbx(parent1_matrix: np.ndarray, parent2_matrix: np.ndarray, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulated binary crossover operator for genetic algorithms.

    :param parent1_matrix: First parent chromosome as a NumPy array.
    :param parent2_matrix: Second parent chromosome as a NumPy array.
    :param eta: The distribution index, which controls the offspring spread. A high eta value generates offspring
    closer to the parents, whereas a low eta value generates more diverse offspring.
    :return: A tuple of two child chromosomes as NumPy arrays.
    """
    u = np.random.uniform(0, 1)

    # Calculate the scaling factor 'bq' using the 'calculate_bq' function.
    bq = calculate_bq(u, eta)

    child1 = 0.5 * ((1 + bq) * parent1_matrix + (1 - bq) * parent2_matrix)
    child2 = 0.5 * ((1 - bq) * parent1_matrix + (1 + bq) * parent2_matrix)

    return child1, child2


def gaussian_mutation(matrix: np.ndarray, mutation_rate: float) -> np.ndarray:
    after_mutation = matrix
    mutation_array = np.random.random(after_mutation.shape) < mutation_rate

    # default mean is 0, standard deviation is 1
    gauss_values = np.random.normal(size=after_mutation.shape)
    after_mutation[mutation_array] += gauss_values[mutation_array]

    return after_mutation


def point_mutation():
    pass


def uniform_mutation(matrix: np.ndarray, mutation_rate: float) -> np.ndarray:
    after_mutation = matrix
    mutation_array = np.random.random(after_mutation.shape) < mutation_rate

    mutation_values = np.random.uniform(-1, 1, size=after_mutation.shape)
    after_mutation[mutation_array] = mutation_values[mutation_array]

    return after_mutation


def full_mutation(individual: NeuralNetwork, mutation_rate: float, operator) -> None:
    individual_dense_layers = individual.get_dense_layers()

    for layer in individual_dense_layers:
        layer.weights = operator(layer.weights, mutation_rate)
        layer.bias = operator(layer.bias, mutation_rate)


def full_crossover(parent1: NeuralNetwork, parent2: NeuralNetwork, operator) -> Tuple[NeuralNetwork, NeuralNetwork]:
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    child1_dense_layers = child1.get_dense_layers()
    child2_dense_layers = child2.get_dense_layers()

    parent1_dense_layers = parent1.get_dense_layers()
    parent2_dense_layers = parent2.get_dense_layers()

    for i in range(len(parent1_dense_layers)):
        child1_dense_layers[i].weights, child2_dense_layers[i].weights = operator(parent1_dense_layers[i].weights, parent2_dense_layers[i].weights)
        child1_dense_layers[i].bias, child2_dense_layers[i].bias = operator(parent1_dense_layers[i].bias, parent2_dense_layers[i].bias)

    return child1, child2
