import copy
import random
from typing import Tuple

from model import Individual
from neural_network import *


# operators from pymoo
# https://github.com/anyoptimization/pymoo/tree/main/pymoo/operators

def roulette_selection(population: List[Individual], selection_count: int) -> List[Individual]:
    """
    In Roulette selection the chance for and individual to be selected is directly proportional with that individual's fitness
    :param population: list containing all individuals
    :param selection_count: number of individuals to be extracted from the population
    :return: list on individuals selected from the population
    """

    total_population_fitness = sum(individual.fitness for individual in population)
    selection_probabilities = [individual.fitness / total_population_fitness for individual in population]
    return list(np.random.choice(population, size=selection_count, p=selection_probabilities))


def roulette_selection_negative(population: List[Individual], selection_count: int) -> List[Individual]:
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


def tournament_selection(population: List[Individual], selection_count: int, tournament_size: int) -> List[Individual]:
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


def elitist_selection(population: List[Individual], selection_count: int) -> List[Individual]:
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


def one_point_crossover(parent1_chromosome: np.ndarray, parent2_chromosome: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    In one point crossover a single index is used to separate genetic material from the parents.

    :param parent1_chromosome: parent1 weight or bias numpy array
    :param parent2_chromosome: parent 2 weight or bias numpy array
    :return Tuple with the generator offspring numpy array from the given parents arrays:
    """
    matrix_row, matrix_col = np.shape(parent1_chromosome)

    child1_chromosome = parent1_chromosome.copy()
    child2_chromosome = parent2_chromosome.copy()

    crossover_point = (np.random.randint(0, matrix_row), np.random.randint(0, matrix_col))

    child1_chromosome[:crossover_point[0], :] = parent2_chromosome[:crossover_point[0], :]
    child1_chromosome[crossover_point[0], :crossover_point[1]] = parent2_chromosome[crossover_point[0], :crossover_point[1]]

    child2_chromosome[:crossover_point[0], :] = parent1_chromosome[:crossover_point[0], :]
    child2_chromosome[crossover_point[0], :crossover_point[1]] = parent1_chromosome[crossover_point[0], :crossover_point[1]]

    return child1_chromosome, child2_chromosome


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

def two_point_crossover(parent1_chromosome: np.ndarray, parent2_chromosome: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    matrix_row, matrix_col = np.shape(parent1_chromosome)

    child1_chromosome = np.empty((matrix_row, matrix_col))
    child2_chromosome = np.empty((matrix_row, matrix_col))

    point1 = (np.random.randint(0, matrix_row), np.random.randint(0, matrix_col))
    point2 = (np.random.randint(0, matrix_row), np.random.randint(0, matrix_col))

    for i in range(matrix_row):
        for j in range(matrix_col):
            if point1 < (i, j) < point2:
                child1_chromosome[i, j] = parent2_chromosome[i, j]
                child2_chromosome[i, j] = parent1_chromosome[i, j]
            else:
                child1_chromosome[i, j] = parent1_chromosome[i, j]
                child2_chromosome[i, j] = parent2_chromosome[i, j]

    return child1_chromosome, child2_chromosome


def uniform_crossover(parent1_chromosome: np.ndarray, parent2_chromosome: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    child1_chromosome = np.empty_like(parent1_chromosome)
    child2_chromosome = np.empty_like(parent2_chromosome)

    probability_mask = np.random.uniform(0, 1, parent1_chromosome.shape)

    child1_chromosome[probability_mask <= 0.5] = parent1_chromosome[probability_mask <= 0.5]
    child1_chromosome[probability_mask > 0.5] = parent2_chromosome[probability_mask > 0.5]

    child2_chromosome[probability_mask <= 0.5] = parent2_chromosome[probability_mask <= 0.5]
    child2_chromosome[probability_mask > 0.5] = parent1_chromosome[probability_mask > 0.5]

    return child1_chromosome, child2_chromosome


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


def sbx(parent1_chromosome: np.ndarray, parent2_chromosome: np.ndarray, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulated binary crossover operator for genetic algorithms.

    :param parent1_chromosome: First parent chromosome as a NumPy array.
    :param parent2_chromosome: Second parent chromosome as a NumPy array.
    :param eta: The distribution index, which controls the offspring spread. A high eta value generates offspring
    closer to the parents, whereas a low eta value generates more diverse offspring.
    :return: A tuple of two child chromosomes as NumPy arrays.
    """
    u = np.random.uniform(0, 1)

    # Calculate the scaling factor 'bq' using the 'calculate_bq' function.
    bq = calculate_bq(u, eta)

    child1 = 0.5 * ((1 + bq) * parent1_chromosome + (1 - bq) * parent2_chromosome)
    child2 = 0.5 * ((1 - bq) * parent1_chromosome + (1 + bq) * parent2_chromosome)

    return child1, child2


def gaussian_mutation(chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
    after_mutation = chromosome

    mutation_array = np.random.random(after_mutation.shape) < mutation_rate
    gauss_values = np.random.normal(size=after_mutation.shape)
    after_mutation[mutation_array] += gauss_values[mutation_array]

    return after_mutation


def point_mutation():
    pass


def full_mutation(individual: NeuralNetwork, mutation_rate: float) -> None:
    individual_dense_layers = individual.get_dense_layers()

    for layer in individual_dense_layers:
        layer.weights = gaussian_mutation(layer.weights, mutation_rate)
        layer.bias = gaussian_mutation(layer.bias, mutation_rate)


def full_crossover(parent1: NeuralNetwork, parent2: NeuralNetwork) -> Tuple[NeuralNetwork, NeuralNetwork]:
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    child1_dense_layers = child1.get_dense_layers()
    child2_dense_layers = child2.get_dense_layers()

    parent1_dense_layers = parent1.get_dense_layers()
    parent2_dense_layers = parent2.get_dense_layers()

    for i in range(len(parent1_dense_layers)):
        child1_dense_layers[i].weights, child2_dense_layers[i].weights = one_point_crossover(parent1_dense_layers[i].weights, parent2_dense_layers[i].weights)
        child1_dense_layers[i].bias, child2_dense_layers[i].bias = one_point_crossover(parent1_dense_layers[i].bias, parent2_dense_layers[i].bias)

    return child1, child2
