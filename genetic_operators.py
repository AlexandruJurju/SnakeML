import copy
import random
from typing import List, Tuple

import numpy as np

from Neural.neural_network import *
from model import Individual


# operators from pymoo
# https://github.com/anyoptimization/pymoo/tree/main/pymoo/operators

def roulette_selection(population: List[Individual], selection_count: int) -> List[Individual]:
    """
    In Roulette selection the chance for and individual to be selected is directly proportional with that individual's fitness
    :param population: list containing all individuals
    :param selection_count: number of individuals to be extracted from the population
    :return: list on individuals selected from the population
    """
    selected = []
    total_population_fitness = sum(individual.fitness for individual in population)

    # Choose a random value between 0 and sum_pop_fitness
    # Loop over individuals in population and sum their fitness in current_fitness
    for i in range(selection_count):
        random_fitness = random.uniform(0, total_population_fitness)
        current_fitness = 0

        for individual in population:
            current_fitness += individual.fitness
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
    matrix_row, matrix_col = np.shape(parent1_chromosome)

    child1 = parent1_chromosome.copy()
    child2 = parent2_chromosome.copy()

    crossover_row = np.random.randint(0, matrix_row)
    crossover_col = np.random.randint(0, matrix_col)

    child1[:crossover_row, :] = parent2_chromosome[:crossover_row, :]
    child1[crossover_row, :crossover_col] = parent2_chromosome[crossover_row, :crossover_col]

    child2[:crossover_row, :] = parent1_chromosome[:crossover_row, :]
    child2[crossover_row, :crossover_col] = parent1_chromosome[crossover_row, :crossover_col]

    return child1, child2


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

def two_point_crossover(parent1: NeuralNetwork, parent2: NeuralNetwork) -> Tuple[NeuralNetwork, NeuralNetwork]:
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    child1_dense_layers = child1.get_dense_layers()
    child2_dense_layers = child2.get_dense_layers()

    parent1_dense_layers = parent1.get_dense_layers()
    parent2_dense_layers = parent2.get_dense_layers()

    for i in range(len(parent1_dense_layers)):
        matrix_rows, matrix_cols = np.shape(parent1_dense_layers[i].weights)
        crossover_row = np.random.randint(0, matrix_rows)
        crossover_col = np.random.randint(0, matrix_cols)
        child1_dense_layers[i].weights[:crossover_row, :crossover_col] = parent1_dense_layers[i].weights[:crossover_row, :crossover_col]
        child1_dense_layers[i].weights[crossover_row:, crossover_col:] = parent2_dense_layers[i].weights[crossover_row:, crossover_col:]
        child2_dense_layers[i].weights[:crossover_row, :crossover_col] = parent2_dense_layers[i].weights[:crossover_row, :crossover_col]
        child2_dense_layers[i].weights[crossover_row:, crossover_col:] = parent1_dense_layers[i].weights[crossover_row:, crossover_col:]

        matrix_rows, matrix_cols = np.shape(parent1_dense_layers[i].bias)
        crossover_row = np.random.randint(0, matrix_rows)
        crossover_col = np.random.randint(0, matrix_cols)
        child1_dense_layers[i].bias[:crossover_row, :crossover_col] = parent1_dense_layers[i].bias[:crossover_row, :crossover_col]
        child1_dense_layers[i].bias[crossover_row:, crossover_col:] = parent2_dense_layers[i].bias[crossover_row:, crossover_col:]
        child2_dense_layers[i].bias[:crossover_row, :crossover_col] = parent2_dense_layers[i].bias[:crossover_row, :crossover_col]
        child2_dense_layers[i].bias[crossover_row:, crossover_col:] = parent1_dense_layers[i].bias[crossover_row:, crossover_col:]

    return child1, child2


def uniform_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pass


def calculate_bq(u: float, eta: float):
    if u <= 0.5:
        return np.power(2 * u, (1 / (eta + 1)))
    else:
        return np.power(1 / (2 * (1 - u)), (1 / (eta + 1)))


def sbx(parent1_chromosome: np.ndarray, parent2_chromosome: np.ndarray, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    For large values of eta there is a higher probability that offspring will be created near the parents.
    For small values of eta, offspring will be more distant from parents
    https://stackoverflow.com/questions/56263132/what-does-crossover-index-of-0-25-means-in-genetic-algorithm-for-real-encoding

    :param parent1_chromosome:
    :param parent2_chromosome:
    :param eta:
    :return:
    """
    # TODO maybe use a matrix vor values
    u = np.random.uniform(0, 1)
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
