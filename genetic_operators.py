import copy
import random
from typing import List, Tuple

import numpy as np

from Neural.neural_network import *
from model import Individual


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


def one_point_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    matrix_row, matrix_col = np.shape(parent1)

    child1 = parent1.copy()
    child2 = parent2.copy()

    crossover_row = np.random.randint(0, matrix_row)
    crossover_col = np.random.randint(0, matrix_col)

    child1[:crossover_row, :] = parent2[:crossover_row, :]
    child1[crossover_row, :crossover_col] = parent2[crossover_row, :crossover_col]

    child2[:crossover_row, :] = parent1[:crossover_row, :]
    child2[crossover_row, :crossover_col] = parent1[crossover_row, :crossover_col]

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


def simulated_binary_crossover():
    pass


def gaussian_mutation():
    pass


def point_mutation():
    pass
