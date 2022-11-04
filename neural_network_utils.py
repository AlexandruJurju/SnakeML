import numpy as np


def relu(x):
    return np.maximum(0.0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def tanh(x):
    return np.tanh(x)
