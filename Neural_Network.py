from typing import Any, Union

import numpy as np
from numpy import exp, array, random, dot

class NN:
    def __init__(self):
        random.seed(1)
        self.weights = 2 * random.random((16, 1)) - 1

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __relu(self, x):
        return np.maximum(0, x)

    def __expo(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def train(self, inputs, outputs, num):
        for x in range(num):
            output = self.think(inputs)
            error = outputs - output
            adjustment = -0.01*dot(inputs.T, error)
            self.weights += adjustment

    def think(self, inputs):
        result = dot(inputs, self.weights)
        return result