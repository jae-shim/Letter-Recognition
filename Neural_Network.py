from typing import Any, Union

import numpy as np
from numpy import exp, array, random, dot

class NN:
    def __init__(self):
        random.seed(26)
        self.weights = 2 * random.random((16, 1)) - 1
        print(self.weights.shape)

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __relu(self, x):
        return np.maximum(0, x)

    def __expo(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def train(self, inputs, outputs, num):
        for x in range(num):
            output = self.think(inputs)
            #cost = (output - outputs)**2
            gradient = 2 * (output - outputs) * inputs
            adjustment = np.empty([16, 1], 'float64')
            for y in range(16):
                adjustment[y] = gradient.T[y].mean()
            self.weights -= adjustment

    def think(self, inputs):
        result = self.__sigmoid(dot(inputs, self.weights))
        return result