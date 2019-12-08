from typing import Any, Union

import numpy as np
from numpy import exp, array, random, dot

class NN:
    def __init__(self):
        random.seed(1)
        self.weights = 26 * np.random.random((16, 1))

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __relu(self, x):
        return np.maximum(0, x)

    def __expo(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def train(self, inputs, outputs, num):
        for x in range(num):
            output = self.think(inputs)
            error = output - outputs
            #sigDer = outputs * (1 - outputs)
            #adjustment = np.empty([16, 1], 'float64')
            #for y in range(16):
            #    adjustment[y] = gradient.T[y].mean()
            adjustment = 0.01*dot(inputs.T, error)
            #self.weights += np.dot(inputs.T, error*sigDer)
            self.weights += adjustment

    def think(self, inputs):
        #result = self.__sigmoid(dot(inputs, self.weights))
        result = dot(inputs, self.weights)
        return result
