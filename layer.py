import numpy as np


class Layer:
    def __init__(self, inputs, neurons, weights=None, biases=None):
        self.inputs = inputs
        self.neurons = neurons
        if weights is None:
            self.weights = [[1] * inputs for _ in range(neurons)]
        else:
            self.weights = weights
        if biases is None:
            self.biases = [1] * neurons
        else:
            self.biases = biases

    def to_dict(self):
        output = {
            "inputs": self.inputs,
            "neurons": self.neurons,
            "biases": self.biases,
            "weights": self.weights
        }
        return output

    def output(self, activation):
        return (np.matmul(self.weights, activation) + self.biases).tolist()