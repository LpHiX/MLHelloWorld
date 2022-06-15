from typing import List

import numpy as np

from layer import Layer

CONST_R = 1


def partial_relu_er(x): return 1 if x >= 0 else 0


def relu(x): return x if x >= 0 else 0


def cost(x , desired_output): return (x - desired_output) ** 2


class Network:

    def __init__(self, layers: List[Layer]):
        self.weights = []  # 3D DONE
        self.biases = []  # 2D Done
        self.inputs = None  # DONE
        self.activations = []  # 2D: a[l,j] DONE
        self.partial_relus = []  # 2D: a[l,j] DONE
        self.costs = []  # DONE
        # Inputting weights and biases
        # ----------------------------
        for layer in layers:
            self.weights.append(layer.weights)
            self.biases.append(layer.biases)

        self.correct_labels = [0] * layers[len(layers)-1].neurons
        # layers is the input configuration, self.layers is the number of layers
        self.layers = len(layers)-1

    def train(self, inputs, correct_label):
        self.inputs = inputs
        self.correct_labels[correct_label] = 1

        # Propagate and calculate outputs
        # ----------------------------
        for i in range(len(self.weights)):
            if i == 0:
                pre_relu = np.matmul(self.weights[i], self.inputs) + self.biases[i]
            else:
                pre_relu = np.matmul(self.weights[i], self.activations[i - 1]) + self.biases[i]
            partial_relu_er_vectorized = np.vectorize(partial_relu_er)
            self.partial_relus.append(partial_relu_er_vectorized(pre_relu))
            relu_vectorized = np.vectorize(relu)
            self.activations.append(relu_vectorized(pre_relu))

        self.costs = cost(self.activations[self.layers], self.correct_labels)

    def print_outputs(self):
        for i, activation in enumerate(self.activations):
            print(i, ":", activation)
        print("Cost:", self.costs)
