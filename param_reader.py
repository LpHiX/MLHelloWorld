import time

import yaml

from layer import Layer
from network_setup import ParamSetup


class ParamReader:
    def __init__(self):
        self.param_file = None
        self.param_yaml = None
        self.layers = {}

    def output(self, inputs, print_output=False):
        activations = inputs
        for layer in self.layers:
            activations = self.layers[layer].output(activations)
            if print_output:
                print("layer",layer, ":", activations)
        return activations

    def load(self):
        if self.check_empty():
            self.write_defaults()
        for layer_number, layer_details in self.param_yaml.items():
            self.layers[layer_number] = Layer(
                layer_details["inputs"],
                layer_details["neurons"],
                layer_details["weights"],
                layer_details["biases"],
            )

    def check_empty(self):
        f = open("neural_parameters.yaml", "a")
        f.close()
        self.param_file = open("neural_parameters.yaml", "r+")
        self.param_yaml = yaml.full_load(self.param_file)
        self.param_file.close()
        return self.param_yaml is None

    def write_defaults(self):
        param_setup = ParamSetup()
        param_setup.load()

        self.param_file = open("neural_parameters.yaml", "w")
        param_setup.dump(self.param_file)

        self.param_file = open("neural_parameters.yaml", "r+")
        self.param_yaml = yaml.full_load(self.param_file)

    def print_layers(self):
        for layer in self.layers:
            print(self.layers[layer].to_dict())

    def close_file(self):
        self.param_file.close()