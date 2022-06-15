import time

import yaml

from layer import Layer
from network_setup import ParamSetup


class ParamReader:
    def __init__(self):
        self.param_file = None
        self.param_yaml = None
        self.layers = []

    def output(self, inputs, print_output=False):
        activations = inputs
        for layer in self.layers:
            activations = layer.output(activations)
            if print_output:
                print("layer", layer, ":", activations)
        return activations

    def cost_layer(self, correct, final_layer):
        output = []
        for i in range(len(final_layer)):
            if i == correct:
                output.append((1 - final_layer[i]) ** 2)
            else:
                output.append(final_layer[i] ** 2)
        return output

    def load(self):
        if self.check_empty():
            self.write_defaults()
        for layer_number, layer_details in self.param_yaml.items():
            self.layers.append(Layer(
                layer_details["inputs"],
                layer_details["neurons"],
                layer_details["weights"],
                layer_details["biases"],
            ))

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

    def print_yaml(self):
        print("Printing the YAML file")
        for layer in self.layers:
            print(layer.to_dict())

    def close_file(self):
        self.param_file.close()
