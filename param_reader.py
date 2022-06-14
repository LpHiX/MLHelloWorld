import time

import yaml

from layer import Layer


class ParamReader:
    def __init__(self):
        self.param_file = None
        self.param_yaml = None
        self.layers = []

    def output(self, inputs):
        activations = inputs
        for layer in self.layers:
            activations = layer.output(activations)
            print(activations)
        return activations

    def load(self):
        if self.check_empty():
            self.write_defaults()
        for layer_number, layer_details in self.param_yaml.items():
            self.layers.append(Layer(
                layer_details["inputs"],
                layer_details["neurons"],
                layer_details["weights"],
                layer_details["biases"]
            ))
    def check_empty(self):
        f = open("neural_parameters.yaml", "a")
        f.close()
        self.param_file = open("neural_parameters.yaml", "r+")
        self.param_yaml = yaml.full_load(self.param_file)
        self.param_file.close()
        return self.param_yaml is None

    def write_defaults(self):
        self.param_file = open("neural_parameters.yaml", "w")

        layer0 = Layer(3, 3)
        layer1 = Layer(3, 3)
        yaml.dump({0: layer0.to_dict(), 1: layer1.to_dict()}, self.param_file)
        self.param_file = open("neural_parameters.yaml", "r+")
        self.param_yaml = yaml.full_load(self.param_file)

    def get_layers(self):
        for layer in self.layers:
            print(layer.to_dict())

    def close_file(self):
        self.param_file.close()