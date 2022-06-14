import yaml

from layer import Layer


class ParamSetup:
    def __init__(self):
        self.setup_file = None
        self.setup_yaml = None
        self.layers_info = []

    def load(self):
        if self.check_empty():
            self.write_defaults()
        for layer_number, layer_details in self.setup_yaml.items():
            self.layers_info.append(layer_details)

    def check_empty(self):
        f = open("network_setup.yaml", "a")
        f.close()
        self.setup_file = open("network_setup.yaml", "r+")
        self.setup_yaml = yaml.full_load(self.setup_file)
        self.setup_file.close()
        return self.setup_yaml is None

    def write_defaults(self):
        self.setup_file = open("network_setup.yaml", "w")
        yaml.dump({
            0: {"inputs": 784, "neurons": 16},
            1: {"inputs": 16, "neurons": 16},
            2: {"inputs": 16, "neurons": 10}
        }, self.setup_file)
        self.setup_file = open("network_setup.yaml", "r+")
        self.setup_yaml = yaml.full_load(self.setup_file)

    def dump(self, param_file):
        layers_dictionary = {}
        index = 0
        for info in self.layers_info:
            layer = Layer(info["inputs"], info["neurons"])
            layers_dictionary[index] = layer.to_dict()
            index += 1

        yaml.dump(layers_dictionary, param_file)
        param_file.close()
