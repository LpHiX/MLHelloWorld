import time

import yaml


class ParamReader:
    def __init__(self):
        f = open("neural_parameters.yaml", "a")
        f.close()

        self.param_file = open("neural_parameters.yaml")

        self.param_yaml = yaml.full_load(self.param_file)

    def write_defaults(self):
        one_char = self.param_file.read(1)
        if one_char:
            return
        self.param_file.write("tesssssting")

    def print_file(self):
        for item, doc in self.param_yaml.items():
            print(item, ":", doc)

    def close_file(self):
        self.param_file.close()
