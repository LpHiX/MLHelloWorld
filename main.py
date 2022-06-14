from layer import Layer
from param_reader import ParamReader
from mnist_reader import MNISTReader

# mnreader = MNISTReader()
# mnreader.plot_data(1)

param_reader = ParamReader()
param_reader.load()

print("Printing the YAML file")
print(param_reader.get_layers())
param_reader.close_file()

print(param_reader.output([1] * 3))