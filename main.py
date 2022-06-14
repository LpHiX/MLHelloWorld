from layer import Layer
from param_reader import ParamReader
from mnist_reader import MNISTReader

param_reader = ParamReader()
param_reader.load()

print("Printing the YAML file")
param_reader.print_layers()
param_reader.close_file()

mnreader = MNISTReader()
image, label = mnreader.return_set(0)

param_reader.output(image, True)

mnreader.plot_data(1)
