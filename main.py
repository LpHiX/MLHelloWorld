from network import Network
from param_reader import ParamReader
from mnist_reader import MNISTReader

# Initializing classes:
param_reader = ParamReader()
param_reader.load()
# param_reader.print_yaml()
param_reader.close_file()
mnreader = MNISTReader()


# Debugging an Index
def debug_image(index):
    image, label = mnreader.return_set(index)
    output = param_reader.output(image, True)
    print(param_reader.cost_layer(label, output))
    # Showing plot, must be the end
    mnreader.plot_data(index)


# Debugging index using Network
def debug_image_2(index):
    image, label = mnreader.return_set(index)
    network = Network(param_reader.layers)
    network.train(image, label)
    network.print_outputs()
    mnreader.plot_data(index)

# debug_image(5)
debug_image_2(5)