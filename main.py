from param_reader import ParamReader
from reader import Reader

#reader = Reader()
#reader.plot_data(1)

param_reader = ParamReader()
param_reader.print_file()
#param_reader.write_defaults()
param_reader.close_file()