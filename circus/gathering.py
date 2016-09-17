from .shared.utils import *

def main(filename, params, nb_cpu, nb_gpu, use_gpu):

    data_file      = io.get_data_file(params)
    io.collect_data(nb_cpu, data_file, erase=False)
    data_file.close()