from .shared.utils import *

def main(params, nb_cpu, nb_gpu, use_gpu):

    data_file      = params.get_data_file()
    io.collect_data(nb_cpu, data_file, erase=False)
    data_file.close()