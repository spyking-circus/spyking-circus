from .shared.utils import *

def main(params, nb_cpu, nb_gpu, use_gpu):

    io.collect_data(nb_cpu, params, erase=False)
