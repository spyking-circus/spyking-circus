from utils import *

def main(filename, params, nb_cpu, use_gpu):
    io.collect_data(nb_cpu, params, erase=False)
