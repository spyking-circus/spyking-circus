from utils import *

params         = io.load_parameters(sys.argv[-1])
nb_threads     = int(sys.argv[-2])

io.collect_data(nb_threads, params, erase=False)