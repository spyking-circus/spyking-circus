from utils import *
from scipy import signal

params         = io.load_parameters(sys.argv[-1])

if comm.rank == 0:
    io.data_stats(params)