from .shared.utils import *

def main(params, nb_cpu, nb_gpu, use_gpu):

	logger         = init_logging(params.logfile)
    logger         = logging.getLogger('circus.gathering')
    io.collect_data(nb_cpu, params, erase=False)
