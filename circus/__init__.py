import importlib
from circus.shared.utils import io

def launch(task, filename, nb_cpu, nb_gpu, use_gpu, output=None, benchmark=None):
    params = io.load_parameters(filename)
    module = importlib.import_module('circus.'+task)
    if task != 'benchmarking':
        module.main(filename, params, nb_cpu, nb_gpu, use_gpu)
    else:
        module.main(filename, params, nb_cpu, nb_gpu, use_gpu, output, benchmark)