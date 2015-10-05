import importlib

from .utils import io


def launch(task, filename, nb_cpu, use_gpu):
    params = io.load_parameters(filename)
    module = importlib.import_module('circus.'+task)
    module.main(filename, params, nb_cpu, use_gpu)
