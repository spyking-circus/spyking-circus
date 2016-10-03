import importlib
import logging

__version__ = '0.5'

def launch(task, filename, nb_cpu, nb_gpu, use_gpu, output=None, benchmark=None, extension=''):

    from circus.shared.parser import CircusParser
    params = CircusParser(filename)
    params.get_data_file()
    module = importlib.import_module('circus.' + task)
    
    if task == 'benchmarking':
        module.main(params, nb_cpu, nb_gpu, use_gpu, output, benchmark)
    elif task in ['converting', 'merging']:
        module.main(params, nb_cpu, nb_gpu, use_gpu, extension)
    else:
        module.main(params, nb_cpu, nb_gpu, use_gpu)