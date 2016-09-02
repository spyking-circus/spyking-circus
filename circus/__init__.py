import importlib
import circus.shared.files as io

__version__ = '0.4.2'

def launch(task, filename, nb_cpu, nb_gpu, use_gpu, output=None, benchmark=None, extension=''):
    params = io.load_parameters(filename)
    module = importlib.import_module('circus.'+task)
        
    if task == 'benchmarking':
        module.main(filename, params, nb_cpu, nb_gpu, use_gpu, output, benchmark)
    elif task in ['converting', 'merging']:
        module.main(filename, params, nb_cpu, nb_gpu, use_gpu, extension)
    else:
        module.main(filename, params, nb_cpu, nb_gpu, use_gpu)