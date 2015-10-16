try:
    import nose
except ImportError:
    raise ImportError('Running the test suite requires the "nose" package.')

import circus
import subprocess
from termcolor import colored

def mpi_launch(subtask, filename, nb_cpu, nb_gpu, use_gpu, output=None, benchmark=None):
    args     = ['mpirun'] 
    
    from mpi4py import MPI
    vendor = MPI.get_vendor()
    if vendor[0] == 'Open MPI':
        args += ['-x', 'LD_LIBRARY_PATH']
        args += ['-x', 'PATH']
        args += ['-x', 'PYTHONPATH']

        if use_gpu == 'True':
            nb_tasks = str(nb_gpu)
        else:
            nb_tasks = str(nb_cpu)

        if subtask != 'benchmarking':
            args += ['-np', nb_tasks,
                        'spyking-circus-subtask.py',
                        subtask, filename, str(nb_cpu), str(nb_gpu), use_gpu]
        else:
            if (output is None) or (benchmark is None):
                print colored("To generate synthetic datasets, you must provide output and type", 'red')
                sys.exit()
            args += ['-np', nb_tasks,
                     'spyking-circus-subtask.py',
                      subtask, filename, str(nb_cpu), str(nb_gpu), use_gpu, output, benchmark]
    subprocess.check_call(args)

def test_MPI():
    HAVE_MPI = False
    try:
        import mpi4py
        HAVE_MPI = True
    except ImportError:
        pass
    assert HAVE_MPI == True 

def test_CUDA():
    HAVE_CUDA = False
    try:
        import cudamat
        HAVE_CUDA = True
    except ImportError:
        pass
    assert HAVE_CUDA == True

def test_import_dataset():
    pass