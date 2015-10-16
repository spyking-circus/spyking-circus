import os, sys, nose
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

def run():
    dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    # We write to stderr since nose does all of its output on stderr as well
    sys.stderr.write('Running tests in "%s" ' % dirname)
    success = []
    argv = ['nosetests', dirname]
    success.append(nose.run(argv=argv))
    all_success = all(success)
    if not all_success:
        sys.stderr.write(('ERROR: %d/%d test suite(s) did not complete '
                          'successfully (see above).\n') % (len(success) - sum(success),
                                                            len(success)))
    else:
        sys.stderr.write(('OK: %d/%d test suite(s) did complete '
                          'successfully.\n') % (len(success), len(success)))



if __name__=='__main__':
    run()