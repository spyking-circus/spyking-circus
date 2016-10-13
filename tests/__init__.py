import os, sys, nose, h5py
import circus
import subprocess
import urllib2
import unittest
import colorama
colorama.init(autoreset=True)
from colorama import Fore, Back, Style
import shutil
import pkg_resources
from circus.shared.utils import *
from circus.shared.parser import CircusParser
    

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

def mpi_launch(subtask, filename, nb_cpu, nb_gpu, use_gpu, output=None, benchmark=None):
    args     = ['mpirun'] 
        
    from mpi4py import MPI
    vendor = MPI.get_vendor()
    if vendor[0] == 'Open MPI':
        args  = ['mpirun']
        if os.getenv('LD_LIBRARY_PATH'):
            args += ['-x', 'LD_LIBRARY_PATH']
        if os.getenv('PATH'):
            args += ['-x', 'PATH']
        if os.getenv('PYTHONPATH'):
            args += ['-x', 'PYTHONPATH']
    elif vendor[0] == 'Microsoft MPI':
        args  = ['mpiexec']
    elif vendor[0] == 'MPICH2':
        mpi_args = ['mpiexec']
    elif vendor[0] == 'MPICH':
        mpi_args = ['mpiexec']
    
    if use_gpu == 'True':
        nb_tasks = str(nb_gpu)
    else:
        nb_tasks = str(nb_cpu)

    if subtask in ['merging', 'converting']:
        args += ['-np', nb_tasks,
                  'spyking-circus-subtask',
                  subtask, filename, str(nb_cpu), str(nb_gpu), use_gpu, '']
    else:
        if subtask == 'benchmarking':
            if (output is None) or (benchmark is None):
                print "To generate synthetic datasets, you must provide output and type"
                sys.exit()
            args += ['-np', nb_tasks,
                     'spyking-circus-subtask',
                     subtask, filename, str(nb_cpu), str(nb_gpu), use_gpu, output, benchmark]
        else:
            args += ['-np', nb_tasks,
                 'spyking-circus-subtask',
                 subtask, filename, str(nb_cpu), str(nb_gpu), use_gpu]
    

    subprocess.check_call(args)


def get_dataset(self):
    dirname  = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    filename = os.path.join(dirname, 'data') 
    if not os.path.exists(filename):
        os.makedirs(filename)
    result   = os.path.join(filename, 'data')
    filename = os.path.join(filename, 'data.dat')
    if not os.path.exists(filename):
        print "Generating a synthetic dataset of 4 channels, 2min at 20kHz..."
        sampling_rate = 20000
        N_total       = 4
        gain          = 0.5
        data          = (gain * numpy.random.randn(sampling_rate * N_total * 2 * 60)).astype(numpy.float32)
        myfile        = open(filename, 'w')
        myfile.write(data.tostring())
        myfile.close()  
    
    src_path = os.path.abspath(os.path.join(dirname, 'snippet'))

    if not os.path.exists(result):
        os.makedirs(result)
        shutil.copy(os.path.join(src_path, 'test.basis.hdf5'), os.path.join(result, 'data.basis.hdf5'))
        shutil.copy(os.path.join(src_path, 'test.templates.hdf5'), os.path.join(result, 'data.templates.hdf5'))
        shutil.copy(os.path.join(src_path, 'test.clusters.hdf5'), os.path.join(result, 'data.clusters.hdf5'))

    config_file = os.path.abspath(pkg_resources.resource_filename('circus', 'config.params'))
    file_params = os.path.abspath(filename.replace('.dat', '.params'))
    if not os.path.exists(file_params):
        
        shutil.copyfile(config_file, file_params)
        probe_file = os.path.join(src_path, 'test.prb')
        parser = CircusParser(filename, mapping=probe_file)
        parser.write('data', 'file_format', 'raw_binary')
        parser.write('data', 'data_offset', '0')
        parser.write('data', 'data_dtype', 'float32')
        parser.write('data', 'sampling_rate', '20000')
        parser.write('whitening', 'temporal', 'False')
        parser.write('data', 'mapping', probe_file)
        parser.write('clustering', 'make_plots', 'png')
        parser.write('clustering', 'nb_repeats', '3')
        parser.write('detection', 'N_t', '3')
        parser.write('clustering', 'smart_search', 'False')
        parser.write('clustering', 'max_elts', '10000')
        parser.write('noedits', 'filter_done', 'True')
        parser.write('clustering', 'extraction', 'median-raw')

    a, b     = os.path.splitext(os.path.basename(filename))
    c, d     = os.path.splitext(filename)
    file_out = os.path.join(os.path.abspath(c), a)

    return filename

if __name__=='__main__':
    run()