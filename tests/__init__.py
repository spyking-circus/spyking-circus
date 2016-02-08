import os, sys, nose, h5py
import circus
import subprocess
from termcolor import colored
import urllib2
import unittest
import shutil
import pkg_resources
from circus.shared.utils import *
    

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
        args += ['-x', 'LD_LIBRARY_PATH']
        args += ['-x', 'PATH']
        args += ['-x', 'PYTHONPATH']
    elif vendor[0] == 'Microsoft MPI':
        args  = ['mpiexec']
    else: 
        args  = ['mpirun']
    
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

'''
def get_dataset(self):
    dirname  = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    filename = os.path.join(dirname, 'data') 
    if not os.path.exists(filename):
        os.makedirs(filename)
    filename = os.path.join(filename, 'data.dat')
    if not os.path.exists(filename): 
        print "Downloading a test dataset..."
        datafile = urllib2.urlopen("http://www.yger.net/wp-content/uploads/silico_0.dat")
        output   = open(filename,'wb')
        output.write(datafile.read())
        output.close()
    config_file = os.path.abspath(pkg_resources.resource_filename('circus', 'config.params'))
    file_params = os.path.abspath(filename.replace('.dat', '.params'))
    if not os.path.exists(file_params):
        shutil.copyfile(config_file, file_params)
        io.change_flag(filename, 'data_offset', '0')
        io.change_flag(filename, 'data_dtype', 'int16')
        io.change_flag(filename, 'temporal', 'True')
        user_path  = os.path.join(os.path.expanduser('~'), 'spyking-circus')
        probe_file = os.path.join(os.path.join(user_path, 'probes'), 'dan.prb')
        io.change_flag(filename, 'mapping', probe_file)
        io.change_flag(filename, 'make_plots', 'False')
        io.change_flag(filename, 'nb_repeats', '1')
        io.change_flag(filename, 'smart_search', '3')
        io.change_flag(filename, 'max_elts', '1000', 'Fraction')

    a, b     = os.path.splitext(os.path.basename(filename))
    c, d     = os.path.splitext(filename)
    file_out = os.path.join(os.path.abspath(c), a)

    mpi_launch('filtering', filename, 2, 0, 'False')
    if not os.path.exists(file_out + '.basis.hdf5'):
        mpi_launch('whitening', filename, 2, 0, 'False')
    if not os.path.exists(file_out + '.templates.hdf5'):
        mpi_launch('clustering', filename, 2, 0, 'False')
    if not os.path.exists(file_out + '.result.hdf5'):
        mpi_launch('fitting', filename, 2, 0, 'False')    
    return filename
'''

def get_dataset(self):
    dirname  = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    filename = os.path.join(dirname, 'data') 
    if not os.path.exists(filename):
        os.makedirs(filename)
    result   = os.path.join(filename, 'data')
    filename = os.path.join(filename, 'data.dat')
    if not os.path.exists(filename):
        print "Generating a synthetic dataset of 30 channels, 5min at 20kHz..."
        sampling_rate = 20000
        N_total       = 30
        gain          = 0.1
        data          = (gain * numpy.random.randn(sampling_rate * N_total * 5 * 60)).astype(numpy.float32)
        myfile        = open(filename, 'w')
        myfile.write(data.tostring())
        myfile.close()
    
    if not os.path.exists(result):
        os.makedirs(result)
        shutil.copy(os.path.join(dirname, 'test.basis.hdf5'), os.path.join(result, 'data.basis.hdf5'))
        shutil.copy(os.path.join(dirname, 'test.templates.hdf5'), os.path.join(result, 'data.templates.hdf5'))
        shutil.copy(os.path.join(dirname, 'test.clusters.hdf5'), os.path.join(result, 'data.clusters.hdf5'))

    config_file = os.path.abspath(pkg_resources.resource_filename('circus', 'config.params'))
    file_params = os.path.abspath(filename.replace('.dat', '.params'))
    if not os.path.exists(file_params):
        shutil.copyfile(config_file, file_params)
        io.change_flag(filename, 'data_offset', '0')
        io.change_flag(filename, 'data_dtype', 'float32')
        io.change_flag(filename, 'temporal', 'False')
        user_path  = os.path.join(os.path.expanduser('~'), 'spyking-circus')
        probe_file = os.path.join(os.path.join(user_path, 'probes'), 'dan.prb')
        io.change_flag(filename, 'mapping', probe_file)
        io.change_flag(filename, 'make_plots', 'False')
        io.change_flag(filename, 'nb_repeats', '3')
        io.change_flag(filename, 'N_t', '3')
        io.change_flag(filename, 'smart_search', '3')
        io.change_flag(filename, 'max_elts', '10000', 'Fraction')
        io.change_flag(filename, 'filter_done', 'True')
        io.change_flag(filename, 'extraction', 'median-raw')

    a, b     = os.path.splitext(os.path.basename(filename))
    c, d     = os.path.splitext(filename)
    file_out = os.path.join(os.path.abspath(c), a)

    return filename

if __name__=='__main__':
    run()