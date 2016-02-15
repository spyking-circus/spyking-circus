#!/usr/bin/env python
import os
import sys
import socket
import getopt
import shutil
import subprocess
import pkg_resources
import platform
from os.path import join as pjoin
from termcolor import colored
import colorama
colorama.init()
import circus.shared.files as io


import circus
from circus.shared.files import print_error, print_info, write_to_logger

def main():

    argv = sys.argv

    import h5py
    parallel_hdf5 = h5py.get_config().mpi

    user_path  = pjoin(os.path.expanduser('~'), 'spyking-circus')
    tasks_list = None

    nb_gpu     = 0
    nb_cpu     = 1
    output     = None
    benchmark  = None
    preview    = False
    result     = False

    if not os.path.exists(user_path):
        os.makedirs(user_path)

    try:
        import cudamat as cmt
        cmt.init()
        HAVE_CUDA = True
        nb_gpu    = 1
    except Exception:
        HAVE_CUDA = False

    all_steps = ['whitening', 'clustering', 'fitting', 'gathering', 'extracting', 'filtering', 'converting', 'benchmarking', 'merging']
    steps     = ['filtering', 'whitening', 'clustering', 'fitting']
    hostfile  = pjoin(user_path, 'circus.hosts')

    if os.path.exists(user_path + 'config.params'):
        config_file = os.path.abspath(user_path + 'config.params')
    else:
        config_file = os.path.abspath(pkg_resources.resource_filename('circus', 'config.params'))

    header = '''
##################################################################
#####              Welcome to the SpyKING CIRCUS             #####
#####                                                        #####
#####              Written by P.Yger and O.Marre             #####
##################################################################
'''

    message = '''   
Syntax is spyking-circus datafile [options]

Options are:
    -h or --help     : display the help
    -m or --method   : by default, all 4 steps of the algorithm are
                       performed, but a subset x,y can be done, 
                       using the syntax -m x,y. Steps are:
                         - filtering
                         - whitening
                         - clustering
                         - fitting
                         - (extra) merging [meta merging]
                         - (extra) gathering [to collect results]
                         - (extra) extracting [templates from spike times]
                         - (extra) converting [to export to phy format]
                         - (extra) benchmarking [with -o and -t]
    -c or --cpu      : number of CPU (default 1)
    -g or --gpu      : number of GPU (default 1 if CUDA available)
    -H or --hostfile : hostfile for MPI (default is ~/spyking-circus/circus.hosts)
    -b or --batch    : datafile is a list of commands to launch, in a batch mode
    -o or --output   : output file [for generation of synthetic benchmarks]
    -p or --preview  : GUI to display the first second filtered with thresholds
    -r or --result   : GUI to display the results on top of raw data 
    -t or --type     : benchmark type [fitting, clustering, synchrony]'''

    noparams='''The parameter file %s is not present!'''
    batch_mode = (('-b' in argv) or ('--batch' in argv))

    if not batch_mode:
        print colored(header, 'green')

    if len(argv) < 2:
        print colored("GPU detected  :", 'green'), colored(HAVE_CUDA, 'cyan')
        print colored("Parallel HDF5 :", 'green'), colored(parallel_hdf5, 'cyan')
        print ""
        print colored("###################################################################", 'green')
        print ""
        print message
        sys.exit()
    else:
        filename   = argv[1]
        if filename in ['-h', '--help']:
            print colored("GPU detected  :", 'green'), colored(HAVE_CUDA, 'cyan')
            print colored("Parallel HDF5 :", 'green'), colored(parallel_hdf5, 'cyan')
            print ""
            print colored("##################################################################", 'green')
            print ""
            print message
            sys.exit()
        filename   = os.path.abspath(filename)
        if not os.path.exists(filename):
            print_error(["The data file %s can not be found!" %filename])
            sys.exit()
        else:
            f_next, extension = os.path.splitext(filename)
            file_params       = f_next + '.params'
            if not os.path.exists(file_params) and not batch_mode:
                print noparams %file_params
                key = ''
                while key not in ['y', 'n']:
                    key = raw_input("Do you want SpyKING CIRCUS to create a parameter file? [y/n]")
                if key == 'y':
                    print "Generating template file", file_params
                    print "Fill it properly before launching the code! (see documentation)"
                    shutil.copyfile(config_file, file_params)
                sys.exit()
            elif batch_mode:
                tasks_list = filename

            opts, args  = getopt.getopt(argv[2:], "hvbprm:H:c:g:o:t:", ["help", "method=", "hostfile=", "cpu=", "gpu=", "output=", "type="])

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print message
            sys.exit()
        elif opt == '-d':
            verbose = True
        elif opt in ('-m', '--method'):
            args = arg.split(',')
            for arg in args:
                if arg not in all_steps:
                    print_error(['The method "%s" is not recognized' % arg])
                    sys.exit(0)
            else:
                steps = args
        elif opt in ('-g', '--gpu'):
            nb_gpu = int(arg)
        elif opt in ('-c', '--cpu'):
            nb_cpu = int(arg)
        elif opt in ('-H', '--hostfile'):
            hostfile = arg
        elif opt in ('-o', '--output'):
            output = arg
        elif opt in ('-t', '--type'):
            benchmark = arg
        elif opt in ('-p', '--preview'):
            preview = True
        elif opt in ('-r', '--result'):
            result = True

    # Print info
    if not batch_mode:
    	params = io.load_parameters(filename)

    if os.path.exists(f_next + '.log'):
        os.remove(f_next + '.log')

    write_to_logger(params, ['Config file: %s' %(f_next + '.params')], 'debug')
    write_to_logger(params, ['Data file  : %s' %filename], 'debug')

    if preview:
        print_info(['Preview mode, showing only first second of the recording'])
        tmp_path_loc = os.path.join(os.path.abspath(params.get('data', 'data_file_noext')), 'tmp')
        if not os.path.exists(tmp_path_loc):
            os.makedirs(tmp_path_loc)
        filename     = os.path.join(tmp_path_loc, 'preview.dat')
        shutil.copyfile(file_params, filename.replace('.dat', '.params'))
        steps        = ['filtering', 'whitening']
        io.prepare_preview(params, filename)
    else: 
	if not batch_mode: 	  
            stationary = params.getboolean('data', 'stationary')
        else:
             stationary = False

    if tasks_list is not None:
        with open(tasks_list, 'r') as f:
            for line in f:
                if len(line) > 0:
                    subprocess.check_call(['spyking-circus'] + line.replace('\n', '').split(" "))
    else:
        print colored("Steps         :", 'green'), colored(", ".join(steps), 'cyan')
        print colored("GPU detected  :", 'green'), colored(HAVE_CUDA, 'cyan')
        print colored("Number of CPU :", 'green'), colored(nb_cpu, 'cyan')
        if HAVE_CUDA:
            print colored("Number of GPU :", 'green'), colored(nb_gpu, 'cyan')
        print colored("Parallel HDF5 :", 'green'), colored(parallel_hdf5, 'cyan')
        print colored("Hostfile      :", 'green'), colored(hostfile, 'cyan')
        print ""
        print colored("##################################################################", 'green')
        print ""        

        if not preview:
            length = io.data_stats(params)

            if length >= 3600 and stationary:
                print_info(['Long recording detected: maybe turn off the stationary mode'])

        # Launch the subtasks
        subtasks = [('filtering', 'mpirun'),
                    ('whitening', 'mpirun'),
                    ('clustering', 'mpirun'),
                    ('fitting', 'mpirun'),
                    ('extracting', 'mpirun'),
                    ('gathering', 'python'),
                    ('converting', 'python'),
                    ('benchmarking', 'mpirun'),
                    ('merging', 'mpirun')]

        if HAVE_CUDA and nb_gpu > 0:
            use_gpu = 'True'
        else:
            use_gpu = 'False'

        if not result:
            for subtask, command in subtasks:
                if subtask in steps:
                    if command == 'python':
                        # Directly call the launcher
                        try:
                            circus.launch(subtask, filename, nb_cpu, nb_gpu, use_gpu)
                        except:
                            print_error(['Step "%s" failed!' % subtask])
                            raise
                    elif command == 'mpirun':
                        # Use mpirun to make the call

                        from mpi4py import MPI
                        vendor = MPI.get_vendor()
                        write_to_logger(params, ['MPI detected: %s' %str(vendor)], 'debug')
                        if vendor[0] == 'Open MPI':
                            args  = ['mpirun']
                            if os.getenv('LD_LIBRARY_PATH'):
                                args += ['-x', 'LD_LIBRARY_PATH']
                            if os.getenv('PATH'):
                                args += ['-x', 'PATH']
                            if os.getenv('PYTHONPATH'):
                                args += ['-x', 'PYTHONPATH']
                            if os.path.exists(hostfile):
                                args += ['-hostfile', hostfile]
                        elif vendor[0] == 'Microsoft MPI':
                            args  = ['mpiexec']
                            if os.path.exists(hostfile):
                                args += ['-gmachinefile', hostfile]
                        elif vendor[0] == 'MPICH2':
                            args  = ['mpiexec']
                            if os.path.exists(hostfile):
                                args += ['-f', hostfile]
                        else: 
                            print_error(['%s may not be yet properly implemented: contact developpers' %vendor[0]])
                            args  = ['mpirun']
                            if os.path.exists(hostfile):
                                args += ['-hostfile', hostfile]

                        if subtask != 'fitting':
                            nb_tasks = str(max(nb_cpu, nb_gpu))
                        else:
                            if use_gpu == 'True':
                                nb_tasks = str(nb_gpu)
                            else:
                                nb_tasks = str(nb_cpu)

                        if subtask != 'benchmarking':
                            if platform.system() == 'Windows':
                                args += ['-np', nb_tasks, sys.executable,
                                       'spyking-circus-subtask',
                                       subtask, filename, str(nb_cpu), str(nb_gpu), use_gpu]
                            else:                
                                args += ['-np', nb_tasks,
                                       'spyking-circus-subtask',
                                       subtask, filename, str(nb_cpu), str(nb_gpu), use_gpu]
                        else:
                            if (output is None) or (benchmark is None):
                                print_error(["To generate synthetic datasets, you must provide output and type"])
                                sys.exit()
                            if platform.system() == 'Windows':
                                args += ['-np', nb_tasks, sys.executable,
                                       'spyking-circus-subtask',
                                       subtask, filename, str(nb_cpu), str(nb_gpu), use_gpu, output, benchmark]
                            else:
                                args += ['-np', nb_tasks,
                                       'spyking-circus-subtask',
                                       subtask, filename, str(nb_cpu), str(nb_gpu), use_gpu, output, benchmark]

                        write_to_logger(params, ['Launching task %s' %subtask], 'debug')
                        write_to_logger(params, ['Command: %s' %str(args)], 'debug')

                        try:
                            subprocess.check_call(args)
                        except:
                            print_error(['Step "%s" failed!' % subtask])
                            raise

    if preview or result:
        import numpy, pylab
        from circus.shared.gui import PreviewGUI
        try:
            pylab.style.use('ggplot')
        except Exception:
            pass
        pylab.switch_backend('QT4Agg')
        if preview:
            PreviewGUI(io.load_parameters(filename))
            mng   = pylab.get_current_fig_manager()
            pylab.tight_layout()
            pylab.show()
            shutil.rmtree(tmp_path_loc)
        elif result:
            PreviewGUI(io.load_parameters(filename), show_fit=True)
            mng   = pylab.get_current_fig_manager()
            pylab.tight_layout()
            pylab.show()
