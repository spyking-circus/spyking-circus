#!/usr/bin/env python
import os
import sys
import argparse
import shutil
import subprocess
import psutil
import h5py
import copy
import pkg_resources
import circus
import logging
import numpy
from os.path import join as pjoin
import colorama
colorama.init(autoreset=True)
from colorama import Fore, Back, Style
from circus.shared.files import data_stats 
from circus.shared.messages import print_error, print_info, print_and_log, get_colored_header, init_logging
from circus.shared.mpi import SHARED_MEMORY, comm, gather_mpi_arguments
from circus.shared.parser import CircusParser
from circus.shared.utils import query_yes_no
from circus.shared.probes import get_averaged_n_edges
from circus.files import __supported_data_files__, list_all_file_format


def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    os.environ['OPENBLAS_NUM_THREADS'] = str(1)

    parallel_hdf5 = h5py.get_config().mpi
    user_path     = pjoin(os.path.expanduser('~'), 'spyking-circus')
    tasks_list    = None

    if not os.path.exists(user_path):
        os.makedirs(user_path)

    try:
        import cudamat as cmt
        cmt.init()
        HAVE_CUDA = True
    except Exception:
        HAVE_CUDA = False


    all_steps = ['whitening', 'clustering', 'fitting', 'gathering', 'extracting', 'filtering', 'converting', 'benchmarking', 'merging', 'validating']

    if os.path.exists(user_path + 'config.params'):
        config_file = os.path.abspath(user_path + 'config.params')
    else:
        config_file = os.path.abspath(pkg_resources.resource_filename('circus', 'config.params'))

    header  = get_colored_header()
    header += Fore.GREEN + 'Local CPUs    : ' + Fore.CYAN + str(psutil.cpu_count()) + '\n'
    header += Fore.GREEN + 'GPU detected  : ' + Fore.CYAN + str(HAVE_CUDA) + '\n'
    header += Fore.GREEN + 'Parallel HDF5 : ' + Fore.CYAN + str(parallel_hdf5) + '\n'
    header += Fore.GREEN + 'Shared memory : ' + Fore.CYAN + str(SHARED_MEMORY) + '\n'
    header += '\n'
    header += Fore.GREEN + "##################################################################"
    header += Fore.RESET

    method_help = '''by default, first 4 steps are performed, 
but a subset x,y can be done. Steps are:
 - filtering
 - whitening
 - clustering
 - fitting
 - (extra) merging [GUI for meta merging]
 - (extra) converting [export results to phy format]
 - (extra) gathering [force collection of results]
 - (extra) extracting [get templates from spike times]
 - (extra) benchmarking [with -o and -t]
 - (extra) validating [to compare performance with GT neurons]'''

    parser = argparse.ArgumentParser(description=header,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('datafile', help='data file (or a list of commands if batch mode)')
    parser.add_argument('-i', '--info', help='list the file formats supported by SpyKING CIRCUS', action='store_true')
    parser.add_argument('-m', '--method',
                        default='filtering,whitening,clustering,fitting',
                        help=method_help)
    parser.add_argument('-c', '--cpu', type=int, default=1, help='number of CPU')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='number of GPU')
    parser.add_argument('-H', '--hostfile', help='hostfile for MPI',
                        default=pjoin(user_path, 'circus.hosts'))
    parser.add_argument('-b', '--batch', help='datafile is a list of commands to launch, in a batch mode',
                        action='store_true')
    parser.add_argument('-p', '--preview', help='GUI to display the first second filtered with thresholds',
                        action='store_true')
    parser.add_argument('-r', '--result', help='GUI to display the results on top of raw data',
                        action='store_true')
    parser.add_argument('-e', '--extension', help='extension to consider for merging and converting',
                        default='None')
    parser.add_argument('-o', '--output', help='output file [for generation of synthetic benchmarks]')
    parser.add_argument('-t', '--type', help='benchmark type',
                        choices=['fitting', 'clustering', 'synchrony'])
    
    if len(argv) == 0:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args(argv)

    steps = args.method.split(',')
    for step in steps:
        if step not in all_steps:
            print_error(['The method "%s" is not recognized' % step])
            sys.exit(0)

    # To save some typing later
    (nb_cpu, nb_gpu, hostfile, batch,
     preview, result, extension, output, benchmark, info) = (args.cpu, args.gpu, args.hostfile, args.batch,
                                                       args.preview, args.result, args.extension, args.output, args.type, args.info)
    filename = os.path.abspath(args.datafile)

    f_next, extens = os.path.splitext(filename)

    if info:    
        if args.datafile.lower() in __supported_data_files__:
            filename = 'tmp'
            if len(__supported_data_files__[args.datafile.lower()].extension) > 0:
                filename += __supported_data_files__[args.datafile.lower()].extension[0]

            __supported_data_files__[args.datafile.lower()](filename, {}, is_empty=True)._display_requirements_()
        else:
            print_and_log(['', 'To get info on any particular file format, do:', '>> spyking-circus file_format -i', ''], 'default')
            print_and_log(list_all_file_format())
        sys.exit(0)

    if extens == '.params':
        print_error(['You should launch the code on the data file!'])
        sys.exit(0)

    file_params = f_next + '.params'
    if not os.path.exists(file_params) and not batch:
        print Fore.RED + 'The parameter file %s is not present!' %file_params
        create_params = query_yes_no(Fore.WHITE + "Do you want SpyKING CIRCUS to create a parameter file?")
        
        if create_params:
            print Fore.WHITE + "Creating", file_params
            print Fore.WHITE + "Fill it properly before launching the code! (see documentation)"
            print_info(['Keep in mind that filtering is performed on site, so please',
                        'be sure to keep a copy of your data elsewhere'])
            shutil.copyfile(config_file, file_params)
        sys.exit(0)
    elif batch:
        tasks_list = filename

    if not batch:
        logfile      = f_next + '.log'
        if os.path.exists(logfile):
            os.remove(logfile)
        logger       = init_logging(logfile)
        params       = CircusParser(filename)
        data_file    = params.get_data_file(source=True, has_been_created=False)
        overwrite    = params.getboolean('data', 'overwrite')
        file_format  = params.get('data', 'file_format')
        if overwrite:
            support_parallel_write = data_file.parallel_write
            is_writable            = data_file.is_writable
        else:
            support_parallel_write = __supported_data_files__['raw_binary'].parallel_write
            is_writable            = __supported_data_files__['raw_binary'].is_writable

    if preview:
        print_and_log(['Preview mode, showing only first second of the recording'], 'info', logger)

        tmp_path_loc = os.path.join(os.path.abspath(params.get('data', 'data_file_noext')), 'tmp')
        if not os.path.exists(tmp_path_loc):
            os.makedirs(tmp_path_loc)
        filename     = os.path.join(tmp_path_loc, os.path.basename(filename))
        f_next, extens = os.path.splitext(filename)
        shutil.copyfile(file_params, f_next + '.params')
        steps        = ['filtering', 'whitening']

        chunk_size = int(2*params.rate)

        data_file.open()
        nb_chunks, _           = data_file.analyze(chunk_size)
        local_chunk, t_offset  = data_file.get_data(0, chunk_size)
        description            = data_file.get_description()
        data_file.close()

        new_params = CircusParser(filename)

        new_params.write('data', 'chunk_size', '2')
        new_params.write('data', 'file_format', 'raw_binary')
        new_params.write('data', 'data_dtype', 'float32')
        new_params.write('data', 'data_offset', '0')
        new_params.write('data', 'dtype_offset', '0')
        new_params.write('data', 'stream_mode', 'None')
        new_params.write('data', 'overwrite', 'True')
        new_params.write('data', 'sampling_rate', str(params.rate))
        new_params.write('whitening', 'safety_time', '0')
        new_params.write('clustering', 'safety_time', '0')
        new_params.write('whitening', 'chunk_size', '2')

        description['data_dtype']   = 'float32'
        description['dtype_offset'] = 0
        description['data_offset']  = 0
        description['gain']         = 1.
        new_params    = CircusParser(filename)
        data_file_out = new_params.get_data_file(is_empty=True, params=description)
        
        support_parallel_write = data_file_out.parallel_write
        is_writable            = data_file_out.is_writable

        data_file_out.allocate(shape=local_chunk.shape, data_dtype=numpy.float32)
        data_file_out.open('r+')
        data_file_out.set_data(0, local_chunk)
        data_file_out.close()

    if tasks_list is not None:
        with open(tasks_list, 'r') as f:
            for line in f:
                if len(line) > 0:
                    subprocess.check_call(['spyking-circus'] + line.replace('\n', '').split(" "))
    else:

        print_and_log(['Config file: %s' %(f_next + '.params')], 'debug', logger)
        print_and_log(['Data file  : %s' %filename], 'debug', logger)

        print get_colored_header()
        if preview:
            print Fore.GREEN + "Steps         :", Fore.CYAN + "preview mode"
        elif result:
            print Fore.GREEN + "Steps         :", Fore.CYAN + "result mode"
        else:
            print Fore.GREEN + "Steps         :", Fore.CYAN + ", ".join(steps)
        print Fore.GREEN + "GPU detected  :", Fore.CYAN + str(HAVE_CUDA)
        print Fore.GREEN + "Number of CPU :", Fore.CYAN + str(nb_cpu) + "/" + str(psutil.cpu_count())
        if HAVE_CUDA:
            print Fore.GREEN + "Number of GPU :", Fore.CYAN + str(nb_gpu)
        print Fore.GREEN + "Parallel HDF5 :", Fore.CYAN + str(parallel_hdf5)

        do_upgrade = ''
        if not SHARED_MEMORY:
            do_upgrade = Fore.WHITE + '   [please consider upgrading MPI]'

        print Fore.GREEN + "Shared memory :", Fore.CYAN + str(SHARED_MEMORY) + do_upgrade
        print Fore.GREEN + "Hostfile      :", Fore.CYAN + hostfile
        print ""
        print Fore.GREEN + "##################################################################"
        print ""        
        print Fore.RESET

        # Launch the subtasks
        subtasks = [('filtering', 'mpirun'),
                    ('whitening', 'mpirun'),
                    ('clustering', 'mpirun'),
                    ('fitting', 'mpirun'),
                    ('extracting', 'mpirun'),
                    ('gathering', 'python'),
                    ('converting', 'mpirun'),
                    ('benchmarking', 'mpirun'),
                    ('merging', 'mpirun'),
                    ('validating', 'mpirun')]

        if HAVE_CUDA and nb_gpu > 0:
            use_gpu = 'True'
        else:
            use_gpu = 'False'

        if preview:
            tmp_params = copy.deepcopy(new_params.data_file._params)

        time = data_stats(params)/60.

        if preview:
            params = new_params
            params.data_file._params = tmp_params

        if nb_cpu < psutil.cpu_count():
            if use_gpu != 'True' and not result:
                print_and_log(['Using only %d out of %d local CPUs available (-c to change)' %(nb_cpu, psutil.cpu_count())], 'info', logger)

        if params.getboolean('detection', 'matched-filter') and not params.getboolean('clustering', 'smart_search'):
            print_and_log(['Smart Search should be activated for matched filtering' ], 'info', logger)

        if time > 30 and not params.getboolean('clustering', 'smart_search'):
            print_and_log(['Smart Search could be activated for long recordings' ], 'info', logger)

        n_edges = get_averaged_n_edges(params)
        if n_edges > 100 and not params.getboolean('clustering', 'compress'):
            print_and_log(['Template compression is highly recommended based on parameters'], 'info', logger)    

        if params.getint('data', 'N_e') > 500:
            if params.getint('data', 'chunk_size') > 10:
                params.write('data', 'chunk_size', '10')
            if params.getint('whitening', 'chunk_size') > 10:
                params.write('whitening', 'chunk_size', '10')
            print_and_log(["Large number of electrodes, reducing chunk sizes to 10s"], 'info', logger)

        if not result:
            for subtask, command in subtasks:
                if subtask in steps:
                    if command == 'python':
                        # Directly call the launcher
                        try:
                            circus.launch(subtask, filename, nb_cpu, nb_gpu, use_gpu)
                        except:
                            print_and_log(['Step "%s" failed!' % subtask], 'error', logger)
                            sys.exit(0)
                    elif command == 'mpirun':
                        # Use mpirun to make the call
                        mpi_args = gather_mpi_arguments(hostfile, params)

                        if subtask in ['filtering', 'benchmarking'] and not is_writable:
                            if not preview and overwrite:
                                print_and_log(['The file format %s is read only!' %file_format,
                                               'You should set overwite to False, to create a copy of the data.',
                                               'However, note that if you have streams, informations on times',
                                               'will be discarded'], 'info', logger)
                                sys.exit(0)
                        
                        if subtask in ['filtering'] and not support_parallel_write and (args.cpu > 1):
                            print_and_log(['No parallel writes for %s: only 1 node used for %s' %(file_format, subtask)], 'info', logger)
                            nb_tasks = str(1)

                        else:
                            if subtask != 'fitting':
                                nb_tasks = str(max(args.cpu, args.gpu))
                            else:
                                if use_gpu == 'True':
                                    nb_tasks = str(args.gpu)
                                else:
                                    nb_tasks = str(args.cpu)

                        if subtask == 'benchmarking':
                            if (output is None) or (benchmark is None):
                                print_and_log(["To generate synthetic datasets, you must provide output and type"], 'error', logger)
                                sys.exit(0)
                            mpi_args += ['-np', nb_tasks,
                                     'spyking-circus-subtask',
                                     subtask, filename, str(nb_cpu), str(nb_gpu), use_gpu, output, benchmark]
                        elif subtask in ['merging', 'converting']:
                            mpi_args += ['-np', nb_tasks,
                                     'spyking-circus-subtask',
                                     subtask, filename, str(nb_cpu), str(nb_gpu), use_gpu, extension]
                        else:
                            mpi_args += ['-np', nb_tasks,
                                     'spyking-circus-subtask',
                                     subtask, filename, str(nb_cpu), str(nb_gpu), use_gpu]

                        print_and_log(['Launching task %s' %subtask], 'debug', logger)
                        print_and_log(['Command: %s' %str(mpi_args)], 'debug', logger)

                        try:
                            subprocess.check_call(mpi_args)
                        except:
                            print_and_log(['Step "%s" failed!' % subtask], 'error', logger)
                            raise

    if preview or result:
        from circus.shared import gui
        import pylab
        from matplotlib.backends import qt_compat
        
        use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE
        if use_pyside:
            from PySide import QtGui, QtCore, uic
        else:
            from PyQt4 import QtGui, QtCore, uic
        app = QtGui.QApplication([])
        try:
            pylab.style.use('ggplot')
        except Exception:
            pass

        if preview:
            print_and_log(['Launching the preview GUI...'], 'debug', logger)
            mygui = gui.PreviewGUI(new_params)
            shutil.rmtree(tmp_path_loc)
        elif result:
            data_file = params.get_data_file()
            print_and_log(['Launching the result GUI...'], 'debug', logger)
            mygui = gui.PreviewGUI(params, show_fit=True)
        sys.exit(app.exec_())
