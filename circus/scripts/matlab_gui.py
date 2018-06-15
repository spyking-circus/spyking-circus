#!/usr/bin/env python
import os
import sys
import subprocess
import argparse
import pkg_resources
import circus
import tempfile
import numpy, logging
from circus.shared.messages import print_and_log, get_colored_header, init_logging
from circus.shared.files import write_datasets
from circus.shared.parser import CircusParser
from circus.shared.utils import apply_patch_for_similarities

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

supported_by_matlab = ['raw_binary', 'mcs_raw_binary']

def which(program):
    path_ext = [""];
    ext_list = None

    if sys.platform == "win32":
        ext_list = [ext.lower() for ext in os.environ["PATHEXT"].split(";")]

    def is_exe(fpath):
        exe = os.path.isfile(fpath) and os.access(fpath, os.X_OK)
        # search for executable under windows
        if not exe:
            if ext_list:
                for ext in ext_list:
                    exe_path = "%s%s" % (fpath,ext)
                    if os.path.isfile(exe_path) and os.access(exe_path, os.X_OK):
                        path_ext[0] = ext
                        return True
                return False
        return exe

    fpath, fname = os.path.split(program)

    if fpath:
        if is_exe(program):
            return "%s%s" % (program, path_ext[0])
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return "%s%s" % (exe_file, path_ext[0])
    return None

def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    header = get_colored_header()
    header += '''Utility to launch the MATLAB GUI and visualize the results.
    '''
    parser = argparse.ArgumentParser(description=header,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('datafile', help='data file')
    parser.add_argument('-e', '--extension', help='extension to consider for visualization',
                        default='')

    if len(argv) == 0:
        parser.print_help()
        sys.exit()

    args = parser.parse_args(argv)

    filename       = os.path.abspath(args.datafile)
    extension      = args.extension
    params         = CircusParser(filename)
    if os.path.exists(params.logfile):
        os.remove(params.logfile)
    logger         = init_logging(params.logfile)
    logger         = logging.getLogger(__name__)
    data_file      = params.get_data_file()
    data_dtype     = data_file.data_dtype
    gain           = data_file.gain
    t_start        = data_file.t_start
    file_format    = data_file.description

    if file_format not in supported_by_matlab:
        print_and_log(["File format %s is not supported by MATLAB. Waveforms disabled" %file_format], 'info', logger)

    if numpy.iterable(gain):
        print_and_log(['Multiple gains are not supported, using a default value of 1'], 'info', logger)
        gain = 1

    file_out_suff  = params.get('data', 'file_out_suff')
    if hasattr(data_file, 'data_offset'):
        data_offset = data_file.data_offset
    else:
        data_offset = 0
    probe          = params.probe
    if extension != '':
        extension = '-' + extension

    def generate_matlab_mapping(probe):
        p         = {}
        positions = []
        nodes     = []
        for key in probe['channel_groups'].keys():
            p.update(probe['channel_groups'][key]['geometry'])
            nodes     +=  probe['channel_groups'][key]['channels']
            positions += [p[channel] for channel in probe['channel_groups'][key]['channels']]
        idx       = numpy.argsort(nodes)
        positions = numpy.array(positions)[idx]
            
        t     = tempfile.NamedTemporaryFile().name + '.hdf5'
        cfile = h5py.File(t, 'w')
        to_write = {'positions' : positions/10., 'permutation' : numpy.sort(nodes), 'nb_total' : numpy.array([probe['total_nb_channels']])}
        write_datasets(cfile, to_write.keys(), to_write) 
        cfile.close()
        return t

    mapping    = generate_matlab_mapping(probe)

    if not params.getboolean('data', 'overwrite'):
        filename = params.get('data', 'data_file_no_overwrite')
    else:
        filename = params.get('data', 'data_file')

    apply_patch_for_similarities(params, extension)

    gui_file = pkg_resources.resource_filename('circus', os.path.join('matlab_GUI', 'SortingGUI.m'))
    # Change to the directory of the matlab file
    os.chdir(os.path.abspath(os.path.dirname(gui_file)))

    # Use quotation marks for string arguments
    if file_format not in supported_by_matlab:
        gui_params = [params.rate, os.path.abspath(file_out_suff), '%s.mat' %extension, mapping, 2, t_start]
        is_string = [False, True, True, True, False]
    
    else:

        gui_params = [params.rate, os.path.abspath(file_out_suff), '%s.mat' %extension, mapping, 2, t_start, data_dtype, data_offset, gain, filename]
        is_string = [False, True, True, True, False, False, True, False, False, True]
    
    arguments = ', '.join(["'%s'" % arg if s else "%s" % arg
                               for arg, s in zip(gui_params, is_string)])
    matlab_command = 'SortingGUI(%s)' % arguments

    print_and_log(["Launching the MATLAB GUI..."], 'info', logger)
    print_and_log([matlab_command], 'debug', logger)

    if params.getboolean('fitting', 'collect_all'):
        print_and_log(['You can not view the unfitted spikes with the MATLAB GUI',
                       'Please consider using phy if you really would like to see them'], 'info', logger)

    try:
        sys.exit(subprocess.call(['matlab',
                              '-nodesktop',
                              '-nosplash',
                              '-r', matlab_command]))
    except Exception:
        if which('matlab') is not None:
            print_and_log(["Something wrong with MATLAB. Try circus-gui-python instead?"], 'error', logger)
        else:
            print_and_log(["MATLAB can not be found in the path. Please add it to the env variables"], 'error', logger)  
        sys.exit(1)

if __name__ == '__main__':
    main()