#!/usr/bin/env python
import os
import sys
import subprocess
import pkg_resources
import argparse
import circus
import shutil
import tempfile
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py
import numpy
import logging
from colorama import Fore
from circus.shared.messages import print_and_log, get_colored_header, init_logging
from circus.shared.algorithms import slice_result
from circus.shared.parser import CircusParser
from circus.shared.utils import query_yes_no

def get_dead_times(dead_file, sampling_rate, dead_in_ms=False):
    dead_times = numpy.loadtxt(dead_file)
    
    if len(dead_times.shape) == 1:
        dead_times = dead_times.reshape(1, 2)

    if dead_in_ms:
        dead_times *= numpy.int64(sampling_rate)

    dead_times = dead_times.astype(numpy.int64)
    return dead_times


def main(argv=None):
    
    if argv is None:
        argv = sys.argv[1:]

    header = get_colored_header()
    header += '''Utility to concatenate artefacts/dead times before using 
stream mode
    '''
    parser = argparse.ArgumentParser(description=header,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('datafile', help='data file')
    parser.add_argument('-w', '--window', help='text file with artefact window files',
                        default='')

    if len(argv) == 0:
        parser.print_help()
        sys.exit()

    args = parser.parse_args(argv)
    window_file = os.path.abspath(args.window)
    
    filename       = os.path.abspath(args.datafile)
    params         = CircusParser(filename)
    dead_in_ms     = params.getboolean('triggers', 'dead_in_ms')

    if os.path.exists(params.logfile):
        os.remove(params.logfile)

    logger         = init_logging(params.logfile)
    logger         = logging.getLogger(__name__)

    if params.get('data', 'stream_mode') == 'multi-files':
        data_file = params.get_data_file(source=True, has_been_created=False)
        all_times = numpy.zeros((0, 2), dtype=numpy.int64)

        for f in data_file._sources:
            name, ext = os.path.splitext(f.file_name)
            dead_file = f.file_name.replace(ext, '.dead')
            if os.path.exists(dead_file):
                print_and_log(['Found file %s' %dead_file], 'default', logger)
                times = get_dead_times(dead_file, data_file.sampling_rate, dead_in_ms)
                times += f.t_start
                all_times = numpy.vstack((all_times, times))

        output_file = os.path.join(os.path.dirname(filename), 'dead_zones.txt')
        if len(all_times) > 0:
            print_and_log(['Saving global artefact file in %s' %output_file], 'default', logger)
            if dead_in_ms:
                all_times = all_times.astype(numpy.float32)/data_file.sampling_rate
            numpy.savetxt(output_file, all_times)

    elif params.get('data', 'stream_mode') == 'single-file':
        print_and_log(['Not implemented'], 'error', logger)
