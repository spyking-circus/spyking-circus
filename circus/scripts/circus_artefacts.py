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
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
import numpy
import logging
from colorama import Fore
from circus.shared.messages import print_and_log, get_colored_header, init_logging
from circus.shared.algorithms import slice_result
from circus.shared.parser import CircusParser


def get_dead_times(dead_file, sampling_rate, dead_in_ms=False):
    dead_times = numpy.loadtxt(dead_file)
    
    if len(dead_times.shape) == 1:
        dead_times = dead_times.reshape(1, 2)

    if dead_in_ms:
        dead_times *= numpy.int64(sampling_rate)/1000

    return dead_times.astype(numpy.int64)


def get_trig_times(trig_file, sampling_rate, trig_in_ms=False):
    trig_times = numpy.loadtxt(trig_file)
    
    if len(trig_times.shape) == 1:
        trig_times = trig_times.reshape(1, 2)

    if trig_in_ms:
        trig_times[:, 1] *= numpy.int64(sampling_rate)/1000

    return trig_times.astype(numpy.int64)


def main(argv=None):
    
    if argv is None:
        argv = sys.argv[1:]

    header = get_colored_header()
    header += '''Utility to concatenate artefacts/dead times before using 
stream mode. Code will look for .dead and .trig files, and 
concatenate them automatically taking care of file offsets
    '''
    parser = argparse.ArgumentParser(description=header,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('datafile', help='data file')
    # parser.add_argument('-w', '--window', help='text file with artefact window files',
    #                     default=None)

    if len(argv) == 0:
        parser.print_help()
        sys.exit()

    args = parser.parse_args(argv)
    # if args.window is None:
    #     window_file = None
    # else:
    #     window_file = os.path.abspath(args.window)
    
    filename = os.path.abspath(args.datafile)
    params = CircusParser(filename)
    dead_in_ms = params.getboolean('triggers', 'dead_in_ms')
    trig_in_ms = params.getboolean('triggers', 'trig_in_ms')

    if os.path.exists(params.logfile):
        os.remove(params.logfile)

    _ = init_logging(params.logfile)
    logger = logging.getLogger(__name__)

    if params.get('data', 'stream_mode') == 'multi-files':
        data_file = params.get_data_file(source=True, has_been_created=False)
        all_times_dead = numpy.zeros((0, 2), dtype=numpy.int64)
        all_times_trig = numpy.zeros((0, 2), dtype=numpy.int64)

        for f in data_file._sources:
            name, ext = os.path.splitext(f.file_name)
            dead_file = f.file_name.replace(ext, '.dead')
            trig_file = f.file_name.replace(ext, '.trig')

            if os.path.exists(dead_file):
                print_and_log(['Found file %s' % dead_file], 'default', logger)
                times = get_dead_times(dead_file, data_file.sampling_rate, dead_in_ms)
                if times.max() > f.duration or times.min() < 0:
                    print_and_log([
                        'Dead zones larger than duration for file %s' % f.file_name,
                        '-> Clipping automatically'
                    ], 'error', logger)
                    times = numpy.minimum(times, f.duration)
                    times = numpy.maximum(times, 0)
                times += f.t_start
                all_times_dead = numpy.vstack((all_times_dead, times))

            if os.path.exists(trig_file):
                print_and_log(['Found file %s' % trig_file], 'default', logger)

                times = get_trig_times(trig_file, data_file.sampling_rate, trig_in_ms)
                if times[:, 1].max() > f.duration or times[:, 1].min() < 0:
                    print_and_log(['Triggers larger than duration for file %s' % f.file_name], 'error', logger)
                    sys.exit(0)
                times[:, 1] += f.t_start
                all_times_trig = numpy.vstack((all_times_trig, times))

        if len(all_times_dead) > 0:
            output_file = os.path.join(os.path.dirname(filename), 'dead_zones.txt')
            print_and_log(['Saving global artefact file in %s' % output_file], 'default', logger)
            if dead_in_ms:
                all_times_dead = all_times_dead.astype(numpy.float32)/data_file.sampling_rate
            numpy.savetxt(output_file, all_times_dead)

        if len(all_times_trig) > 0:
            output_file = os.path.join(os.path.dirname(filename), 'triggers.txt')
            print_and_log(['Saving global artefact file in %s' % output_file], 'default', logger)
            if trig_in_ms:
                all_times_trig = all_times_trig.astype(numpy.float32)/data_file.sampling_rate
            numpy.savetxt(output_file, all_times_trig)

    elif params.get('data', 'stream_mode') == 'single-file':
        print_and_log(['Not implemented'], 'error', logger)
        sys.exit(0)
    else:
        print_and_log(['You should select a valid stream_mode such as multi-files'], 'error', logger)
        sys.exit(0)
