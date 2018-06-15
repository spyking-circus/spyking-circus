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

    if os.path.exists(params.logfile):
        os.remove(params.logfile)

    logger         = init_logging(params.logfile)
    logger         = logging.getLogger(__name__)


    if params.get('data', 'stream_mode') == 'multi-files':
        data_file = params.get_data_file(source=True, has_been_created=False)
        print ' '.join(data_file.get_file_names())
    elif params.get('data', 'stream_mode') == 'single-file':
        pass