#!/usr/bin/env python
import os
import sys
import subprocess
import pkg_resources
import argparse
import circus
import tempfile
import h5py
import numpy
import logging
from circus.shared.messages import print_and_log, get_colored_header, init_logging
from circus.shared.algorithms import slice_result
from circus.shared.parser import CircusParser

def main(argv=None):
    
    if argv is None:
        argv = sys.argv[1:]

    header = get_colored_header()
    parser = argparse.ArgumentParser(description=header,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('datafile', help='data file')
    parser.add_argument('-e', '--extension', help='extension to consider for slicing results',
                        default='')

    if len(argv) == 0:
        parser.print_help()
        sys.exit()

    args = parser.parse_args(argv)

    filename       = os.path.abspath(args.datafile)
    extension      = args.extension
    if extension != '':
        extension = '-' + extension
    params         = CircusParser(filename)
    if os.path.exists(params.logfile):
        os.remove(params.logfile)
    logger         = init_logging(params.logfile)
    logger         = logging.getLogger(__name__)
    file_out_suff  = params.get('data', 'file_out_suff')

    if params.get('data', 'stream_mode') in ['None', 'none']:
        print_and_log(['No streams in the datafile!'], 'error', logger)
        sys.exit(1)

    data_file   = params.get_data_file()
    result      = circus.shared.files.get_results(params, extension=extension)
    times       = []
    for source in data_file._sources:
        times += [[source.t_start, source.t_stop]]

    sub_results = slice_result(result, times)

    for count, result in enumerate(sub_results):
        keys   = ['spiketimes', 'amplitudes']
        mydata = h5py.File(file_out_suff + '.result%s_%d.hdf5' %(extension, count), 'w', libver='latest')
        for key in keys:
            mydata.create_group(key)
            for temp in result[key].keys():
                tmp_path = '%s/%s' %(key, temp)
                mydata.create_dataset(tmp_path, data=result[key][temp])
        mydata.close()
