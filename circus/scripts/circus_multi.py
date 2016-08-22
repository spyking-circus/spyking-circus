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
from circus.shared.files import print_error, print_and_log
from circus.shared.algorithms import slice_result
import colorama
colorama.init(autoreset=True)
from colorama import Fore, Back, Style


def main(argv=None):
    
    if argv is None:
        argv = sys.argv[1:]

    gheader = Fore.GREEN + '''
##################################################################
#####            Welcome to the SpyKING CIRCUS (0.5)         #####
#####                                                        #####
#####              Written by P.Yger and O.Marre             #####
##################################################################

'''
    header  = gheader + Fore.RESET

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
    params         = circus.shared.utils.io.load_parameters(filename)
    file_out_suff  = params.get('data', 'file_out_suff')

    if not params.get('data', 'multi-files'):
        print_and_log(['Not a multi-file!'], 'error', params)
        sys.exit(0)

    to_process  = circus.shared.files.get_multi_files(params)
    result      = circus.shared.files.get_results(params, extension=extension)
    times       = circus.shared.files.data_stats(params, show=False, export_times=True)
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
