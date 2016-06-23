#!/usr/bin/env python
import os, shutil
import sys
import subprocess
import pkg_resources
import argparse
import circus
import tempfile
import numpy, h5py
from circus.shared.files import print_error, print_info, print_and_log, read_probe
import colorama
colorama.init(autoreset=True)
from colorama import Fore, Back, Style


import logging
from phy import add_default_handler
from phy.utils._misc import _read_python
from phy.gui import create_app, run_app
from phycontrib.template import TemplateController

import numpy as np

def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    gheader = Fore.GREEN + '''
##################################################################
#####            Welcome to the SpyKING CIRCUS (0.4)         #####
#####                                                        #####
#####              Written by P.Yger and O.Marre             #####
##################################################################

'''
    header  = gheader + Fore.RESET

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
    params         = circus.shared.utils.io.load_parameters(filename)
    sampling_rate  = float(params.getint('data', 'sampling_rate'))
    data_dtype     = params.get('data', 'data_dtype')
    file_out_suff  = params.get('data', 'file_out_suff')
    data_offset    = params.getint('data', 'data_offset')
    probe          = read_probe(params)
    if extension != '':
        extension = '-' + extension
    output_path    = params.get('data', 'file_out_suff') + extension + '.GUI'

    do_export      = True

    if not os.path.exists(output_path):
        print_and_log(['Data should be first exported with the converting method!'], 'error', params)
    else:

        print_and_log(["Launching the phy GUI..."], 'info', params)

        gui_params                   = {}
        gui_params['dat_path']       = params.get('data', 'data_file')
        gui_params['n_channels_dat'] = params.getint('data', 'N_total')
        gui_params['n_features_per_channel'] = 5
        gui_params['dtype']          = params.get('data', 'data_dtype')
        gui_params['offset']         = params.getint('data', 'data_offset')
        gui_params['sample_rate']    = params.getint('data', 'sampling_rate')
        gui_params['hp_filtered']    = True

        os.chdir(output_path)
        create_app()
        controller = TemplateController(**gui_params)
        gui = controller.create_gui()

        gui.show()
        run_app()
        gui.close()
        del gui

    



    

