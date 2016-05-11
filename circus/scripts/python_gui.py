#!/usr/bin/env python
import os, shutil
import sys
import subprocess
import pkg_resources
import circus
import tempfile
import numpy, h5py
from circus.shared.files import print_error, print_info, print_and_log, write_datasets, get_results, read_probe, load_data, get_nodes_and_edges, load_data, get_stas
from circus.shared.utils import get_progressbar

import numpy as np

import logging

from phy import add_default_handler
from phy.utils._misc import _read_python
from phy.gui import create_app, run_app
from phycontrib.template import TemplateController


def main():

    argv = sys.argv

    if len(sys.argv) < 2:
        print_error(['No data file!'])
        message = '''   
Syntax is circus-gui-python datafile [extension]
        '''
        print(message)

        sys.exit(0)

    filename       = os.path.abspath(sys.argv[1])

    if len(sys.argv) == 2:
        filename   = os.path.abspath(sys.argv[1])
        extension  = ''
    elif len(sys.argv) == 3:
        filename   = os.path.abspath(sys.argv[1])
        extension  = sys.argv[2]

    params         = circus.shared.utils.io.load_parameters(filename)
    sampling_rate  = float(params.getint('data', 'sampling_rate'))
    data_dtype     = params.get('data', 'data_dtype')
    file_out_suff  = params.get('data', 'file_out_suff')
    data_offset    = params.getint('data', 'data_offset')
    probe          = read_probe(params)
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

    



    

