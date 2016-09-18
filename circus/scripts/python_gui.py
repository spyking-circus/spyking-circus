#!/usr/bin/env python
import os, shutil, phycontrib
import sys
import subprocess
import pkg_resources
import argparse
import circus
import tempfile
import numpy, h5py
from distutils.version import LooseVersion, StrictVersion
from circus.shared.messages import print_error, print_info, print_and_log, get_header, get_colored_header
from circus.shared.files import read_probe

import logging
from phy import add_default_handler
from phy.utils._misc import _read_python
from phy.gui import create_app, run_app
from phycontrib.template import TemplateController
import numpy as np


supported_by_phy = ['raw_binary', 'mcs_raw_binary']


def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    header = get_colored_header()
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
    
    mytest = StrictVersion(phycontrib.__version__) >= StrictVersion("1.0.12")
    if not mytest:
        print_and_log(['You need to update phy-contrib to the latest git version'], 'error', params)
        sys.exit(0)

    data_file      = circus.shared.utils.io.get_data_file(params)
    sampling_rate  = data_file.rate
    data_dtype     = data_file.data_dtype
    data_offset    = data_file.data_offset
    file_format    = data_file._description
    file_out_suff  = params.get('data', 'file_out_suff')

    if file_format not in supported_by_phy:
        print_and_log(["File format %s is not supported by phy. TraceView disabled" %file_format], 'info', params)

    probe          = read_probe(params)
    if extension != '':
        extension = '-' + extension
    output_path    = params.get('data', 'file_out_suff') + extension + '.GUI'

    if not os.path.exists(output_path):
        print_and_log(['Data should be first exported with the converting method!'], 'error', params)
    else:

        print_and_log(["Launching the phy GUI..."], 'info', params)

        gui_params                   = {}
        if file_format in supported_by_phy:
            gui_params['dat_path']   = params.get('data', 'data_file')
        else:
            gui_params['dat_path']   = ''
        gui_params['n_channels_dat'] = data_file.N_tot
        gui_params['n_features_per_channel'] = 5
        gui_params['dtype']          = data_dtype
        gui_params['offset']         = data_offset
        gui_params['sample_rate']    = sampling_rate
        gui_params['hp_filtered']    = True

        os.chdir(output_path)
        create_app()
        controller = TemplateController(**gui_params)
        gui = controller.create_gui()

        gui.show()
        run_app()
        gui.close()
        del gui

if __name__ == '__main__':
    main()

    



    

