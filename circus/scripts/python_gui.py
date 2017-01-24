#!/usr/bin/env python
import os, shutil, phycontrib
import sys
import subprocess
import pkg_resources
import argparse
import circus
import tempfile
import numpy, h5py, logging
from distutils.version import LooseVersion, StrictVersion
from circus.shared.messages import print_and_log, get_header, get_colored_header, init_logging
from circus.shared.parser import CircusParser

from phy import add_default_handler
from phy.utils._misc import _read_python
from phy.gui import create_app, run_app
from phycontrib.template import TemplateController
import numpy as np

supported_by_phy = ['raw_binary', 'mcs_raw_binary']

#------------------------------------------------------------------------------
# Set up logging with the CLI tool
#------------------------------------------------------------------------------

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
    params         = CircusParser(filename)
    if os.path.exists(params.logfile):
        os.remove(params.logfile)
    logger         = init_logging(params.logfile)
    logger         = logging.getLogger(__name__)

    mytest = StrictVersion(phycontrib.__version__) >= StrictVersion("1.0.12")
    if not mytest:
        print_and_log(['You need to update phy-contrib to the latest git version'], 'error', logger)
        sys.exit(1)

    data_file      = params.get_data_file()
    data_dtype     = data_file.data_dtype
    if hasattr(data_file, 'data_offset'):
        data_offset = data_file.data_offset
    else:
        data_offset = 0

    file_format    = data_file.description
    file_out_suff  = params.get('data', 'file_out_suff')

    if file_format not in supported_by_phy:
        print_and_log(["File format %s is not supported by phy. TraceView disabled" %file_format], 'info', logger)

    if numpy.iterable(data_file.gain):
        print_and_log(['Multiple gains are not supported, using a default value of 1'], 'info', logger)
        gain = 1
    else:
        if data_file.gain != 1:
            print_and_log(["Gain of %g is not supported by phy. Expecting a scaling mismatch" %data_file.gain], 'info', logger)
            gain = data_file.gain

    probe          = params.probe
    if extension != '':
        extension = '-' + extension
    output_path    = params.get('data', 'file_out_suff') + extension + '.GUI'

    if not os.path.exists(output_path):
        print_and_log(['Data should be first exported with the converting method!'], 'error', logger)
    else:

        print_and_log(["Launching the phy GUI..."], 'info', logger)

        gui_params                   = {}
        if file_format in supported_by_phy:
            if not params.getboolean('data', 'overwrite'):
                gui_params['dat_path']   = params.get('data', 'data_file_no_overwrite')
            else:
                gui_params['dat_path']   = params.get('data', 'data_file')
        else:
            gui_params['dat_path']   = ''
        gui_params['n_channels_dat'] = params.nb_channels
        gui_params['n_features_per_channel'] = 5
        gui_params['dtype']          = data_dtype
        gui_params['offset']         = data_offset
        gui_params['sample_rate']    = params.rate
        gui_params['dir_path']       = output_path
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
