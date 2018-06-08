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

os_symlink = getattr(os, "symlink", None)
if callable(os_symlink):
    pass
else:
    def symlink_ms(source, link_name):
        import ctypes
        csl = ctypes.windll.kernel32.CreateSymbolicLinkW
        csl.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
        csl.restype = ctypes.c_ubyte
        flags = 1 if os.path.isdir(source) else 0
        if csl(link_name, source.replace('/', '\\'), flags) == 0:
            raise ctypes.WinError()
    os.symlink = symlink_ms

def main(argv=None):
    
    if argv is None:
        argv = sys.argv[1:]

    header = get_colored_header()
    header += '''Utility to group files within several folders into a single
virtual folder, such that they can be processed together with the
multi-files mode
    '''

    parser = argparse.ArgumentParser(description=header,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('folders', help='a text file with the list of folders to consider')
    parser.add_argument('extension', help='the file extension to consider within folders')
    parser.add_argument('output', help='the name of the output folder', default='sc_folders')

    if len(argv) == 0:
        parser.print_help()
        sys.exit()

    args = parser.parse_args(argv)

    folders_file = os.path.abspath(args.folders)
    output      = os.path.abspath(args.output)
    extension   = args.extension
    
    filename, ext = os.path.splitext(os.path.basename(folders_file))

    logger = init_logging(filename + '.log')
    logger = logging.getLogger(__name__)

    if not os.path.exists(folders_file):
        print_and_log(['The folder file %s does not exists!' %folders_file], 'error', logger)
        sys.exit(0)

    try:
        folders = []
        myfile = open(folders_file, 'r')
        lines  = myfile.readlines()
        myfile.close()
        for l in lines:
            folders += [os.path.abspath(l.strip())]
    except Exception:
        print_and_log(['Check the syntax of the folder file'], 'error', logger)
        sys.exit(0)        

    do_folders = True

    if os.path.exists(output):
        do_folders = query_yes_no(Fore.WHITE + "Folder %s already exists! Do you want to erase everything?" %output, default=None)
        if not do_folders:
            sys.exit(0)
        else:
            shutil.rmtree(output)

    os.makedirs(output)

    for folder in folders:
        files = os.listdir(folder)
        for file in files:
            _, ext = os.path.splitext(file)
            ext = ext.strip('.')
            if ext.lower() == extension.lower():
                original_file = os.path.join(folder, file)
                linked_file = os.path.join(output, os.path.basename(original_file))
                if not os.path.exists(linked_file):
                    os.symlink(original_file, linked_file)
                else:
                    linked_file = linked_file.replace('02230_2015', '02231_2015')
                    os.symlink(original_file, linked_file)
