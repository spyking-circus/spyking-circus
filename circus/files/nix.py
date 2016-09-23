import h5py, numpy, re, sys
from circus.shared.messages import print_error, print_and_log
from hdf5 import H5File

class NWBFile(H5File):

    _description    = "nix"    
    _extension      = [".nix", ".h5", ".hdf5"]
    _parallel_write = h5py.get_config().mpi
    _is_writable    = True

    def __init__(self, file_name, is_empty=False, **kwargs):

        H5File.__init__(self, file_name, is_empty, **kwargs)
