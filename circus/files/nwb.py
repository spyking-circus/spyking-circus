import h5py, numpy, re, sys
import ConfigParser as configparser
from circus.shared.messages import print_error, print_and_log
from hdf5 import H5File

class NWBFile(H5File):

    _description    = "nwb"    
    _extension      = [".nwb", ".h5", ".hdf5"]
    _parallel_write = h5py.get_config().mpi
    _is_writable    = True

    def __init__(self, file_name, params, empty=False):

        H5File.__init__(self, file_name, params, True)

        self.h5_key = self.params.get('data', 'hdf5_key_data')
        self.empty  = empty
        if not self.empty:
            self._get_info_(self.h5_key)
