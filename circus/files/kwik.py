import h5py, numpy, re, sys
import ConfigParser as configparser
from circus.shared.messages import print_error, print_and_log
from hdf5 import H5File

class KwikFile(H5File):

    _description = "kwik"    
    _parrallel_write = h5py.get_config().mpi

    def __init__(self, file_name, params, empty=False, comm=None):

        self.h5_key      = self.params.get('data', 'hdf5_key_data')
        self.compression = 'gzip'
        if not self.empty:
            self._get_info_()