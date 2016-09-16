import h5py, numpy, re, sys
import ConfigParser as configparser
from circus.shared.messages import print_error, print_and_log
from hdf5 import H5File

class KwikFile(H5File):

    _description = "kwik"    
    _parallel_write = h5py.get_config().mpi

    '''
    /recordings
    [X]  # Recording index from 0 to Nrecordings-1
        name*
        start_time*
        start_sample*
        sample_rate*
        bit_depth*
        band_high*
        band_low*
        raw
            hdf5_path* [='{raw.kwd}/recordings/X']
        high
            hdf5_path* [='{high.kwd}/recordings/X']
        low
            hdf5_path* [='{low.kwd}/recordings/X']
    '''

    def __init__(self, file_name, params, empty=False, comm=None):

        self.main_h5_key = 'recordings/data'
        self.h5_key      = 'recordings/data/0'
        self.compression = ''
        if not self.empty:
            self._get_info_()

    def _get_info_(self):
        self.empty = False
        self.open()
        self.data_dtype  = self.my_file.get(self.h5_key).dtype
        self.compression = self.my_file.get(self.h5_key).compression

        # HDF5 does not support parallel writes with compression
        if self.compression != '':
            self._parallel_write = False
        
        self.size        = self.my_file.get(self.h5_key).shape
        self.set_dtype_offset(self.data_dtype)
        self.N_tot = len(self.my_file.get(self.main_h5_key))
        self._shape = self.size
        self.max_offset = self._shape[0]
        self.data_offset = 0
        self.close()


    def copy_header(self, file_name):
        pass

    def get_data(self):
        pass

    def open(self):
        pass