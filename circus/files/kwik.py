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

        self.h5_key      = 'recordings/data'
        self.compression = ''
        #self.rate        = 
        self.N_e         = params.getint('data', 'N_e')
        self.N_tot       = params.getint('data', 'N_total')
        self.rate        = params.getint('data', 'sampling_rate')
        if not self.empty:
            self._get_info_()


    def copy_header(self, file_name):
        pass