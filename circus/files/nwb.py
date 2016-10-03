import h5py, numpy, re, sys
import ConfigParser as configparser
from hdf5 import H5File

class NWBFile(H5File):

    _description    = "nwb"    
    _extension      = [".nwb", ".h5", ".hdf5"]
    _parallel_write = h5py.get_config().mpi
    _is_writable    = True

    _requiered_fields = {'h5_key'            : ['string', None],
                         'dtype_offset'      : ['string', 'auto'], 
                         'sampling_rate'     : ['float', None], 
                         'gain'              : ['float', 1.]}


    def __init__(self, file_name, is_empty=False, **kwargs):

        H5File.__init__(self, file_name, is_empty, **kwargs)
