import h5py, numpy, re, sys
from hdf5 import H5File

class KwdFile(H5File):

    description    = "kwd"    
    extension      = [".kwd"]
    parallel_write = h5py.get_config().mpi

    _required_fields = {'sampling_rate'    : float}
    
    _default_values  = {'recording_number'  : 0, 
                       'dtype_offset'       : 'auto',
                       'gain'               : 1.}


    def __init__(self, file_name, params, is_empty=False):

        params['h5_key'] = 'recordings/%s/data' %params['recording_number']
        H5File.__init__(self, file_name, params, is_empty)

    def _read_from_header_(self):
        header = H5File._read_from_header_(self)
        header['gain'] = dict(h5py.File(file_name).get('recordings/0/application_data').attrs.items())['channel_bit_volts']
        return header