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


    def _read_from_header(self):
        
        self._params['h5_key'] = 'recordings/%s/data' %self._params['recording_number']
        
        header           = H5File._read_from_header(self)
        header['h5_key'] = self.h5_key
        header['gain']   = dict(h5py.File(self.file_name).get('recordings/0/application_data').attrs.items())['channel_bit_volts']
        
        return header