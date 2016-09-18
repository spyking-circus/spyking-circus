import h5py, numpy, re, sys
import ConfigParser as configparser
from circus.shared.messages import print_error, print_and_log
from hdf5 import H5File


'''
/kwik_version* [=2]
/recordings
    [X]
        data* [(Nsamples x Nchannels) EArray of Int16]
        filter
            name*
            param1*
        downsample_factor*

        # The following metadata fields are duplicated from the .kwik files
        # and are here for convenience only. The KWIK programs will not read
        # them, they are only there for other programs.
        name
        start_time
        start_sample
        sample_rate
        bit_depth

        application_data
            band_high
            band_low
'''

class KwdFile(H5File):

    _description = "kwd"    
    _extension   = [".kwd"]
    _parallel_write = h5py.get_config().mpi

    def __init__(self, file_name, params, empty=False, comm=None):

        H5File.__init__(self, file_name, params, True, comm)
        self.h5_key      = 'recordings/0/data'
        self.empty = empty
        if not self.empty:
            self._get_info_(self.h5_key)