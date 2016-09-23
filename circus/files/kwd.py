import h5py, numpy, re, sys
from circus.shared.messages import print_error, print_and_log
from hdf5 import H5File

class KwdFile(H5File):

    _description = "kwd"    
    _extension   = [".kwd"]
    _parallel_write = h5py.get_config().mpi

    _requiered_fields = {'recording_number'  : ['int', 0],
                         'dtype_offset'      : ['string', 'auto'], 
                         'sampling_rate'     : ['float', None], 
                         'gain'              : ['float', 1.]}


    def __init__(self, file_name, is_empty=False, **kwargs):

        kwargs['h5_key'] = 'recordings/%s/data' %kwargs['recording_number']
        H5File.__init__(self, file_name, is_empty, **kwargs)