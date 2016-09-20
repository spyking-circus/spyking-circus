import h5py, numpy, re, sys
import ConfigParser as configparser
from circus.shared.messages import print_error, print_and_log
from hdf5 import H5File
from datafile import _check_requierements_


class KwdFile(H5File):

    _description = "kwd"    
    _extension   = [".kwd"]
    _parallel_write = h5py.get_config().mpi

    _requiered_fields = {'recording_number'  : ['int', 0], 
                         'sampling_rate'     : ['float', None]}


    def __init__(self, file_name, params, empty=False, comm=None):

        kwargs = {}
        kwargs = _check_requierements_(self._description, self._requiered_fields, params, **kwargs)
        kwargs['h5_key'] = 'recordings/%s/data' %kwargs['recording_number']
        H5File.__init__(self, file_name, params, empty, comm, **kwargs)