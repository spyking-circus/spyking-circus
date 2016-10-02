import h5py, numpy, re, sys
from circus.shared.messages import print_error
from hdf5 import H5File

class NWBFile(H5File):

    _description    = "nix"    
    _extension      = [".nix", ".h5", ".hdf5"]
    _parallel_write = h5py.get_config().mpi
    _is_writable    = True

    def __init__(self, file_name, is_empty=False, **kwargs):

    	#if not is_empty:
        #    self.__check_valid_key__(file_name, kwargs['h5_key'])
        #    all_keys           = h5py.File(file_name).get(kwargs['h5_key']).keys()
        #    channels, idx      = self._get_sorted_channels_(all_keys, kwargs['channel_name'])    
        #    kwargs['channels']      = channels
        #    key                     = kwargs['h5_key'] + '/' + channels[0]
        #    kwargs['sampling_rate'] = dict(h5py.File(file_name).get(key).attrs.items())['sampling_rate']
        #    kwargs['indices']       = idx.astype(numpy.int32)

        H5File.__init__(self, file_name, is_empty, **kwargs)
