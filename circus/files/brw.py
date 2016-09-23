import h5py, numpy, re, sys
from circus.shared.messages import print_error, print_and_log
from hdf5 import H5File

class BRWFile(H5File):

    _description = "brw"    
    _extension   = [".brw"]
    _parallel_write = h5py.get_config().mpi

    _requiered_fields = {'recording_number'  : ['int', 0],
                         'dtype_offset'      : ['string', 'auto'], 
                         'sampling_rate'     : ['float', None], 
                         'gain'              : ['float', 1.]}


    def __init__(self, file_name, is_empty=False, **kwargs):

        
        if not is_empty:
            f = h5py.File(file_name)
            kwargs['sampling_rate'] = f.get('3BRecInfo/3BRecVars/SamplingRate').value
            kwargs['n_frames']      = f.get('3BRecInfo/3BRecVars/NRecFrames').value
            kwargs['h5_key']        = 'recordings/%s/data' %kwargs['recording_number']



        H5File.__init__(self, file_name, is_empty, **kwargs)