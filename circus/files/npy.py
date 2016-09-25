import h5py, numpy, re, sys
from circus.shared.messages import print_error, print_and_log
from raw_binary import NumpyFile
from numpy.lib.format import open_memmap

class NumpyFile(RawBinaryFile):

    _description    = "numpy"
    _extension      = [".npy"]
    _parallel_write = True
    _is_writable    = True

    _requiered_fields = {'dtype_offset'  : ['string', 'auto'],
                         'sampling_rate' : ['float', None],
                         'gain'          : ['float', 1.]}

    def __init__(self, file_name, is_empty=False, **kwargs):

        kwargs = {}

        if not is_empty:
            f = open_memmap(file_name)
            kwargs['nb_channels']  = f.shape[1]
            kwargs['data_dtype']   = str(f.dtype)
            f.close()

        RawBinaryFile.__init__(self, file_name, is_empty, **kwargs)

    def _get_info_(self):
        self.data_dtype    = 
        self.dtype_offset  = get_offset(self.data_dtype, self.dtype_offset)
        self.open()
        self.size          = len(self.data)
        self._shape        = (self.size, self.nb_channels)
        self.close()

    def open(self, mode='r'):
        self.data = open_memmap(self.file_name, mode=mode)

    def allocate(self, shape, data_dtype=None):
        if data_dtype is None:
            data_dtype = self.data_dtype
        
        if self.is_master:
            self.data = open_memmap(self.file_name, shape=(self.duration, self.nb_channels), dtype=self.data_dtype, mode='w+')
        comm.Barrier()
        
        self._get_info_()
        del self.data

    def close(self):
        self.data = None