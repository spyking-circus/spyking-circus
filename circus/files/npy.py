import h5py, numpy, re, sys
from circus.shared.messages import print_error
from raw_binary import RawBinaryFile
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

        if not is_empty:
            f = open_memmap(file_name)
            kwargs['nb_channels']  = f.shape[1]
            kwargs['data_dtype']   = str(f.dtype)
            f.close()

        RawBinaryFile.__init__(self, file_name, is_empty, **kwargs)

    def _get_info_(self):
        self.dtype_offset  = get_offset(self.data_dtype, self.dtype_offset)
        self.open()
        self.size          = len(self.data)
        self._shape        = (self.size, self.nb_channels)
        self.close()


    def get_data(self, idx, chunk_size, padding=(0, 0), nodes=None):
        
        self.open()
        local_chunk  = self.data[idx*numpy.int64(chunk_size)+padding[0]:(idx+1)*numpy.int64(chunk_size)+padding[1], :]
        self.close()

        if nodes is not None:
            if not numpy.all(nodes == numpy.arange(self.nb_channels)):
                local_chunk = numpy.take(local_chunk, nodes, axis=1)

        return self._scale_data_to_float32(local_chunk)


    def set_data(self, time, data):
        self.open(mode='r+')
        data = self._unscale_data_from_from32(data)
        self.data[time:time+len(data)] = data
        self.close()

    def open(self, mode='r'):
        self.data = open_memmap(self.file_name, mode=mode)

    def allocate(self, shape, data_dtype=None):
        if data_dtype is None:
            data_dtype = self.data_dtype
        
        if self.is_master:
            self.data = open_memmap(self.file_name, shape=shape, dtype=data_dtype, mode='w+')
        comm.Barrier()
        
        self._get_info_()
        del self.data

    def close(self):
        self.data = None