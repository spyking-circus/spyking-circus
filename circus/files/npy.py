import h5py, numpy, re, sys
from raw_binary import RawBinaryFile
from numpy.lib.format import open_memmap

class NumpyFile(RawBinaryFile):

    description    = "numpy"
    extension      = [".npy"]
    parallel_write = True
    is_writable    = True

    _required_fields = {'sampling_rate' : float}

    _default_values  = {'dtype_offset'  : 'auto',
                        'gain'          : 1.}

    def _read_from_header(self):
        
        header = {}

        self.open()
        header['nb_channels']  = self.data.shape[1]
        header['data_dtype']   = self.data.dtype

        self.size          = len(self.data)
        self._shape        = (self.size, self.nb_channels)
        self.close()

        return header


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