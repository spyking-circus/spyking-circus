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

        self._open()
        self.size = self.data.shape

        if self.size[0] > self.size[1]:
            self.time_axis = 0
            self._shape = (self.size[0], self.size[1])
        else:
            self.time_axis = 1
            self._shape = (self.size[1], self.size[0])

        header['nb_channels'] = self._shape[1]
        header['data_dtype']  = self.data.dtype
        self.size             = len(self.data)
        self._shape           = (self.size, self.nb_channels)
        self._close()

        return header


    def read_chunk(self, idx, chunk_size, padding=(0, 0), nodes=None):
        
        self._open()

        t_start, t_stop = self._get_t_start_t_stop(idx, chunk_size, padding)

        if self.time_axis == 0:
            local_chunk  = self.data[t_start:t_stop, :]
        elif self.time_axis == 1:
            local_chunk  = self.data[:, t_start:t_stop].T
        self._close()

        if nodes is not None:
            if not numpy.all(nodes == numpy.arange(self.nb_channels)):
                local_chunk = numpy.take(local_chunk, nodes, axis=1)

        return self._scale_data_to_float32(local_chunk)


    def write_chunk(self, time, data):
        self._open(mode='r+')
        data = self._unscale_data_from_from32(data)
        if self.time_axis == 0:
            self.data[time:time+len(data)] = data
        elif self.time_axis == 1:
            self.data[:, time:time+len(data)] = data.T
        self._close()


    def _open(self, mode='r'):
        self.data = open_memmap(self.file_name, mode=mode)


    def _close(self):
        self.data = None