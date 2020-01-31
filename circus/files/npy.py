import numpy
import re
import sys
from .raw_binary import RawBinaryFile
from numpy.lib.format import open_memmap


class NumpyFile(RawBinaryFile):

    description = "numpy"
    extension = [".npy"]
    parallel_write = True
    is_writable = True

    _required_fields = {
        'sampling_rate': float
    }

    _default_values = {
        'dtype_offset': 'auto',
        'gain': 1.0
    }

    def _read_from_header(self):
        
        header = {}

        self._open()
        self.size = self.data.shape
        self.grid_ids = False

        if len(self.size) == 2:
            if self.size[0] > self.size[1]:
                self.time_axis = 0
                self._shape = (self.size[0], self.size[1])
            else:
                self.time_axis = 1
                self._shape = (self.size[1], self.size[0])
            header['nb_channels'] = self._shape[1]
        elif len(self.size) == 3:
            self.grid_ids = True
            if self.size[0] > self.size[-1]:
                self.time_axis = 0
                self._shape = (self.size[0], self.size[1], self.size[2])
            else:
                self.time_axis = 1
                self._shape = (self.size[2], self.size[1], self.size[0])
            header['nb_channels'] = self._shape[1] * self._shape[2]

        header['data_dtype'] = self.data.dtype
        self.size = len(self.data)
        self._close()

        return header

    def read_chunk(self, idx, chunk_size, padding=(0, 0), nodes=None):
        
        self._open()

        t_start, t_stop = self._get_t_start_t_stop(idx, chunk_size, padding)
        do_slice = nodes is not None and not numpy.all(nodes == numpy.arange(self.nb_channels))

        if self.time_axis == 0:
            if not self.grid_ids:
                if do_slice:
                    local_chunk = self.data[t_start:t_stop, nodes].copy()
                else:
                    local_chunk = self.data[t_start:t_stop, :].copy()
            else:
                local_chunk = self.data[t_start:t_stop, :, :].copy().reshape(t_stop-t_start, self.nb_channels)
                if do_slice:
                    local_chunk = numpy.take(local_chunk, nodes, axis=1)
        elif self.time_axis == 1:
            if not self.grid_ids:
                if do_slice:
                    local_chunk = self.data[nodes, t_start:t_stop].copy().T
                else:
                    local_chunk = self.data[:, t_start:t_stop].copy().T
            else:
                local_chunk = self.data[:, :, t_start:t_stop].copy().reshape(self.nb_channels, t_stop-t_start).T
                if do_slice:
                    local_chunk = numpy.take(local_chunk, nodes, axis=1)
        self._close()

        return self._scale_data_to_float32(local_chunk)

    def write_chunk(self, time, data):
        self._open(mode='r+')
        data = self._unscale_data_from_float32(data)
        if self.time_axis == 0:
            if not self.grid_ids:
                self.data[time:time+len(data)] = data
            else:
                self.data[time:time+len(data), :, :] = data.reshape(len(data), self._shape[1], self._shape[2])
        elif self.time_axis == 1:
            if not self.grid_ids:
                self.data[:, time:time+len(data)] = data.T
            else:
                self.data[:, :, time:time+len(data)] = data.reshape(len(data), self._shape[1], self._shape[2]).T
        self._close()

    def _open(self, mode='r'):
        self.data = open_memmap(self.file_name, mode=mode)

    def _close(self):
        self.data = None
