import h5py, numpy, re, sys, os
from datafile import DataFile, comm

class RawBinaryFile(DataFile):

    description    = "raw_binary"
    extension      = []
    parallel_write = True
    is_writable    = True

    _required_fields = {'data_dtype'    : str,
                        'sampling_rate' : float,
                        'nb_channels'   : int}

    _default_values  = {'dtype_offset'  : 'auto',
                        'data_offset'   : 0,
                        'gain'          : 1.}

    def _read_from_header(self):
        self._open()
        self.size   = len(self.data)
        self._shape = (self.size//self.nb_channels, self.nb_channels)
        self._close()
        return {}

    def allocate(self, shape, data_dtype=None):
        if data_dtype is None:
            data_dtype = self.data_dtype

        if self.is_master:
            self.data = numpy.memmap(self.file_name, offset=self.data_offset, dtype=data_dtype, mode='w+', shape=shape)
        comm.Barrier()

        self._read_from_header()
        del self.data

    def read_chunk(self, idx, chunk_size, padding=(0, 0), nodes=None):

        t_start, t_stop = self._get_t_start_t_stop(idx, chunk_size, padding)
        t_start, t_stop = int(t_start), int(t_stop)
        local_shape     = t_stop - t_start

        self._open()
        local_chunk  = self.data[t_start*self.nb_channels:t_stop*self.nb_channels]
        local_chunk  = local_chunk.reshape(local_shape, self.nb_channels)
        self._close()

        if nodes is not None:
            if not numpy.all(nodes == numpy.arange(self.nb_channels)):
                local_chunk = numpy.take(local_chunk, nodes, axis=1)

        return self._scale_data_to_float32(local_chunk)


    def write_chunk(self, time, data):
        self._open(mode='r+')

        data = self._unscale_data_from_from32(data)
        data = data.ravel()
        self.data[self.nb_channels*time:self.nb_channels*time+len(data)] = data
        self._close()


    def _open(self, mode='r'):
        self.data = numpy.memmap(self.file_name, offset=self.data_offset, dtype=self.data_dtype, mode=mode)

    def _close(self):
        self.data = None
