import h5py, numpy, re, sys
import pyMCStream as mc
from .datafile import DataFile

class MCDFile(DataFile):

    description    = "mcd"    
    extension      = [".mcd"]
    parallel_write = False
    is_writable    = False

    _params            = {'data_dtype'   : 'uint16',
                          'dtype_offset' : 0}

    def _read_from_header(self):

        header = {}
        self._open()
        header['nb_channels']   = self.stream.shape[0]
        header['sampling_rate'] = self.stream.fs
        header['gain']          = 1.

        self.size   = self.stream.shape
        self._shape = (self.stream.shape[1], self.stream.shape[0])

        return header

    def read_chunk(self, idx, chunk_size, padding=(0, 0), nodes=None):

        t_start, t_stop = self._get_t_start_t_stop(idx, chunk_size, padding)
        local_shape     = t_stop - t_start

        if nodes is None:
            nodes = numpy.arange(self.nb_channels, dtype=numpy.int32)

        local_chunk = self.stream[:, t_start:t_stop].T

        if nodes is not None:
            if not numpy.all(nodes == numpy.arange(self.nb_channels)):
                local_chunk = numpy.take(local_chunk, nodes, axis=1)

        return local_chunk.astype(numpy.float32)

    def _open(self, mode='r'):
        self.stream = mc.open_stream(self.file_name, 'analog')

    def _close(self):
        del self.stream
