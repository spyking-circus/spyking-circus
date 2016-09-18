import h5py, numpy, re, sys
import ConfigParser as configparser
from circus.shared.messages import print_error, print_and_log
from datafile import DataFile
import neuroshare as ns

class MCDFile(DataFile):

    _description = "mcd"    
    _extension   = [".mcd"]
    _parallel_write = False

    def __init__(self, file_name, params, empty=False, comm=None):

        DataFile.__init__(self, file_name, params, empty, comm)
        if not self.empty:
            self._get_info_()

    def _get_info_(self):
        self.empty = False
        self.open()
        self.N_tot = self.data.entity_count
        self.rate  = self.data.entities[0].sample_rate
        self.size  = self.data.time_span * self.rate
        self._shape = (self.size, self.N_tot)
        self.max_offset = self._shape[0]
        self.data_offset = 0
        self.data_dtype  = numpy.uint16
        self.close()

    def allocate(self, shape, data_dtype=None):
        raise('Not Implemented for .mcd file')


    def get_data(self, idx, chunk_size=None, padding=(0, 0), nodes=None):
        
        if chunk_size is None:
            chunk_size = self.params.getint('data', 'chunk_size')

        default_shape = chunk_size + (padding[1]-padding[0])
        local_shape   = numpy.min(default_shape, self.max_offset - (idx*chunk_size + padding[0]))

        if nodes is None:
            nodes = numpy.arange(self.N_tot)

        local_chunk = numpy.zeros((local_shape, len(nodes)), dtype=self.data_dtype)

        for i in nodes:
            local_chunk = self.data.entities[i].get_data(idx*numpy.int64(chunk_len)+padding[0], local_shape)[0]
        
        local_chunk  = local_chunk.astype(numpy.float32)
        local_chunk -= self.dtype_offset

        return numpy.ascontiguousarray(local_chunk)

    def set_data(self, time, data):
        raise('Not Implemented for .mcd file')

    def open(self, mode='r'):
        self.data = ns.File(self.file_name)
        
    def close(self):
        self.data.close()
