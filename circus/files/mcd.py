import h5py, numpy, re, sys
from circus.shared.messages import print_error, print_and_log
from datafile import DataFile
import neuroshare as ns

class MCDFile(DataFile):

    _description    = "mcd"    
    _extension      = [".mcd"]
    _parallel_write = False
    _is_writable    = False

    def __init__(self, file_name, params, empty=False, comm=None):

        kwargs = {}
        kwargs['data_offset']  = 0
        kwargs['data_dtype']   = 'float64'
        kwargs['dtype_offset'] = 0
        
        if not empty:
            f = ns.File(file_name)
            kwargs['N_tot'] = f.entity_count
            kwargs['rate']  = f.entities[0].sample_rate
            kwargs['gain']  = f.entities[0].resolution
            f.close()

        DataFile.__init__(self, file_name, params, empty, comm, **kwargs)


    def _get_info_(self):
        self.empty = False
        self.open()        
        self.size  = self.data.time_span * self.rate
        self._shape = (self.size, self.N_tot)
        self.max_offset = self._shape[0]
        self.close()

    def allocate(self, shape, data_dtype=None):
        print_error(['No write support for %s file' %self._description])
        sys.exit(0)

    def get_data(self, idx, chunk_size=None, padding=(0, 0), nodes=None):
        
        if chunk_size is None:
            chunk_size = self.params.getint('data', 'chunk_size')

        t_start     = numpy.int64(idx*numpy.int64(chunk_size)+padding[0])
        t_stop      = numpy.int64((idx+1)*numpy.int64(chunk_size)+padding[1])
        local_shape = t_stop - t_start

        if (t_start + local_shape) > self.max_offset:
            local_shape = self.max_offset - t_start

        if nodes is None:
            nodes = numpy.arange(self.N_tot, dtype=numpy.int32)

        local_chunk = numpy.zeros((local_shape, len(nodes)), dtype=self.data_dtype)

        for count, i in enumerate(nodes):
            local_chunk[:, count] = self.data.get_entity(numpy.int64(i)).get_data(t_start, numpy.int64(local_shape))[0]
        
        local_chunk  = local_chunk.astype(numpy.float32)

        return numpy.ascontiguousarray(local_chunk)

    def set_data(self, time, data):
        print_error(['No write support for %s file' %self._description])
        sys.exit(0)

    def open(self, mode='r'):
        self.data = ns.File(self.file_name)
        
    def close(self):
        self.data.close()
