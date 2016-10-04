import h5py, numpy, re, sys
from datafile import DataFile
import neuroshare as ns

class MCDFile(DataFile):

    _description    = "mcd"    
    _extension      = [".mcd"]
    _parallel_write = False
    _is_writable    = False

    def __init__(self, file_name, is_empty=False, **kwargs):

        kwargs['data_dtype']   = 'float64'
        kwargs['dtype_offset'] = 0
        
        if not is_empty:
            f = ns.File(file_name)
            kwargs['nb_channels']   = f.entity_count
            kwargs['sampling_rate'] = f.entities[0].sample_rate
            kwargs['gain']          = f.entities[0].resolution
            f.close()

        DataFile.__init__(self, file_name, is_empty, **kwargs)


    def _get_info_(self):
        self.empty = False
        self.open()        
        self.size  = self.data.time_span * self.sampling_rate
        self._shape = (self.size, self.nb_channels)
        self.close()


    def get_data(self, idx, chunk_size, padding=(0, 0), nodes=None):
        
        t_start     = numpy.int64(idx*numpy.int64(chunk_size)+padding[0])
        t_stop      = numpy.int64((idx+1)*numpy.int64(chunk_size)+padding[1])
        local_shape = t_stop - t_start

        if (t_start + local_shape) > self.duration:
            local_shape = self.duration - t_start

        if nodes is None:
            nodes = numpy.arange(self.nb_channels, dtype=numpy.int32)

        local_chunk = numpy.zeros((local_shape, len(nodes)), dtype=self.data_dtype)

        for count, i in enumerate(nodes):
            local_chunk[:, count] = self.data.get_entity(numpy.int64(i)).get_data(t_start, numpy.int64(local_shape))[0]
        
        return self._scale_data_to_float32(local_chunk)


    def open(self, mode='r'):
        self.data = ns.File(self.file_name)
        
    def close(self):
        self.data.close()
