import h5py, numpy, re, sys
from datafile import DataFile
import neuroshare as ns

class MCDFile(DataFile):

    description    = "mcd"    
    extension      = [".mcd"]
    parallel_write = False
    is_writable    = False
        

    def _read_from_header(self):

        header                 = {}
        header['data_dtype']   = 'float64'
        header['dtype_offset'] = 0
        
        self.open()
        header['nb_channels']   = self.data.entity_count
        header['sampling_rate'] = self.data.entities[0].sample_rate
        header['gain']          = self.data.entities[0].resolution

        self.size  = self.data.time_span * self.sampling_rate
        self._shape = (self.size, self.nb_channels)
        self.close()

        return header


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
