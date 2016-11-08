import h5py, numpy, re, sys
from datafile import DataFile
import neuroshare as ns

class NeuroShareFile(DataFile):

    description    = "neuroshare"    
    extension      = []
    parallel_write = False
    is_writable    = False   
    _params        = {'data_dtype'   : 'float64',
                      'dtype_offset' : 0}     

    def _read_from_header(self):

        header = {}        
        self._open()
        header['nb_channels']   = self.data.entity_count
        header['sampling_rate'] = self.data.entities[0].sample_rate
        header['gain']          = self.data.entities[0].resolution

        self.size   = self.data.time_span * header['sampling_rate']
        self._shape = (self.size, header['nb_channels'])
        self._close()

        return header


    def read_chunk(self, idx, chunk_size, padding=(0, 0), nodes=None):
        
        t_start, t_stop = self._get_t_start_t_stop(idx, chunk_size, padding)
        local_shape     = t_stop - t_start

        if nodes is None:
            nodes = numpy.arange(self.nb_channels, dtype=numpy.int32)

        local_chunk = numpy.zeros((local_shape, len(nodes)), dtype=self.data_dtype)

        for count, i in enumerate(nodes):
            local_chunk[:, count] = self.data.get_entity(numpy.int64(i)).get_data(t_start, numpy.int64(local_shape))[0]
        
        return self._scale_data_to_float32(local_chunk)


    def _open(self, mode='r'):
        self.data = ns.File(self.file_name)
        
        
    def _close(self):
        self.data.close()
