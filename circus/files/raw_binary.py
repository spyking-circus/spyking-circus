import h5py, numpy, re, sys, os
from datafile import DataFile, get_offset, comm

class RawBinaryFile(DataFile):

    description    = "raw_binary"    
    extension      = None
    parallel_write = True
    is_writable    = True
    is_streamable  = True

    _required_fields = {'data_dtype'    : str,
                        'sampling_rate' : float,
                        'nb_channels'   : int}
    
    _default_values  = {'dtype_offset'  : 'auto', 
                        'data_offset'   : 0,
                        'gain'          : 1}

    def _read_from_header(self):
        self.open()
        self.size   = len(self.data)
        self._shape = (self.size//self.nb_channels, self.nb_channels)
        self.close()
        return {}

    def set_streams(self):
        dirname     = os.path.abspath(os.path.dirname(self.file_name))
        all_files   = os.listdir(dirname)
        pattern     = os.path.basename(self.file_name)
        streams     = []
        count       = 0

        while pattern in all_files:
            to_process = os.path.join(os.path.abspath(dirname), pattern)
            pattern    = pattern.replace(str(count), str(count+1))
            count     += 1
            streams   += [RawBinaryFile(to_process, self.get_description())]
            print to_process

        return streams

    def allocate(self, shape, data_dtype=None):
        if data_dtype is None:
            data_dtype = self.data_dtype
        
        if self.is_master:
            self.data = numpy.memmap(self.file_name, offset=self.data_offset, dtype=data_dtype, mode='w+', shape=shape)
        comm.Barrier()
        
        self._read_from_header()
        del self.data

    def read_chunk(self, idx, chunk_size, padding=(0, 0), nodes=None):
    	
        chunk_size  *= self.nb_channels
        padding      = numpy.array(padding) * self.nb_channels

        self.open()
        local_chunk  = self.data[idx*numpy.int64(chunk_size)+padding[0]:(idx+1)*numpy.int64(chunk_size)+padding[1]]
        local_shape  = len(local_chunk)//self.nb_channels
        local_chunk  = local_chunk.reshape(local_shape, self.nb_channels)
        self.close()

        if nodes is not None:
            if not numpy.all(nodes == numpy.arange(self.nb_channels)):
                local_chunk = numpy.take(local_chunk, nodes, axis=1)

        return self._scale_data_to_float32(local_chunk)


    def write_chunk(self, time, data):
        self.open(mode='r+')

        data = self._unscale_data_from_from32(data)
        data = data.ravel()
        self.data[self.nb_channels*time:self.nb_channels*time+len(data)] = data
        self.close()


    def open(self, mode='r'):
        self.data = numpy.memmap(self.file_name, offset=self.data_offset, dtype=self.data_dtype, mode=mode)

    def close(self):
        self.data = None