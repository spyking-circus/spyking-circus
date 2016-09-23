import h5py, numpy, re, sys
from circus.shared.messages import print_error, print_and_log
from datafile import DataFile, get_offset

class RawBinaryFile(DataFile):

    _description    = "raw_binary"    
    _extension      = None
    _parallel_write = True
    _is_writable    = True

    _requiered_fields = {'data_offset'   : ['int', 0],
                         'data_dtype'    : ['string', None],
                         'dtype_offset'  : ['string', 'auto'],
                         'sampling_rate' : ['float', None],
                         'gain'          : ['float', 1.],
                         'nb_channels'   : ['int' , None]}

    def __init__(self, file_name, is_empty=False, **kwargs):

        DataFile.__init__(self, file_name, is_empty, **kwargs)     
    
    def _get_info_(self):
        self.dtype_offset  = get_offset(self.data_dtype, self.dtype_offset)
        self.open()
        self.size          = len(self.data)
        self._shape        = (self.size//self.nb_channels, self.nb_channels)
        self.close()

    def allocate(self, shape, data_dtype=None):
        if data_dtype is None:
            data_dtype = self.data_dtype
        self.data = numpy.memmap(self.file_name, offset=self.data_offset, dtype=self.data_dtype, mode='w+', shape=shape)
        self._get_info_()
        del self.data

    def get_data(self, idx, chunk_size, padding=(0, 0), nodes=None):
    	
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


    def set_data(self, time, data):
        self.open(mode='r+')

        data = self._unscale_data_from_from32(data)
        data = data.ravel()
        self.data[self.nb_channels*time:self.nb_channels*time+len(data)] = data
        self.close()


    def open(self, mode='r'):
        self.data = numpy.memmap(self.file_name, offset=self.data_offset, dtype=self.data_dtype, mode=mode)

    def close(self):
        self.data = None