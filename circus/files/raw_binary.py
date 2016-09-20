import h5py, numpy, re, sys
import ConfigParser as configparser
from circus.shared.messages import print_error, print_and_log
from datafile import DataFile, _check_requierements_

class RawBinaryFile(DataFile):

    _description    = "raw_binary"    
    _extension      = None
    _parallel_write = True
    _is_writable    = True

    _requiered_fields = {'data_offset'   : ['int', 0],
                         'data_dtype'    : ['string', None],
                         'dtype_offset'  : ['string', 'auto'],
                         'sampling_rate' : ['float', None],
                         'gain'          : ['float', 1]}

    def __init__(self, file_name, params, empty=False, comm=None, **kwargs):

        kwargs = _check_requierements_(self._description, self._requiered_fields, params, **kwargs)

        if kwargs['dtype_offset'] == 'auto':
            if kwargs['data_dtype'] == 'uint16':
                kwargs['dtype_offset'] = 32768
            elif kwargs['data_dtype'] == 'int16':
                kwargs['dtype_offset'] = 0
            elif kwargs['data_dtype'] == 'float32':
                kwargs['dtype_offset'] = 0
            elif kwargs['data_dtype'] == 'int8':
                kwargs['dtype_offset'] = 0        
            elif kwargs['data_dtype'] == 'uint8':
                kwargs['dtype_offset'] = 127
            elif kwargs['data_dtype'] == 'float64':
                kwargs['dtype_offset'] = 0    
        else:
            try:
                kwargs['dtype_offset'] = int(kwargs['dtype_offset'])
            except Exception:
                print_error(["Offset %s is not valid" %kwargs['dtype_offset']])
                sys.exit(0)

        DataFile.__init__(self, file_name, params, empty, comm, **kwargs)     
    
    def _get_info_(self):
        self.empty = False
        self.open()
        self.size       = len(self.data)
        self._shape     = (self.size//self.N_tot, self.N_tot)
        self.max_offset = self._shape[0] 
        self.close()

    def allocate(self, shape, data_dtype=None):
        if data_dtype is None:
            data_dtype = self.data_dtype
        self.data = numpy.memmap(self.file_name, offset=self.data_offset, dtype=self.data_dtype, mode='w+', shape=shape)
        self._get_info_()
        del self.data

    def get_data(self, idx, chunk_size=None, padding=(0, 0), nodes=None):
    	
        chunk_size   = self._get_chunk_size_(chunk_size) * self.N_tot
        padding      = numpy.array(padding) * self.N_tot

        self.open()
        local_chunk  = self.data[idx*numpy.int64(chunk_size)+padding[0]:(idx+1)*numpy.int64(chunk_size)+padding[1]]
        local_shape  = len(local_chunk)//self.N_tot
        local_chunk  = local_chunk.reshape(local_shape, self.N_tot)
        local_chunk  = local_chunk.astype(numpy.float32)*self.gain
        local_chunk -= self.dtype_offset
        self.close()

        if nodes is not None:
            if not numpy.all(nodes == numpy.arange(self.N_tot)):
                local_chunk = numpy.take(local_chunk, nodes, axis=1)

        return numpy.ascontiguousarray(local_chunk)

    def set_data(self, time, data):
        self.open(mode='r+')
        data  += self.dtype_offset
        data   = data.astype(self.data_dtype)/self.gain
        data   = data.ravel()
        self.data[self.N_tot*time:self.N_tot*time+len(data)] = data
        self.close()

    def analyze(self, chunk_size=None):

        chunk_size     = self._get_chunk_size_(chunk_size) * self.N_tot
        nb_chunks      = numpy.int64(self.size) // chunk_size
        last_chunk_len = self.size - (nb_chunks * chunk_size)
        last_chunk_len = last_chunk_len//self.N_tot
        
        if last_chunk_len > 0:
            nb_chunks += 1

        return nb_chunks, last_chunk_len

    def open(self, mode='r'):
        self.data = numpy.memmap(self.file_name, offset=self.data_offset, dtype=self.data_dtype, mode=mode)

    def close(self):
        self.data = None