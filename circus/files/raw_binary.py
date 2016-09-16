import h5py, numpy, re, sys
import ConfigParser as configparser
from circus.shared.messages import print_error, print_and_log
from datafile import DataFile

class RawBinaryFile(DataFile):

    _description = "raw_binary"    
    _parallel_write = True

    def __init__(self, file_name, params, empty=False, comm=None):
        DataFile.__init__(self, file_name, params, empty, comm)
        
        try:
            self.data_offset = self.params.getint('data', 'data_offset')
        except Exception:
            self.data_offset = 0

        self.data_dtype  = self.params.get('data', 'data_dtype')
        self.set_dtype_offset(self.data_dtype)

        if not self.empty:
            self._get_info_()    

    def _get_info_(self):
        self.empty = False
        self.open()
        self.N      = len(self.data)
        self._shape = (self.N//self.N_tot, self.N_tot)
        self.max_offset = self._shape[0] 
        self.close()

    def allocate(self, shape, data_dtype=None):
        if data_dtype is None:
            data_dtype = self.data_dtype
        self.data = numpy.memmap(self.file_name, offset=self.data_offset, dtype=self.data_dtype, mode='w+', shape=shape)
        self._get_info_()
        del self.data

    def get_data(self, idx, chunk_size=None, padding=(0, 0), nodes=None):
    	
        if chunk_size is None:
            chunk_size = self.params.getint('data', 'chunk_size')

        chunk_len    = chunk_size * self.N_tot 
        padding      = numpy.array(padding) * self.N_tot

        self.open()
        local_chunk  = self.data[idx*numpy.int64(chunk_len)+padding[0]:(idx+1)*numpy.int64(chunk_len)+padding[1]]
        local_shape  = len(local_chunk)//self.N_tot
        local_chunk  = local_chunk.reshape(local_shape, self.N_tot)
        local_chunk  = local_chunk.astype(numpy.float32)
        local_chunk -= self.dtype_offset
        self.close()

        if nodes is not None:
            if not numpy.all(nodes == numpy.arange(self.N_tot)):
                local_chunk = numpy.take(local_chunk, nodes, axis=1)

        return numpy.ascontiguousarray(local_chunk), local_shape


    def get_snippet(self, time, length, nodes=None):
        
        self.open()
        local_chunk  = self.data[time*self.N_tot:time*self.N_tot + length*self.N_tot]
        local_chunk  = local_chunk.reshape(length, self.N_tot)
        local_chunk  = local_chunk.astype(numpy.float32)
        local_chunk -= self.dtype_offset
        self.close()
        
        if nodes is not None:
            if not numpy.all(nodes == numpy.arange(self.N_tot)):
                local_chunk = numpy.take(local_chunk, nodes, axis=1)

        return numpy.ascontiguousarray(local_chunk)

    def set_data(self, time, data):
        self.open(mode='r+')
        data  += self.dtype_offset
        data   = data.astype(self.data_dtype)
        data   = data.ravel()
        self.data[self.N_tot*time:self.N_tot*time+len(data)] = data
        self.close()

    def analyze(self, chunk_size=None):

        if chunk_size is None:
            chunk_size = self.params.getint('data', 'chunk_size')
	    
        chunk_len      = numpy.int64(self.N_tot) * chunk_size
        borders        = self.template_shift
        nb_chunks      = numpy.int64(self.N) // chunk_len
        last_chunk_len = self.N - (nb_chunks * chunk_len)
        last_chunk_len = last_chunk_len//self.N_tot
        
        if last_chunk_len > 0:
            nb_chunks += 1

        return nb_chunks, last_chunk_len

    def copy_header(self, file_out):
        if self.is_master:
            fin  = open(self.file_name, 'rb')
            fout = open(file_out, 'wb')
            data = fin.read(self.data_offset)
            fout.write(data)
            fin.close()
            fout.close()

    def open(self, mode='r'):
        self.data = numpy.memmap(self.file_name, offset=self.data_offset, dtype=self.data_dtype, mode=mode)

    def close(self):
        self.data = None