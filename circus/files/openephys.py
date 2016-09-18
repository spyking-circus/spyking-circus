import h5py, numpy, re, sys, os
import ConfigParser as configparser
from circus.shared.messages import print_error, print_and_log
from datafile import DataFile

class OpenEphysFile(DataFile):

    _description = "openephys"    
    _extension   = [".openephys"]
    _parallel_write = True

    # constants
    NUM_HEADER_BYTES = 1024L
    SAMPLES_PER_RECORD = 1024L
    RECORD_SIZE = 8 + 16 + SAMPLES_PER_RECORD*2 + 10 # size of each continuous record in bytes
    RECORD_MARKER = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 255])

    def _get_sorted_channels(self, folderpath):
        return sorted([int(f.split('_CH')[1].split('.')[0]) for f in os.listdir(folderpath) 
                    if '.continuous' in f and '_CH' in f]) 

    def __init__(self, file_name, params, empty=False, comm=None):

        DataFile.__init__(self, file_name, params, empty, comm)
        folder_path = os.path.dirname(os.path.realpath(self.file_name))
        self.all_channels = self._get_sorted_channels(folder_path)
        self.all_files = [os.path.join(folder_path, '100_CH'+x+'.continuous') for x in map(str,self.all_channels)]
        
        if not self.empty:
            self._get_info_()

    def _get_info_(self):

        self.empty = False
        self.open()
        #self.data_dtype  = self.my_file.get(self.h5_key + '0').dtype
        #self.compression = self.my_file.get(self.h5_key + '0').compression
        #self.rate        = self.my_file.get(self.h5_key + 'rate')[0]

        #self.size = self.my_file.get(self.h5_key + '/0').shape[0]
        self.set_dtype_offset(self.data_dtype)
        self.N_tot       = len(self.my_file.get(self.h5_key))
        self._shape      = (self.size, self.N_tot)
        self.max_offset  = self._shape[0]
        self.data_offset = 0
        self.close()

    def allocate(self, shape, data_dtype=None):
        pass

    def get_data(self, idx, chunk_size=None, padding=(0, 0), nodes=None):
        
        if chunk_size is None:
            chunk_size = self.params.getint('data', 'chunk_size')

        default_shape = chunk_size + (padding[1]-padding[0])
        local_shape   = min(default_shape, self.max_offset - (idx*chunk_size + padding[0]))

        if nodes is None:
            nodes = numpy.arange(self.N_tot)

        local_chunk = numpy.zeros((local_shape, len(nodes)), dtype=self.data_dtype)

        for i in nodes:
            local_chunk[:, i] = self.data[i][idx*numpy.int64(chunk_size)+padding[0]:(idx+1)*numpy.int64(chunk_size)+padding[1]]
        
        local_chunk  = local_chunk.astype(numpy.float32)
        local_chunk -= self.dtype_offset

        return numpy.ascontiguousarray(local_chunk)

    def set_data(self, time, data):
        
        data += self.dtype_offset
        data  = data.astype(self.data_dtype)
        for i in xrange(self.N_tot):
            self.data[i][time:time+data.shape[0]] = data[:, i]

    def open(self, mode='r'):
        self.data = [numpy.memmap(self.all_files[i], offset=self.data_offset, dtype=self.data_dtype, mode=mode) for i in xrange(self.N_tot)]
        
    def close(self):
        del self.data
