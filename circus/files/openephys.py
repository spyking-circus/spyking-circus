import h5py, numpy, re, sys, os
import ConfigParser as configparser
from circus.shared.messages import print_error, print_and_log
from datafile import DataFile

class OpenEphysFile(DataFile):

    _description = "openephys"    
    _extension   = [".openephys"]
    _parallel_write = True

    # constants
    NUM_HEADER_BYTES   = 1024
    SAMPLES_PER_RECORD = 1024
    RECORD_SIZE        = 8 + 16 + SAMPLES_PER_RECORD*2 + 10 # size of each continuous record in bytes

    def _get_sorted_channels(self, folderpath):
        return sorted([int(f.split('_CH')[1].split('.')[0]) for f in os.listdir(folderpath) 
                    if '.continuous' in f and '_CH' in f]) 

    def _read_header(self, file):
        header = { }
        f = open(file, 'r')
        h = f.read(1024).replace('\n','').replace('header.','')
        for i,item in enumerate(h.split(';')):
            if '=' in item:
                header[item.split(' = ')[0]] = item.split(' = ')[1]
        f.close()
        return header

    def __init__(self, file_name, params, empty=False, comm=None):

        DataFile.__init__(self, file_name, params, empty, comm)
        folder_path = os.path.dirname(os.path.realpath(self.file_name))
        self.all_channels = self._get_sorted_channels(folder_path)
        self.all_files = [os.path.join(folder_path, '100_CH'+x+'.continuous') for x in map(str,self.all_channels)]
        self.header = self._read_header(self.all_files[0])
        self.rate   = float(self.header['sampleRate'])
        self.data_dtype  = self.params.get('data', 'data_dtype')

        if self.data_dtype not in ['int16', 'float32']:
            print_error(['Data type for OpenEphys sould be int16 or float32'])
            sys.exit(0)
        
        self.data_offset = self.NUM_HEADER_BYTES
        self.set_dtype_offset(self.data_dtype)
        if not self.empty:
            self._get_info_()

    def _get_info_(self):

        self.empty       = False
        self.open()
        g = open(self.all_files[0], 'rb')
        self.size        = (os.fstat(g.fileno()).st_size//self.RECORD_SIZE) * self.SAMPLES_PER_RECORD
        g.close()
        self.N_tot       = len(self.all_files)
        self._shape      = (self.size, self.N_tot)
        self.max_offset  = self._shape[0]
        self.data_offset = 0
        self.close()

    def allocate(self, shape, data_dtype=None):
        pass

    def copy_header(self, file_out):
        if self.is_master:
            fin  = open(self.file_name, 'rb')
            fout = open(file_out, 'wb')
            data = fin.read(self.data_offset)
            fout.write(data)
            fin.close()
            fout.close()

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
        
        if self.data_dtype != 'int16':
            local_chunk *= self.header['bitVolts']

        local_chunk  = local_chunk.astype(numpy.float32)
        local_chunk -= self.dtype_offset

        return numpy.ascontiguousarray(local_chunk)


    def set_data(self, time, data):
        
        data += self.dtype_offset
        data  = data.astype(self.data_dtype)
        for i in xrange(self.N_tot):
            self.data[i][time:time+data.shape[0]] = data[:, i]

    def open(self, mode='r'):
        self.data = [numpy.memmap(self.all_files[i], offset=self.data_offset, dtype=numpy.int16, mode=mode) for i in xrange(self.N_tot)]
        
    def close(self):
        del self.data
