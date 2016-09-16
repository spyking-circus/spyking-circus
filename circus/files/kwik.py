import h5py, numpy, re, sys
import ConfigParser as configparser
from circus.shared.messages import print_error, print_and_log
from hdf5 import H5File

class KwikFile(H5File):

    _description = "kwik"    
    _parallel_write = h5py.get_config().mpi

    '''
    /recordings
    [X]  # Recording index from 0 to Nrecordings-1
        name*
        start_time*
        start_sample*
        sample_rate*
        bit_depth*
        band_high*
        band_low*
        raw
            hdf5_path* [='{raw.kwd}/recordings/X']
        high
            hdf5_path* [='{high.kwd}/recordings/X']
        low
            hdf5_path* [='{low.kwd}/recordings/X']
    '''

    def __init__(self, file_name, params, empty=False, comm=None):

        self.h5_key      = 'recordings/data/'
        self.compression = ''
        if not self.empty:
            self._get_info_()

    def _get_info_(self):
        self.empty = False
        self.open()
        self.data_dtype  = self.my_file.get(self.h5_key + '0').dtype
        self.compression = self.my_file.get(self.h5_key + '0').compression
        self.rate        = self.my_file.get(self.h5_key + 'rate')[0]

        # HDF5 does not support parallel writes with compression
        if self.compression != '':
            self._parallel_write = False
        
        self.size = self.my_file.get(self.h5_key + '/0').shape[0]
        self.set_dtype_offset(self.data_dtype)
        self.N_tot       = len(self.my_file.get(self.h5_key))
        self._shape      = (self.size, self.N_tot)
        self.max_offset  = self._shape[0]
        self.data_offset = 0
        self.close()

    def allocate(self, shape, data_dtype=None):

        if data_dtype is None:
            data_dtype = self.data_dtype

        shape = (shape[0], )

        if self._parallel_write and (self.comm is not None):
            self.my_file = h5py.File(self.file_name, mode='r+', driver='mpio', comm=self.comm)
            self.my_file.create_group(self.h5_key)
            for i in xrange(self.N_tot):
                self.my_file.create_dataset(self.h5_key+str(i), dtype=data_dtype, shape=shape)
        else:
            self.my_file = h5py.File(self.file_name, mode='r+')
            if self.is_master:
                if self.compression != '':
                    self.my_file.create_dataset(self.h5_key, dtype=data_dtype, shape=shape, compression=self.compression, chunks=True)
                    for i in xrange(self.N_tot):
                        self.my_file.create_dataset(self.h5_key+str(i), dtype=data_dtype, compression=self.compression, shape=shape)
                else:
                    for i in xrange(self.N_tot):
                        self.my_file.create_dataset(self.h5_key+str(i), dtype=data_dtype, shape=shape, chunks=True)

        self.my_file.close()
        self._get_info_()

    def get_data(self, idx, chunk_size=None, padding=(0, 0), nodes=None):
        
        if chunk_size is None:
            chunk_size = self.params.getint('data', 'chunk_size')

        local_shape = numpy.max()

        if nodes is None:
            nodes = numpy.arange(self.N_tot)

        local_chunk = numpy.zeros((local_shape, len(nodes)), dtype=self.data_dtype)

        for i in nodes:
            local_chunk = self.data[i][idx*numpy.int64(chunk_len)+padding[0]:(idx+1)*numpy.int64(chunk_len)+padding[1]]
        
        local_chunk  = local_chunk.astype(numpy.float32)
        local_chunk -= self.dtype_offset

        return numpy.ascontiguousarray(local_chunk)


    def set_data(self, time, data):
        
        data += self.dtype_offset
        data  = data.astype(self.data_dtype)
        for i in xrange(self.N_tot):
            local_chunk = self.data[i][time:time+data.shape[0]] = data[:, i]


    def open(self):
        if self._parallel_write and (self.comm is not None):
            self.my_file = h5py.File(self.file_name, mode=mode, driver='mpio', comm=self.comm)
        else:
            self.my_file = h5py.File(self.file_name, mode=mode)

        self.data = [self.my_file.get(self.h5_key + str(i)) for i in xrange(self.N_tot)]