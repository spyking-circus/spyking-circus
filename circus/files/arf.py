import h5py, numpy, re, sys
from circus.shared.messages import print_error, print_and_log
from hdf5 import H5File
from datafile import _check_requierements_

class ARFFile(H5File):

    _description    = "arf"    
    _extension      = [".arf", ".hdf5", ".h5"]
    _parallel_write = h5py.get_config().mpi
    _is_writable    = True

    _requiered_fields = {'h5_key'        : ['string', None], 
                         'channel_name'  : ['string', None], 
                         'sampling_rate' : ['float' , None]}

    def __init__(self, file_name, params, empty=False, **kwargs):

        kwargs = _check_requierements_(self._description, self._requiered_fields, params, **kwargs)

        if not empty:
            self.__check_valid_key__(file_name, kwargs['h5_key'])
            all_keys = h5py.File(file_name).get(kwargs['h5_key']).keys()
            channels, idx = self._get_sorted_channels_(all_keys, kwargs['channel_name'])    
            kwargs['N_tot'] = len(channels)
            kwargs['channels'] = channels
            kwargs['indices']  = idx

        H5File.__init__(self, file_name, params, empty, **kwargs)

    def _get_sorted_channels_(self, all_keys, pattern):
        sub_list     = [f for f in all_keys if pattern in f]
        all_channels = [int(f.split(pattern)[1]) for f in sub_list]
        idx          = numpy.argsort(all_channels)
        return sub_list, idx

    def _get_channel_key_(self, i):
        return self.h5_key + '/' + self.channels[i]

    def _get_info_(self):

        self.empty = False
        self.open()

        self.data_dtype  = self.my_file.get(self._get_channel_key_(0)).dtype
        self.compression = self.my_file.get(self._get_channel_key_(0)).compression

        # HDF5 does not support parallel writes with compression
        if self.compression != '':
        	self._parallel_write = False
        
        self.size   = self.my_file.get(self._get_channel_key_(0)).shape
        self._shape = (self.size[0], self.N_tot)
        
        self._max_offset = self._shape[0]
        self.data_offset = 0
        self.close()

    def allocate(self, shape, data_dtype=None):

        if data_dtype is None:
            data_dtype = self.data_dtype

        if self._parallel_write:
            self.my_file = h5py.File(self.file_name, mode='w', driver='mpio', comm=comm)
            self.my_file.create_group(self.h5_key)
            for i in xrange(self.N_tot):
                self.my_file.create_dataset(self._get_channel_key_(i), dtype=data_dtype, shape=shape)
        else:
            self.my_file = h5py.File(self.file_name, mode='w')
            if self.is_master:
                if self.compression != '':
                    self.my_file.create_dataset(self.h5_key, dtype=data_dtype, shape=shape, compression=self.compression, chunks=True)
                    for i in xrange(self.N_tot):
                        self.my_file.create_dataset(self._get_channel_key_(i), dtype=data_dtype, compression=self.compression, shape=shape)
                else:
                    for i in xrange(self.N_tot):
                        self.my_file.create_dataset(self._get_channel_key_(i), dtype=data_dtype, shape=shape, chunks=True)

        self.my_file.close()
        self._get_info_()

    def get_data(self, idx, chunk_size=None, padding=(0, 0), nodes=None):

        chunk_size  = self._get_chunk_size_(chunk_size)
        t_start     = idx*numpy.int64(chunk_size)+padding[0]
        t_stop      = (idx+1)*numpy.int64(chunk_size)+padding[1]
        local_shape = t_stop - t_start

        if (t_start + local_shape) > self.max_offset:
            local_shape = self.max_offset - t_start

        if nodes is None:
            nodes = numpy.arange(self.N_tot)

        local_chunk = numpy.zeros((local_shape, len(nodes)), dtype=self.data_dtype)

        for count, i in enumerate(nodes):
            local_chunk[:, count] = self.data[i][t_start:t_stop]
        
        local_chunk  = local_chunk.astype(numpy.float32)

        return numpy.ascontiguousarray(local_chunk)

    def set_data(self, time, data):
        
    	data  = data.astype(self.data_dtype)
        for i in xrange(self.N_tot):
            self.data[i][time:time+data.shape[0]] = data[:, i]

    def open(self, mode='r'):
        if self._parallel_write:
            self.my_file = h5py.File(self.file_name, mode=mode, driver='mpio', comm=comm)
        else:
            self.my_file = h5py.File(self.file_name, mode=mode)

        self.data = [self.my_file.get(self._get_channel_key_(i)) for i in xrange(self.N_tot)]