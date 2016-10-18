import h5py, numpy, re, sys
from hdf5 import H5File
from datafile import get_offset

class ARFFile(H5File):

    description    = "arf"    
    extension      = [".arf", ".hdf5", ".h5"]
    parallel_write = h5py.get_config().mpi
    is_writable    = True
    is_streamable  = True

    _required_fields = {'h5_key'        : str,
                        'channel_name'  : str}
    
    _default_values  = {'dtype_offset'  : 'auto', 
                        'gain'          : 1}

    def _get_sorted_channels_(self, all_keys, pattern):
        sub_list     = [f for f in all_keys if pattern in f]
        all_channels = [int(f.split(pattern)[1]) for f in sub_list]
        idx          = numpy.argsort(all_channels)
        return sub_list, idx

    def _get_channel_key_(self, i):
        return self.h5_key + '/' + self.channels[int(i)]

    @property
    def channel_name(self):
        return self._params['channel_name']


    def set_streams(self):
        
        params  = self.get_description()
        streams = []
        for i in source:
            params['channel_name'] = myname
            streams += [ARFFile(to_process, params)]
            
        return streams

    def _read_from_header(self):

        header = {}
        
        self.__check_valid_key__(self.h5_key)
        self.open()

        all_keys                = self.my_file.get(self.h5_key).keys()
        channels, idx           = self._get_sorted_channels_(all_keys, self.channel_name)    
        self.channels           = channels
        key                     = self.h5_key + '/' + self.channels[0]
        header['sampling_rate'] = dict(h5py.File(file_name).get(key).attrs.items())['sampling_rate']
        header['data_dtype']    = self.my_file.get(self._get_channel_key_(0)).dtype
        self.compression        = self.my_file.get(self._get_channel_key_(0)).compression

        # HDF5 does not support parallel writes with compression
        if self.compression != '':
            self._parallel_write = False
        
        self.size   = self.my_file.get(self._get_channel_key_(0)).shape
        self._shape = (self.size[0], len(self.channels))
        self.close()

        return header


    def allocate(self, shape, data_dtype=None):

        if data_dtype is None:
            data_dtype = self.data_dtype

        if self._parallel_write:
            self.my_file = h5py.File(self.file_name, mode='w', driver='mpio', comm=comm)
            self.my_file.create_group(self.h5_key)
            for i in xrange(self.nb_channels):
                self.my_file.create_dataset(self._get_channel_key_(i), dtype=data_dtype, shape=shape)
        else:
            self.my_file = h5py.File(self.file_name, mode='w')
            if self.is_master:
                if self.compression != '':
                    self.my_file.create_dataset(self.h5_key, dtype=data_dtype, shape=shape, compression=self.compression, chunks=True)
                    for i in xrange(self.nb_channels):
                        self.my_file.create_dataset(self._get_channel_key_(i), dtype=data_dtype, compression=self.compression, shape=shape)
                else:
                    for i in xrange(self.nb_channels):
                        self.my_file.create_dataset(self._get_channel_key_(i), dtype=data_dtype, shape=shape, chunks=True)

        self.my_file.close()
        self._read_from_header()

    def read_chunk(self, idx, chunk_size, padding=(0, 0), nodes=None):

        t_start     = idx*numpy.int64(chunk_size)+padding[0]
        t_stop      = (idx+1)*numpy.int64(chunk_size)+padding[1]
        local_shape = t_stop - t_start

        if (t_start + local_shape) > self.duration:
            local_shape = self.duration - t_start

        if nodes is None:
            nodes = numpy.arange(self.nb_channels)

        local_chunk = numpy.zeros((local_shape, len(nodes)), dtype=self.data_dtype)

        for count, i in enumerate(nodes):
            local_chunk[:, count] = self.data[i][t_start:t_stop]
        
        return self._scale_data_to_float32(local_chunk)

    def write_chunk(self, time, data):
        
        data  = data.astype(self.data_dtype)
        for i in xrange(self.nb_channels):
            self.data[i][time:time+data.shape[0]] = self._unscale_data_from_from32(data[:, i])

    def open(self, mode='r'):
        if mode in ['r+', 'w'] and self._parallel_write:
            self.my_file = h5py.File(self.file_name, mode=mode, driver='mpio', comm=comm)
        else:
            self.my_file = h5py.File(self.file_name, mode=mode)

        self.data = [self.my_file.get(self._get_channel_key_(i)) for i in xrange(self.nb_channels)]