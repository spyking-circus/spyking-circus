import h5py, numpy, re, sys, logging
from circus.shared.messages import print_and_log
from datafile import DataFile, comm, get_offset

logger = logging.getLogger(__name__)

class H5File(DataFile):

    description    = "hdf5"    
    extension      = [".h5", ".hdf5"]
    parallel_write = h5py.get_config().mpi
    is_writable    = True

    _required_fields = {'h5_key'        : str,
                        'sampling_rate' : float,
                        'nb_channels'   : int}
    
    _default_values  = {'dtype_offset'  : 'auto', 
                        'gain'          : 1}


    def __check_valid_key__(self, key):
        file       = h5py.File(self.file_name)
        all_fields = []
        file.visit(all_fields.append)    
        if not key in all_fields:
            print_and_log(['The key %s can not be found in the dataset! Keys found are:' %key, 
                         ", ".join(all_fields)], 'error', logger)
            sys.exit(1)
        file.close()

    def _read_from_header(self):

        self.__check_valid_key__(self.h5_key)
        self.open()

        header = {}
        header['data_dtype']   = self.my_file.get(self.h5_key).dtype
        self.compression       = self.my_file.get(self.h5_key).compression

        # HDF5 does not support parallel writes with compression
        if self.compression != '':
            self._parallel_write = False
        
        self.size        = self.my_file.get(self.h5_key).shape
        
        if self.size[0] > self.size[1]:
            self.time_axis = 0
            self._shape = (self.size[0], self.size[1])
        else:
            self.time_axis = 1
            self._shape = (self.size[1], self.size[0])

        self.close()

        return header

    def allocate(self, shape, data_dtype=None):

        if data_dtype is None:
            data_dtype = self.data_dtype

        if self._parallel_write:
            self.my_file = h5py.File(self.file_name, mode='w', driver='mpio', comm=comm)
            self.my_file.create_dataset(self.h5_key, dtype=data_dtype, shape=shape)
        else:
            self.my_file = h5py.File(self.file_name, mode='w')
            if self.is_master:
                if self.compression != '':
                    self.my_file.create_dataset(self.h5_key, dtype=data_dtype, shape=shape, compression=self.compression, chunks=True)
                else:
                    self.my_file.create_dataset(self.h5_key, dtype=data_dtype, shape=shape, chunks=True)

        self.my_file.close()
        self._read_from_header()

    def read_chunk(self, idx, chunk_size, padding=(0, 0), nodes=None):

        if nodes is None:
            if self.time_axis == 0:
                local_chunk = self.data[idx*numpy.int64(chunk_size)+padding[0]:(idx+1)*numpy.int64(chunk_size)+padding[1], :]
            elif self.time_axis == 1:
                local_chunk = self.data[:, idx*numpy.int64(chunk_size)+padding[0]:(idx+1)*numpy.int64(chunk_size)+padding[1]].T
        else:
            if self.time_axis == 0:
                local_chunk = self.data[idx*numpy.int64(chunk_size)+padding[0]:(idx+1)*numpy.int64(chunk_size)+padding[1], nodes]
            elif self.time_axis == 1:
                local_chunk = self.data[nodes, idx*numpy.int64(chunk_size)+padding[0]:(idx+1)*numpy.int64(chunk_size)+padding[1]].T

        return self._scale_data_to_float32(local_chunk)

    def write_chunk(self, time, data):

        data = self._unscale_data_from_from32(data)
        
        if self.time_axis == 0:
            self.data[time:time+data.shape[0], :] = data
        elif self.time_axis == 1:
            self.data[:, time:time+data.shape[0]] = data.T

    def open(self, mode='r'):
        if mode in ['r+', 'w'] and self._parallel_write:
            self.my_file = h5py.File(self.file_name, mode=mode, driver='mpio', comm=comm)
        else:
            self.my_file = h5py.File(self.file_name, mode=mode)

        self.data = self.my_file.get(self.h5_key)
        
    def close(self):
        self.my_file.close()
        del self.data

    @property
    def h5_key(self):
        return self._params['h5_key']