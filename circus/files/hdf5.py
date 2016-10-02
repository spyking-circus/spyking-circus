import h5py, numpy, re, sys
from circus.shared.messages import print_error
from datafile import DataFile, comm, get_offset

class H5File(DataFile):

    _description    = "hdf5"    
    _extension      = [".h5", ".hdf5"]
    _parallel_write = h5py.get_config().mpi
    _is_writable    = True

    _requiered_fields = {'h5_key'        : ['string', None], 
                         'sampling_rate' : ['float' , None],
                         'gain'          : ['float', 1.],
                         'dtype_offset'  : ['string', 'auto']}

    def __init__(self, file_name, is_empty=False, **kwargs):

        DataFile.__init__(self, file_name, is_empty, **kwargs)

    def __check_valid_key__(self, file_name, key):
        file       = h5py.File(file_name)
        all_fields = []
        file.visit(all_fields.append)    
        if not key in all_fields:
            print_error(['The key %s can not be found in the dataset! Keys found are:' %key, 
                         ", ".join(all_fields)])
            sys.exit(0)
        file.close()

    def _get_info_(self):

        self.__check_valid_key__(self.file_name, self.h5_key)
        self.open()
        self.data_dtype   = self.my_file.get(self.h5_key).dtype
        self.dtype_offset = get_offset(self.data_dtype, self.dtype_offset)
        self.compression  = self.my_file.get(self.h5_key).compression

        # HDF5 does not support parallel writes with compression
        if self.compression != '':
        	self._parallel_write = False
        
        self.size        = self.my_file.get(self.h5_key).shape
        
        if self.size[0] > self.size[1]:
            self.time_axis = 0
            self._shape = (self.size[0], self.size[1])
        else:
            self.time_axis = 1
            self._shape = self.size

        self.close()

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
        self._get_info_()

    def get_data(self, idx, chunk_size, padding=(0, 0), nodes=None):

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

    def set_data(self, time, data):

        data = self._unscale_data_from_from32(data)
        
        if self.time_axis == 0:
            self.data[time:time+data.shape[0], :] = data
        elif self.time_axis == 1:
            self.data[:, time:time+data.shape[0]] = data.T

    def open(self, mode='r'):
        if self._parallel_write:
            self.my_file = h5py.File(self.file_name, mode=mode, driver='mpio', comm=comm)
        else:
            self.my_file = h5py.File(self.file_name, mode=mode)

        self.data = self.my_file.get(self.h5_key)
        
    def close(self):
        self.my_file.close()