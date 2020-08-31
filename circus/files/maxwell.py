import h5py, numpy, re, sys, logging, os
from .hdf5 import H5File
from circus.shared.messages import print_and_log

logger = logging.getLogger(__name__)

class MaxwellFile(H5File):

    description = "maxwell"    
    extension = [".h5", ".hdf5"]
    parallel_write = h5py.get_config().mpi

    _required_fields = {}
    _default_values = {'h5_key': ''}

    def _read_from_header(self):

        header = {}

        my_file = h5py.File(self.file_name, mode='r')
        header['version'] = my_file['version'][0]

        if header['version'] == b'20160704':
            h5_key = 'sig'
            header['h5_key'] = 'sig'
            header['sampling_rate'] = 20000
            header['dtype_offset'] = 'auto'
        else:
            pass
            header['sampling_rate'] = 10000
            header['dtype_offset'] = 512
        
        header['gain'] = my_file.get('settings/lsb')[0]
        header['data_dtype'] = my_file.get(header['h5_key']).dtype
        self.compression = my_file.get(header['h5_key']).compression

        self._check_compression()

        nb_channels, n_frames = my_file.get(header['h5_key']).shape
        self.size = nb_channels * n_frames
        header['nb_channels'] = nb_channels
        self._shape = (n_frames, header['nb_channels'])
        my_file.close()
        return header

    def _read_chunk(self, do_slice, t_start, t_stop, nodes):
        if do_slice:
            local_chunk = self.data[nodes, t_start:t_stop].T
        else:
            local_chunk = self.data[:, t_start:t_stop].T
        return local_chunk

    def write_chunk(self, time, data):
        data = self._unscale_data_from_float32(data)
        self.data[:, time:time+data.shape[0]] = data.T