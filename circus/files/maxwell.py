import h5py, numpy, re, sys, logging, os
from .hdf5 import H5File
from circus.shared.messages import print_and_log

logger = logging.getLogger(__name__)

class MaxwellFile(H5File):

    description = "maxwell"    
    extension = [".h5", ".hdf5"]
    parallel_write = h5py.get_config().mpi

    _required_fields = {}
    _default_values = {'well': '0000'}

    def _read_from_header(self):

        header = {}

        self.my_file = h5py.File(self.file_name, mode='r')
        header['version'] = self.my_file['version'][0]

        if header['version'] == b'20160704':
            h5_key = 'sig'
            header['h5_key'] = 'sig'
            header['sampling_rate'] = 20000
            header['dtype_offset'] = 512
            header['gain'] = self.my_file.get('settings/lsb')[0]
        elif header['version'] == b'20190530':
            if not 'data%s' %self.params['well'] in list(self.my_file['data_store'].keys()):
                print_and_log(['Well %s not found!' %self.params['well']], 'error', logger)
                sys.exit(0)
            header['h5_key'] = '/data_store/data%s/groups/routed/raw' %self.params['well']
            header['sampling_rate'] = self.my_file.get('/data_store/data%s/settings/sampling' %self.params['well'])[0]
            header['dtype_offset'] = 512
            header['gain'] = self.my_file.get('/data_store/data%s/settings/lsb' %self.params['well'])[0]
        
        header['data_dtype'] = self.my_file.get(header['h5_key']).dtype
        self.compression = self.my_file.get(header['h5_key']).compression

        self._check_compression()

        nb_channels, n_frames = self.my_file.get(header['h5_key']).shape
        self.size = nb_channels * n_frames
        header['nb_channels'] = nb_channels
        self._shape = (n_frames, header['nb_channels'])
        self.my_file.close()
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