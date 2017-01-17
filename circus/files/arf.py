import h5py, numpy, re, sys, logging
from hdf5 import H5File
from circus.shared.messages import print_and_log

logger = logging.getLogger(__name__)

class ARFFile(H5File):

    description    = "arf"    
    extension      = [".arf", ".hdf5", ".h5"]
    parallel_write = h5py.get_config().mpi
    is_writable    = True
    is_streamable  = ['multi-files', 'single-file']

    _required_fields = {'h5_key'        : str,
                        'channel_name'  : str}
    
    _default_values  = {'dtype_offset'  : 'auto', 
                        'gain'          : 1.}

    def _get_sorted_channels_(self, all_keys, pattern):
        sub_list     = [f for f in all_keys if pattern in f]
        all_channels = [int(f.split(pattern)[1]) for f in sub_list]
        idx          = numpy.argsort(all_channels)
        return sub_list, idx

    def _get_channel_key_(self, i):
        return self.h5_key + '/' + self.channels[self.indices[i]]

    @property
    def channel_name(self):
        return self._params['channel_name']


    def set_streams(self, stream_mode):
        
        if stream_mode == 'single-file':
            
            sources     = []
            to_write    = []
            count       = 0
            params      = self.get_description()
            my_file     = h5py.File(self.file_name)
            all_matches = [re.findall('\d+', u) for u in my_file.keys()]
            all_streams = []
            for m in all_matches:
                if len(m) > 0:
                    all_streams += [int(m[0])]

            idx = numpy.argsort(all_streams)

            for i in xrange(len(all_streams)):
                params['h5_key']  = my_file.keys()[idx[i]]
                new_data          = type(self)(self.file_name, params)
                sources          += [new_data]
                to_write         += ['We found the datafile %s with t_start %d and duration %d' %(new_data.file_name, new_data.t_start, new_data.duration)]

            print_and_log(to_write, 'debug', logger)

            return sources

        elif stream_mode == 'multi-files':
            return H5File.set_streams(stream_mode)

    def _read_from_header(self):

        header = {}
        
        self.__check_valid_key__(self.h5_key)
        
        self.my_file            = h5py.File(self.file_name)
        all_keys                = self.my_file.get(self.h5_key).keys()
        channels, idx           = self._get_sorted_channels_(all_keys, self.channel_name)    
        self.channels           = channels
        self.indices            = idx
        key                     = self.h5_key + '/' + self.channels[0]
        header['sampling_rate'] = dict(self.my_file.get(key).attrs.items())['sampling_rate']
        header['data_dtype']    = self.my_file.get(self._get_channel_key_(0)).dtype
        header['nb_channels']   = len(self.channels)
        self.compression        = self.my_file.get(self._get_channel_key_(0)).compression
        self._t_start           = dict(self.my_file.get(self.h5_key).attrs.items())['timestamp'][0]
        
        # HDF5 does not support parallel writes with compression
        if self.compression != '':
            self._parallel_write = False
        
        self.size     = self.my_file.get(self._get_channel_key_(0)).shape
        self._shape   = (self.size[0], header['nb_channels'])
        self.my_file.close()

        return header

    def read_chunk(self, idx, chunk_size, padding=(0, 0), nodes=None):

        t_start, t_stop = self._get_t_start_t_stop(idx, chunk_size, padding)
        local_shape     = t_stop - t_start

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

    def _open(self, mode='r'):
        if mode in ['r+', 'w'] and self._parallel_write:
            self.my_file = h5py.File(self.file_name, mode=mode, driver='mpio', comm=comm)
        else:
            self.my_file = h5py.File(self.file_name, mode=mode)

        self.data = [self.my_file.get(self._get_channel_key_(i)) for i in xrange(self.nb_channels)]