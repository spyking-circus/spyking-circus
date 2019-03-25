import h5py, numpy, re, sys, logging
from .hdf5 import H5File
from circus.shared.messages import print_and_log

logger = logging.getLogger(__name__)

class BRWFile(H5File):

    description = "brw"    
    extension   = [".brw"]
    parallel_write = h5py.get_config().mpi

    _required_fields = {}
    _default_values  = {}

    def _read_from_header(self):

        header = {}

        self.params['h5_key']  = '3BData/Raw'
        header['h5_key']        = self.h5_key


        self._open() 
        f                       = h5py.File(self.file_name, mode='r')
        header['sampling_rate'] = f['3BRecInfo/3BRecVars/SamplingRate'][0]
        

        max_volt       = f['3BRecInfo/3BRecVars/MaxVolt'][0]
        min_volt       = f['3BRecInfo/3BRecVars/MinVolt'][0]
        bit_depth      = f['3BRecInfo/3BRecVars/BitDepth'][0]
        inversion      = f['3BRecInfo/3BRecVars/SignalInversion'][0]
        header['gain'] = inversion * ((max_volt - min_volt) / 2**bit_depth)
        header['dtype_offset'] = -inversion*min_volt
        header['data_dtype']   = self.my_file.get(header['h5_key']).dtype
        self.compression       = self.my_file.get(header['h5_key']).compression

        self._check_compression()
        
        n_frames    = f['3BRecInfo/3BRecVars/NRecFrames'][0]
        self.size   = self.my_file.get(header['h5_key']).shape[0]
        header['nb_channels']  = int(self.size/n_frames)
        
        self._shape = (n_frames, header['nb_channels'])
        
        self._close()

        return header

    def read_chunk(self, idx, chunk_size, padding=(0, 0), nodes=None):

        t_start, t_stop = self._get_t_start_t_stop(idx, chunk_size, padding)
        local_shape     = t_stop - t_start

        local_chunk  = self.data[t_start*self.nb_channels:t_stop*self.nb_channels]
        local_chunk  = local_chunk.reshape(local_shape, self.nb_channels)
        
        if nodes is not None:
            if not numpy.all(nodes == numpy.arange(self.nb_channels)):
                local_chunk = numpy.take(local_chunk, nodes, axis=1)

        return self._scale_data_to_float32(local_chunk)

    def write_chunk(self, time, data):

        data = self._unscale_data_from_float32(data)
        data = data.ravel()
        self.data[self.nb_channels*time:self.nb_channels*time+len(data)] = data
        