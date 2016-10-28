import h5py, numpy, re, sys
from hdf5 import H5File

class BRWFile(H5File):

    description = "brw"    
    extension   = [".brw"]
    parallel_write = h5py.get_config().mpi

    _required_fields = {}
    _default_values  = {}

    def _read_from_header(self):

        header = {}

        self._params['h5_key']  = '3BData/Raw'
        header['h5_key']        = self.h5_key


        self._open() 
        f                       = h5py.File(self.file_name)
        header['sampling_rate'] = f.get('3BRecInfo/3BRecVars/SamplingRate').value[0]
        

        max_volt       = f.get('3BRecInfo/3BRecVars/MaxVolt').value[0]
        min_volt       = f.get('3BRecInfo/3BRecVars/MinVolt').value[0]
        bit_depth      = f.get('3BRecInfo/3BRecVars/BitDepth').value[0]
        inversion      = f.get('3BRecInfo/3BRecVars/SignalInversion').value[0]
        header['gain'] = inversion * ((max_volt - min_volt) / 2**bit_depth)
        header['dtype_offset'] = inversion*min_volt
        header['data_dtype']   = self.my_file.get(header['h5_key']).dtype
        self.compression       = self.my_file.get(header['h5_key']).compression

        # HDF5 does not support parallel writes with compression
        if self.compression != '':
            self._parallel_write = False
        
        n_frames    = f.get('3BRecInfo/3BRecVars/NRecFrames').value[0]
        self.size   = self.my_file.get(header['h5_key']).shape[0]
        header['nb_channels']  = self.size/n_frames
        
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

        data = self._unscale_data_from_from32(data)
        data = data.ravel()
        self.data[self.nb_channels*time:self.nb_channels*time+len(data)] = data
        