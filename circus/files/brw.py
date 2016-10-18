import h5py, numpy, re, sys
from hdf5 import H5File
from datafile import get_offset

class BRWFile(H5File):

    description = "brw"    
    extension   = [".brw"]
    parallel_write = h5py.get_config().mpi

    def _read_from_header(self):

        header = {}

        self.open() 
        f                       = h5py.File(file_name)
        header['sampling_rate'] = f.get('3BRecInfo/3BRecVars/SamplingRate').value[0]
        header['n_frames']      = f.get('3BRecInfo/3BRecVars/NRecFrames').value[0]
        header['h5_key']        = '3BData/Raw'

        max_volt       = f.get('3BRecInfo/3BRecVars/MaxVolt').value[0]
        min_volt       = f.get('3BRecInfo/3BRecVars/MinVolt').value[0]
        bit_depth      = f.get('3BRecInfo/3BRecVars/BitDepth').value[0]
        inversion      = f.get('3BRecInfo/3BRecVars/SignalInversion').value[0]
        header['gain'] = inversion * ((max_volt - min_volt) / 2**bit_depth)
        header['dtype_offset'] = inversion*min_volt

        header['data_dtype']  = self.my_file.get(self.h5_key).dtype
        header['compression'] = self.my_file.get(self.h5_key).compression

        # HDF5 does not support parallel writes with compression
        if header['compression'] != '':
            self._parallel_write = False
        
        self.size   = self.my_file.get(self.h5_key).shape[0]
        self._shape = (self.n_frames, self.size/self.n_frames)
        self.close()

        return header

    def get_data(self, idx, chunk_size, padding=(0, 0), nodes=None):

        chunk_size  *= self.nb_channels
        padding      = numpy.array(padding) * self.nb_channels

        local_chunk  = self.data[idx*numpy.int64(chunk_size)+padding[0]:(idx+1)*numpy.int64(chunk_size)+padding[1]]
        local_shape  = len(local_chunk)//self.nb_channels
        local_chunk  = local_chunk.reshape(local_shape, self.nb_channels)
        
        if nodes is not None:
            if not numpy.all(nodes == numpy.arange(self.nb_channels)):
                local_chunk = numpy.take(local_chunk, nodes, axis=1)

        return self._scale_data_to_float32(local_chunk)

    def set_data(self, time, data):

        data = self._unscale_data_from_from32(data)
        data = data.ravel()
        self.data[self.nb_channels*time:self.nb_channels*time+len(data)] = data
        