import h5py, numpy, re, sys
from circus.shared.messages import print_error
from hdf5 import H5File
from datafile import get_offset

class BRWFile(H5File):

    _description = "brw"    
    _extension   = [".brw"]
    _parallel_write = h5py.get_config().mpi

    _requiered_fields = {}


    def __init__(self, file_name, is_empty=False, **kwargs):

        
        if not is_empty:
            f = h5py.File(file_name)
            kwargs['sampling_rate'] = f.get('3BRecInfo/3BRecVars/SamplingRate').value[0]
            kwargs['n_frames']      = f.get('3BRecInfo/3BRecVars/NRecFrames').value[0]
            kwargs['h5_key']        = '3BData/Raw'

            max_volt       = f.get('3BRecInfo/3BRecVars/MaxVolt').value[0]
            min_volt       = f.get('3BRecInfo/3BRecVars/MinVolt').value[0]
            bit_depth      = f.get('3BRecInfo/3BRecVars/BitDepth').value[0]
            inversion      = f.get('3BRecInfo/3BRecVars/SignalInversion').value[0]
            kwargs['gain'] = inversion * ((max_volt - min_volt) / 2**bit_depth)
            kwargs['dtype_offset'] = inversion*min_volt

        H5File.__init__(self, file_name, is_empty, **kwargs)


    def _get_info_(self):

        self.open() 
        self.data_dtype   = self.my_file.get(self.h5_key).dtype
        self.dtype_offset = get_offset(self.data_dtype, self.dtype_offset)
        self.compression  = self.my_file.get(self.h5_key).compression

        # HDF5 does not support parallel writes with compression
        if self.compression != '':
            self._parallel_write = False
        
        self.size   = self.my_file.get(self.h5_key).shape[0]
        self._shape = (self.n_frames, self.size/self.n_frames)
        self.close()

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
        