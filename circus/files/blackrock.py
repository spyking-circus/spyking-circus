from .datafile import DataFile, comm
from utils import brpylib
from utils.brpylib import NsxFile, brpylib_ver

class BlackRockFile(DataFile):

    description    = "blackrock"    
    extension      = [".ns5", ".ns6"]
    parallel_write = False
    is_writable    = False

    def _read_from_header(self):
        self._open()
        header                  = {}
        header['sampling_rate'] = float(self.data.basic_header['TimeStampResolution'])        
        header['nb_channels']   = self.data.basic_header['ChannelCount']
        header['gain']          = 1.
        header['data_dtype']    = 'float32'
        header['dtype_offset']  = 0

        self.size   = self.data.getdata('all', 0, 1/header['sampling_rate'])['data_headers'][0]['NumDataPoints']
        self._shape = (self.size, int(header['nb_channels']))

        return header

    def read_chunk(self, idx, chunk_size, padding=(0, 0), nodes=None):

        t_start, t_stop = self._get_t_start_t_stop(idx, chunk_size, padding)
        local_shape     = t_stop - t_start

        t_start /= self.sampling_rate
        local_shape /= self.sampling_rate

        if nodes is None:
            nodes = 'all'
        else:
            nodes = list(nodes)

        self._open()
        local_chunk  = self.data.getdata(nodes, t_start, local_shape)['data'].T
        self._close()

        return local_chunk

    def _open(self, mode='r'):
        self.data = NsxFile(self.file_name)

    def _close(self):
        self.data.close()