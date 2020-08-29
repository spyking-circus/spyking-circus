import h5py, numpy, re, sys, logging, os
from .hdf5 import H5File
from circus.shared.messages import print_and_log


logger = logging.getLogger(__name__)


class MaxwellFile(H5File):

    description = "maxwell"    
    extension = [".h5", ".hdf5"]
    parallel_write = h5py.get_config().mpi

    _required_fields = {}
    _default_values = {}

    def _generate_probe(self, mapping, nb_channels):
        dirname = os.path.dirname(os.path.abspath(self.file_name))
        probe_name = os.path.join(dirname, 'mapping.prb')
        channels = list(mapping['channel'])
        x_pos = list(mapping['x'])
        y_pos = list(mapping['y'])
        geometry = {}
        for c, x, y in zip(channels, x_pos, y_pos):
            geometry[c] = (x, y)
        f=open(probe_name, 'w')
        to_write = '''
total_nb_channels=%s
radius = 100
channel_groups = {
1: {
    'channels': %s,
    'geometry': {
    %s
    }
    }
}
''' %(nb_channels, channels, geometry)
        f.write(to_write)
        f.close()

    def _read_from_header(self):

        header = {}

        self.params['h5_key'] = 'sig'

        self._open() 
        f = h5py.File(self.file_name, mode='r')
        header['h5_key'] = self.h5_key
        header['sampling_rate'] = 20000

        header['gain'] = 1.0
        header['dtype_offset'] = 'auto'
        header['data_dtype'] = self.my_file.get(header['h5_key']).dtype
        self.compression = self.my_file.get(header['h5_key']).compression

        self._check_compression()

        nb_channels, n_frames = self.my_file.get(header['h5_key']).shape
        self.size = nb_channels * n_frames
        header['nb_channels'] = nb_channels
        self._shape = (n_frames, header['nb_channels'])
        mapping = f['mapping'][:]
        self._generate_probe(mapping, nb_channels)

        self._close()

        return header
