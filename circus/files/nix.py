import numpy
import re
import sys
import logging
from .hdf5 import H5File
from circus.shared.messages import print_and_log

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py


logger = logging.getLogger(__name__)


class NixFile(H5File):

    description = "nix"
    extension = [".nix", ".h5", ".hdf5"]
    parallel_write = h5py.get_config().mpi
    is_writable = True

    _required_fields = {
        'block': str,
        'data_array': str
    }

    _default_values = {
        'dtype_offset': 'auto',
        'gain': 1.0
    }

    def _read_from_header(self):

        nix_name = 'data/%s/data_arrays/%s' % (self.params['block'], self.params['data_array'])
        self.params['h5_key'] = '%s/data' % nix_name

        self.__check_valid_key__(self.h5_key)
        self._open()

        header = {}
        header['data_dtype'] = self.my_file.get(self.h5_key).dtype

        for key in self.my_file.get('%s/dimensions' % nix_name).keys():
            tmp = dict(self.my_file.get('%s/dimensions/%s' % (nix_name, key)).attrs.items())
            if tmp['label'] == 'time':
                header['sampling_rate'] = 1./['sampling_interval']

        self.compression = self.my_file.get(self.h5_key).compression
        self._check_compression()

        self.size = self.my_file.get(self.h5_key).shape

        if self.size[0] > self.size[1]:
            self.time_axis = 0
            self._shape = (self.size[0], self.size[1])
        else:
            self.time_axis = 1
            self._shape = (self.size[1], self.size[0])

        header['nb_channels'] = self._shape[1]
        self._close()

        return header
