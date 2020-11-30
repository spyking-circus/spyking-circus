import numpy
import re
import sys
import logging
from circus.shared.messages import print_and_log
from .datafile import DataFile, comm

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

logger = logging.getLogger(__name__)


class H5File(DataFile):

    description = "hdf5"
    extension = [".h5", ".hdf5"]
    parallel_write = h5py.get_config().mpi
    is_writable = True

    _required_fields = {
        'h5_key': str,
        'sampling_rate': float
    }

    _default_values = {
        'dtype_offset': 'auto',
        'gain': 1.0
    }

    def _check_compression(self):
        # HDF5 does not support parallel writes with compression
        if self.compression != '':
            self.parallel_write = False
            if self.is_master:
                print_and_log(['Data are compressed thus parallel writing is disabled'], 'debug', logger)

    def __check_valid_key__(self, key):
        file = h5py.File(self.file_name, mode='r')
        all_fields = []
        file.visit(all_fields.append)    
        if key not in all_fields:
            print_and_log([
                "The key %s can not be found in the dataset! Keys found are:" % key,
                ", ".join(all_fields)
            ], 'error', logger)
            sys.exit(1)
        file.close()

    def _read_from_header(self):

        self.__check_valid_key__(self.h5_key)
        self._open()

        header = {}
        header['data_dtype'] = self.my_file.get(self.h5_key).dtype
        self.compression = self.my_file.get(self.h5_key).compression
        self.grid_ids = False
        self._check_compression()

        self.size = self.my_file.get(self.h5_key).shape

        if len(self.size) == 2:
            if self.size[0] > self.size[1]:
                self.time_axis = 0
                self._shape = (self.size[0], self.size[1])
            else:
                self.time_axis = 1
                self._shape = (self.size[1], self.size[0])
            header['nb_channels'] = self._shape[1]
        elif len(self.size) == 3:
            self.grid_ids = True
            if self.size[0] > self.size[-1]:
                self.time_axis = 0
                self._shape = (self.size[0], self.size[1], self.size[2])
            else:
                self.time_axis = 1
                self._shape = (self.size[2], self.size[1], self.size[0])
            header['nb_channels'] = self._shape[1] * self._shape[2]
        self._close()

        if self.time_axis == 0:
            if not self.grid_ids:
                self._read_chunk = self._read_time_0_no_grid
            else:
                self._read_chunk = self._read_time_0_grid
        elif self.time_axis == 1:
            if not self.grid_ids:
                self._read_chunk = self._read_time_1_no_grid
            else:
                self._read_chunk = self._read_time_1_grid

        return header

    def _read_time_0_no_grid(self, do_slice, t_start, t_stop, nodes):
        if do_slice:
            is_sorted = numpy.all(numpy.diff(nodes) > 0)
            if not is_sorted:
                mapping = numpy.argsort(nodes)
                sorted_nodes = numpy.sort(nodes)
            if not is_sorted:
                local_chunk = self.data[t_start:t_stop, sorted_nodes]
                local_chunk = local_chunk[:, mapping]
            else:
                local_chunk = self.data[t_start:t_stop, nodes]
        else:
            local_chunk = self.data[t_start:t_stop, :]
        return local_chunk

    def _read_time_1_no_grid(self, do_slice, t_start, t_stop, nodes):
        if do_slice:
            is_sorted = numpy.all(numpy.diff(nodes) > 0)
            if not is_sorted:
                mapping = numpy.argsort(nodes)
            sorted_nodes = numpy.sort(nodes)
            if not is_sorted:
                local_chunk = self.data[sorted_nodes, t_start:t_stop].T
                local_chunk = local_chunk[:, mapping]
            else:
                local_chunk = self.data[nodes, t_start:t_stop].T
        else:
            local_chunk = self.data[:, t_start:t_stop].T
        return local_chunk

    def _read_time_0_grid(self, do_slice, t_start, t_stop, nodes):
        local_chunk = self.data[t_start:t_stop, :, :].reshape(t_stop-t_start, self.nb_channels)
        if do_slice:
            local_chunk = numpy.take(local_chunk, nodes, axis=1)
        return local_chunk

    def _read_time_1_grid(self, do_slice, t_start, t_stop, nodes):
        local_chunk = self.data[:, :, t_start:t_stop].reshape(self.nb_channels, t_stop-t_start).T
        if do_slice:
            local_chunk = numpy.take(local_chunk, nodes, axis=1)
        return local_chunk

    def read_chunk(self, idx, chunk_size, padding=(0, 0), nodes=None):

        t_start, t_stop = self._get_t_start_t_stop(idx, chunk_size, padding)
        do_slice = nodes is not None and not numpy.all(nodes == numpy.arange(self.nb_channels))
        local_chunk = self._read_chunk(do_slice, t_start, t_stop, nodes)

        return self._scale_data_to_float32(local_chunk)

    def write_chunk(self, time, data):

        data = self._unscale_data_from_float32(data)
        
        if self.time_axis == 0:
            if not self.grid_ids:
                self.data[time:time+data.shape[0], :] = data
            else:
                self.data[time:time+data.shape[0], :, :] = data.reshape(data.shape[0], self._shape[1], self._shape[2])
        elif self.time_axis == 1:
            if not self.grid_ids:
                self.data[:, time:time+data.shape[0]] = data.T
            else:
                self.data[:, :, time:time+data.shape[0]] = data.reshape(data.shape[0], self._shape[1], self._shape[2]).T

    def _open(self, mode='r'):
        if mode in ['r+', 'w'] and self.parallel_write:
            self.my_file = h5py.File(self.file_name, mode=mode, driver='mpio', comm=comm)
        else:
            self.my_file = h5py.File(self.file_name, mode=mode)

        self.data = self.my_file.get(self.h5_key)
        
    def _close(self):
        self.my_file.close()
        try:
            del self.data
        except Exception:
            pass

    @property
    def h5_key(self):
        return self.params['h5_key']
