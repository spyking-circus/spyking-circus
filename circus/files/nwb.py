import h5py, numpy, re, sys
from hdf5 import H5File

class NWBFile(H5File):

    description    = "nwb"    
    extension      = [".nwb", ".h5", ".hdf5"]
    parallel_write = h5py.get_config().mpi
    is_writable    = True

