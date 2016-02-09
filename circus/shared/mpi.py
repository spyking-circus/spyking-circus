import numpy, pylab, os, mpi4py, progressbar, tempfile
from mpi4py import MPI
comm = MPI.COMM_WORLD


def gather_array(data, mpi_comm, root=0, shape=0, dtype='float32'):
    # gather 1D or 2D numpy arrays
    assert isinstance(data, numpy.ndarray)
    assert len(data.shape) < 3
    # first we pass the data size
    size  = data.size
    sizes = mpi_comm.gather(size, root=root) or []
    # now we pass the data
    displacements = [int(sum(sizes[:i])) for i in range(len(sizes))]
    if dtype is 'float32':
        gdata         = numpy.empty(int(sum(sizes)), dtype=numpy.float32)
        mpi_comm.Gatherv([data.flatten(), size, MPI.FLOAT], [gdata, (sizes, displacements), MPI.FLOAT], root=root)
    elif dtype is 'int32':
        gdata         = numpy.empty(int(sum(sizes)), dtype=numpy.int32)
        mpi_comm.Gatherv([data.flatten(), size, MPI.INT], [gdata, (sizes, displacements), MPI.INT], root=root)        
    if len(data.shape) == 1:
        return gdata
    else:
        if shape == 0:
            num_lines = data.shape[0]
            if num_lines > 0:
                return gdata.reshape((num_lines, gdata.size/num_lines))
            else: 
                return gdata.reshape((0, gdata.shape[1]))
        if shape == 1:
            num_columns = data.shape[1]
            if num_columns > 0:
                return gdata.reshape((gdata.size/num_columns, num_columns))
            else:
                return gdata.reshape((gdata.shape[0], 0))

def all_gather_array(data, mpi_comm, shape=0, dtype='float32'):
    # gather 1D or 2D numpy arrays
    assert isinstance(data, numpy.ndarray)
    assert len(data.shape) < 3
    # first we pass the data size
    size  = data.size
    sizes = mpi_comm.allgather(size) or []
    # now we pass the data
    displacements = [int(sum(sizes[:i])) for i in range(len(sizes))]
    if dtype is 'float32':
        gdata         = numpy.empty(int(sum(sizes)), dtype=numpy.float32)
        mpi_comm.Allgatherv([data.flatten(), size, MPI.FLOAT], [gdata, (sizes, displacements), MPI.FLOAT])
    elif dtype is 'int32':
        gdata         = numpy.empty(int(sum(sizes)), dtype=numpy.int32)
        mpi_comm.Allgatherv([data.flatten(), size, MPI.INT], [gdata, (sizes, displacements), MPI.INT])        
    if len(data.shape) == 1:
        return gdata
    else:
        if shape == 0:
            num_lines = data.shape[0]
            if num_lines > 0:
                return gdata.reshape((num_lines, gdata.size/num_lines))
            else: 
                return gdata.reshape((0, gdata.shape[1]))
        if shape == 1:
            num_columns = data.shape[1]
            if num_columns > 0:
                return gdata.reshape((gdata.size/num_columns, num_columns))
            else:
                return gdata.reshape((gdata.shape[0], 0))

def get_mpi_type(data_type):
    if data_type == 'int16':
        return MPI.SHORT
    elif data_type == 'uint16':
        return MPI.UNSIGNED_SHORT
    elif data_type == 'float32':
        return MPI.FLOAT