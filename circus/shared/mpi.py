import numpy, os, mpi4py, logging
from mpi4py import MPI
from messages import print_and_log
comm = MPI.COMM_WORLD

try:
    MPI.Win.Allocate_shared(1, 1, MPI.INFO_NULL, MPI.COMM_SELF).Free()
    SHARED_MEMORY = True
except NotImplementedError:
    SHARED_MEMORY = False


logger = logging.getLogger(__name__)

def gather_mpi_arguments(hostfile, params):
    from mpi4py import MPI
    vendor = MPI.get_vendor()
    print_and_log(['MPI detected: %s' % str(vendor)], 'debug', logger)
    if vendor[0] == 'Open MPI':
        mpi_args = ['mpirun']
        if os.getenv('LD_LIBRARY_PATH'):
            mpi_args += ['-x', 'LD_LIBRARY_PATH']
        if os.getenv('PATH'):
            mpi_args += ['-x', 'PATH']
        if os.getenv('PYTHONPATH'):
            mpi_args += ['-x', 'PYTHONPATH']
        if os.path.exists(hostfile):
            mpi_args += ['-hostfile', hostfile]
    elif vendor[0] == 'Microsoft MPI':
        mpi_args = ['mpiexec']
        if os.path.exists(hostfile):
            mpi_args += ['-machinefile', hostfile]
    elif vendor[0] == 'MPICH2':
        mpi_args = ['mpiexec']
        if os.path.exists(hostfile):
            mpi_args += ['-f', hostfile]
    elif vendor[0] == 'MPICH':
        mpi_args = ['mpiexec']
        if os.path.exists(hostfile):
            mpi_args += ['-f', hostfile]
    else:
        print_and_log([
                        '%s may not be yet properly implemented: contact developpers' %
                        vendor[0]], 'error', logger)
        mpi_args = ['mpirun']
        if os.path.exists(hostfile):
            mpi_args += ['-hostfile', hostfile]
    return mpi_args

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
    elif dtype is 'float64':
        gdata         = numpy.empty(int(sum(sizes)), dtype=numpy.float64)
        mpi_comm.Gatherv([data.flatten(), size, MPI.DOUBLE], [gdata, (sizes, displacements), MPI.DOUBLE], root=root)
    elif dtype is 'int32':
        gdata         = numpy.empty(int(sum(sizes)), dtype=numpy.int32)
        mpi_comm.Gatherv([data.flatten(), size, MPI.INT], [gdata, (sizes, displacements), MPI.INT], root=root)
    elif dtype is 'int64':
        gdata = numpy.empty(int(sum(sizes)), dtype=numpy.int64)
        mpi_comm.Gatherv([data.flatten(), size, MPI.LONG], [gdata, (sizes, displacements), MPI.LONG], root=root)
    
    if len(data.shape) == 1:
        return gdata
    else:
        if shape == 0:
            num_lines = data.shape[0]
            if num_lines > 0:
                return gdata.reshape((num_lines, gdata.size//num_lines))
            else: 
                return gdata.reshape((0, gdata.shape[1]))
        if shape == 1:
            num_columns = data.shape[1]
            if num_columns > 0:
                return gdata.reshape((gdata.size//num_columns, num_columns))
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
                return gdata.reshape((num_lines, gdata.size//num_lines))
            else: 
                return gdata.reshape((0, gdata.shape[1]))
        if shape == 1:
            num_columns = data.shape[1]
            if num_columns > 0:
                return gdata.reshape((gdata.size//num_columns, num_columns))
            else:
                return gdata.reshape((gdata.shape[0], 0))

def get_mpi_type(data_type):
    if data_type == 'int16':
        return MPI.SHORT
    elif data_type == 'uint16':
        return MPI.UNSIGNED_SHORT
    elif data_type == 'float32':
        return MPI.FLOAT
    elif data_type == 'int32':
        return MPI.INT