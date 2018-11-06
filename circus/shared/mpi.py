import numpy, os, mpi4py, logging
from mpi4py import MPI
from messages import print_and_log
comm = MPI.COMM_WORLD
import blosc

logger = logging.getLogger(__name__)

MPI_VENDOR = MPI.get_vendor()
SHARED_MEMORY = (hasattr(MPI.Win, 'Allocate_shared') and callable(getattr(MPI.Win, 'Allocate_shared')))

def check_if_cluster():
    from uuid import getnode as get_mac
    myip = numpy.array([numpy.int64(get_mac()) % 100000], dtype='int64')
    ips = all_gather_array(myip, comm, 1, 'int64')
    return not len(numpy.unique(ips)) == 1

def get_local_ring(local_only=False):
    ## First we need to identify machines in the MPI ring
    from uuid import getnode as get_mac
    myip = numpy.int64(get_mac()) % 100000
    is_local = False

    if local_only:
        master_ip = comm.bcast(numpy.array([myip], dtype='int64'), root=0)
        is_local = myip == master_ip[0]
        sub_comm  = comm.Split(is_local, 0)
    else:
        sub_comm  = comm.Split(myip, 0)

    return sub_comm, is_local

def gather_mpi_arguments(hostfile, params):
    print_and_log(['MPI detected: %s' % str(MPI_VENDOR)], 'debug', logger)
    if MPI_VENDOR[0] == 'Open MPI':
        mpi_args = ['mpirun']
        if os.getenv('LD_LIBRARY_PATH'):
            mpi_args += ['-x', 'LD_LIBRARY_PATH']
        if os.getenv('PATH'):
            mpi_args += ['-x', 'PATH']
        if os.getenv('PYTHONPATH'):
            mpi_args += ['-x', 'PYTHONPATH']
        if os.path.exists(hostfile):
            mpi_args += ['-hostfile', hostfile]
    elif MPI_VENDOR[0] == 'Microsoft MPI':
        mpi_args = ['mpiexec']
        if os.path.exists(hostfile):
            mpi_args += ['-machinefile', hostfile]
    elif MPI_VENDOR[0] == 'MPICH2':
        mpi_args = ['mpiexec']
        if os.path.exists(hostfile):
            mpi_args += ['-f', hostfile]
    elif MPI_VENDOR[0] == 'MPICH':
        mpi_args = ['mpiexec']
        if os.path.exists(hostfile):
            mpi_args += ['-f', hostfile]
    else:
        print_and_log([
                        '%s may not be yet properly implemented: contact developpers' %
                        MPI_VENDOR[0]], 'error', logger)
        mpi_args = ['mpirun']
        if os.path.exists(hostfile):
            mpi_args += ['-hostfile', hostfile]
    return mpi_args

def gather_array(data, mpi_comm, root=0, shape=0, dtype='float32', compress=False):
    # gather 1D or 2D numpy arrays
    assert isinstance(data, numpy.ndarray)
    assert len(data.shape) < 3
    # first we pass the data size
    size  = data.size
    sizes = mpi_comm.gather(size, root=root) or []
    # now we pass the data
    displacements = [numpy.int64(sum(sizes[:i])) for i in range(len(sizes))]

    np_type       = get_np_dtype(dtype)
    mpi_type      = get_mpi_type(dtype)    

    if not compress:
        gdata = numpy.empty(numpy.int64(sum(sizes)), dtype=np_type)
        mpi_comm.Gatherv([data.flatten(), size, mpi_type], [gdata, (sizes, displacements), mpi_type], root=root)
    else:
        new_data = blosc.compress(data, typesize=mpi_type.size, cname='blosclz')
        new_data = mpi_comm.gather(new_data, root=0)
        gdata = numpy.empty(0, dtype=np_type)
        if comm.rank == 0:
            for blosc_data in new_data:
                gdata = numpy.concatenate((gdata, numpy.frombuffer(blosc.decompress(blosc_data), dtype=np_type)))

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

def all_gather_array(data, mpi_comm, shape=0, dtype='float32', compress=False):
    # gather 1D or 2D numpy arrays
    assert isinstance(data, numpy.ndarray)
    assert len(data.shape) < 3
    # first we pass the data size
    size  = data.size
    sizes = mpi_comm.allgather(size) or []
    # now we pass the data
    displacements = [numpy.int64(sum(sizes[:i])) for i in range(len(sizes))]

    np_type       = get_np_dtype(dtype)
    mpi_type      = get_mpi_type(dtype)

    if not compress:
        gdata = numpy.empty(numpy.int64(sum(sizes)), dtype=np_type)
        mpi_comm.Allgatherv([data.flatten(), size, mpi_type], [gdata, (sizes, displacements), mpi_type])
    else:
        new_data = blosc.compress(data, typesize=mpi_type.size, cname='blosclz')
        new_data = mpi_comm.allgather(new_data)
        gdata = numpy.empty(0, dtype=np_type)
        for blosc_data in new_data:
            gdata = numpy.concatenate((gdata, numpy.frombuffer(blosc.decompress(blosc_data), dtype=np_type)))

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


def get_np_dtype(data_type):
    return numpy.dtype(data_type)

def get_mpi_type(data_type):
    return MPI._typedict[get_np_dtype(data_type).char]
