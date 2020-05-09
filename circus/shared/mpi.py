import numpy, os, mpi4py, logging, sys
import mpi4py
mpi4py.rc.threads = False
from mpi4py import MPI
from circus.shared.messages import print_and_log
comm = MPI.COMM_WORLD
import blosc
# from distutils.version import StrictVersion


logger = logging.getLogger(__name__)
MPI_VENDOR = MPI.get_vendor()
SHARED_MEMORY = (hasattr(MPI.Win, 'Allocate_shared') and callable(getattr(MPI.Win, 'Allocate_shared')))


def get_local_ring(local_only=False):
    # # First we need to identify machines in the MPI ring.
    from uuid import getnode as get_mac
    myip = numpy.int64(get_mac()) % 100000
    is_local = False

    if local_only:
        master_ip = comm.bcast(numpy.array([myip], dtype='int64'), root=0)
        is_local = myip == master_ip[0]
        sub_comm  = comm.Split_type(MPI.COMM_TYPE_SHARED, is_local)
    else:
        sub_comm  = comm.Split_type(MPI.COMM_TYPE_SHARED, myip)

    return sub_comm


sub_comm = get_local_ring()

def test_mpi_ring(nb_nodes):
    if comm.size != nb_nodes:
        if comm.rank == 0:
            lines = [
                "The MPI install does not seems to be correct!",
                "Be sure mpi4py has been compiled for your default version!",
            ]
            print_and_log(lines, 'warning', logger)
        sys.exit()


def check_if_cluster():
    from uuid import getnode as get_mac
    myip = numpy.array([numpy.int64(get_mac()) % 100000], dtype='int64')
    ips = all_gather_array(myip, comm, 1, 'int64')
    return not len(numpy.unique(ips)) == 1


def check_valid_path(path):

    data = numpy.array([os.path.exists(path)], dtype='int32')
    res = all_gather_array(data, comm, dtype='int32').astype(numpy.bool)
    return numpy.all(res)



def detect_memory(params, whitening=False, filtering=False, fitting=False):
    from psutil import virtual_memory

    N_e  = params.getint('data', 'N_e')
    safety_threshold = params.getfloat('data', 'memory_usage')
    data_file = params.data_file
    data_file.open()
    sampling_rate  = data_file.sampling_rate
    duation = data_file.duration
    data_file.close()

    from uuid import getnode as get_mac
    myip = numpy.int64(get_mac()) % 100000
    sub_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, myip)

    res = numpy.zeros(1, dtype=numpy.int64)
    mem = virtual_memory()

    if sub_comm.rank == 0:
        res[0] = safety_threshold * numpy.int64(mem.available//sub_comm.size)

    sub_comm.Barrier()
    sub_comm.Free()

    memory = all_gather_array(res, comm, 1, 'int64')

    idx = numpy.where(memory > 0)
    max_memory = numpy.min(memory[idx]) // (4 * N_e)

    if whitening or filtering:
        max_size = int(30*data_file.sampling_rate)
    elif fitting:
        max_size = int(0.1*data_file.sampling_rate)
    else:       
        max_size = (data_file.duration//comm.size)

    chunk_size = min(max_memory, max_size)
    
    if comm.rank == 0:
        print_and_log(['Setting data chunk size to %g second' % (chunk_size/float(sampling_rate))], 'debug', logger)

    return chunk_size


def gather_mpi_arguments(hostfile, params):
    print_and_log(['MPI detected: %s' % str(MPI_VENDOR)], 'debug', logger)
    if MPI_VENDOR[0] == 'Open MPI':
        if MPI_VENDOR[1][0] >= 3:
            print_and_log(['SpyKING CIRCUS does not work with OPENMPI >= 3.0',
                           'Consider downgrading or switching to MPICH'], 'error', logger)
            sys.exit(0)
        mpi_args = ['mpirun', '--mca', 'mpi_warn_on_fork', '0']
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
    elif MPI_VENDOR[0] == 'Intel MPI':
        mpi_args = ['mpiexec']
        if os.path.exists(hostfile):
            mpi_args += ['-machinefile', hostfile]
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
    size = data.size
    sizes = mpi_comm.gather(size, root=root) or []
    # now we pass the data
    displacements = [numpy.int64(sum(sizes[:i])) for i in range(len(sizes))]

    np_type = get_np_dtype(dtype)
    mpi_type = get_mpi_type(dtype)
    data_shape = data.shape

    if not compress:
        gdata = numpy.empty(numpy.int64(sum(sizes)), dtype=np_type)
        mpi_comm.Gatherv([data.flatten(), size, mpi_type], [gdata, (sizes, displacements), mpi_type], root=root)
    else:
        data = blosc.compress(data, typesize=mpi_type.size, cname='blosclz')
        data = mpi_comm.gather(data, root=root)
        gdata = [numpy.empty(0, dtype=np_type)]
        if comm.rank == root:
            for blosc_data in data:
                gdata.append(numpy.frombuffer(blosc.decompress(blosc_data), dtype=np_type))
        gdata = numpy.concatenate(gdata)

    if len(data_shape) == 1:
        return gdata
    else:
        if shape == 0:
            num_lines = data_shape[0]
            if num_lines > 0:
                return gdata.reshape((num_lines, gdata.size//num_lines))
            else:
                return gdata.reshape((0, gdata.shape[1]))
        if shape == 1:
            num_columns = data_shape[1]
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

    np_type = get_np_dtype(dtype)
    mpi_type = get_mpi_type(dtype)
    data_shape = data.shape

    if not compress:
        gdata = numpy.empty(numpy.int64(sum(sizes)), dtype=np_type)
        mpi_comm.Allgatherv([data.flatten(), size, mpi_type], [gdata, (sizes, displacements), mpi_type])
    else:
        data = blosc.compress(data, typesize=mpi_type.size, cname='blosclz')
        data = mpi_comm.allgather(data)
        gdata = [numpy.empty(0, dtype=np_type)]
        for blosc_data in data:
            gdata.append(numpy.frombuffer(blosc.decompress(blosc_data), dtype=np_type))
        gdata = numpy.concatenate(gdata)

    if len(data_shape) == 1:
        return gdata
    else:
        if shape == 0:
            num_lines = data_shape[0]
            if num_lines > 0:
                return gdata.reshape((num_lines, gdata.size//num_lines))
            else:
                return gdata.reshape((0, gdata.shape[1]))
        if shape == 1:
            num_columns = data_shape[1]
            if num_columns > 0:
                return gdata.reshape((gdata.size//num_columns, num_columns))
            else:
                return gdata.reshape((gdata.shape[0], 0))


def get_np_dtype(data_type):
    return numpy.dtype(data_type)


def get_mpi_type(data_type):
    if hasattr(MPI, '_typedict'):
        mpi_type = MPI._typedict[numpy.dtype(data_type).char]
    elif hasattr(MPI, '__TypeDict__'):
        mpi_type = MPI.__TypeDict__[numpy.dtype(data_type).char]
    else:
        raise ValueError('cannot convert type')
    return mpi_type
