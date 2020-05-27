from __future__ import division

from circus.shared.utils import get_tqdm_progressbar
import numpy
import os
import platform
import re
import scipy
import logging
import sys
import gc

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

from colorama import Fore
from circus.shared.mpi import all_gather_array, gather_array, comm, get_local_ring, MPI, sub_comm
from circus.shared.probes import get_nodes_and_edges, get_central_electrode
from circus.shared.messages import print_and_log
from circus.shared.utils import purge, get_parallel_hdf5_flag, indices_for_dead_times, get_shared_memory_flag
import circus


logger = logging.getLogger(__name__)


def data_stats(params, show=True, export_times=False):

    data_file = params.get_data_file(source=True, has_been_created=False)
    stream_mode = data_file.is_stream
    chunk_size = 60 * data_file.sampling_rate
    nb_chunks = data_file.duration // chunk_size
    last_chunk_len = data_file.duration - nb_chunks * chunk_size

    nb_seconds = last_chunk_len // params.rate
    last_chunk_len -= (nb_seconds * params.rate)
    if nb_seconds > 60:
        nb_extra_seconds = nb_seconds // 60
        nb_chunks += nb_extra_seconds
        nb_seconds -= 60 * nb_extra_seconds
    last_chunk_len = int(1000 * last_chunk_len / params.rate)

    N_t = params.getint('detection', 'N_t')
    N_t = numpy.round(1000.0 * N_t / params.rate, 1)

    if params.get('detection', 'peaks') == 'both':
        threshold = 'positive and negative'
    else:
        threshold = params.get('detection', 'peaks')

    lines = [
        "Number of recorded channels : %d" % params.nb_channels,
        "Number of analyzed channels : %d" % params.getint('data', 'N_e'),
        "File format                 : %s" % params.get('data', 'file_format').upper(),
        "Data type                   : %s" % str(data_file.data_dtype),
        "Sampling rate               : %d kHz" % (params.rate//1000.0),
        "Duration of the recording   : %d min %s s %s ms" % (nb_chunks, int(nb_seconds), last_chunk_len),
        "Width of the templates      : %d ms" % N_t,
        "Spatial radius considered   : %d um" % params.getint('detection', 'radius'),
        "Threshold crossing          : %s" % threshold
    ]

    if stream_mode:
        lines += [
            "Streams                     : %s (%d found)" % (params.get('data', 'stream_mode'), data_file.nb_streams)
        ]

    if show:
        print_and_log(lines, 'info', logger)

    if not export_times:
        return nb_chunks * 60 + nb_seconds + last_chunk_len/1000.
    else:
        return times


def get_stas(
        params, times_i, labels_i, src, neighs, nodes=None,
        mean_mode=False, all_labels=False, pos='neg', auto_align=True
):

    data_file = params.data_file
    data_file.open()
    N_t = params.getint('detection', 'N_t')
    if not all_labels:
        if not mean_mode:
            stas = numpy.zeros((len(times_i), len(neighs), N_t), dtype=numpy.float32)
        else:
            stas = numpy.zeros((len(neighs), N_t), dtype=numpy.float32)
    else:
        nb_labels = numpy.unique(labels_i)
        stas = numpy.zeros((len(nb_labels), len(neighs), N_t), dtype=numpy.float32)

    alignment = params.getboolean('detection', 'alignment') and auto_align
    over_factor = float(params.getint('detection', 'oversampling_factor'))
    nb_jitter = params.getint('detection', 'nb_jitter')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening = params.getboolean('whitening', 'spatial')
    template_shift = params.getint('detection', 'template_shift')
    jitter_range = params.getint('detection', 'jitter_range')
    smoothing_factor = params.getfloat('detection', 'smoothing_factor')
    template_shift_2 = template_shift + jitter_range
    mads = load_data(params, 'mads')

    if do_spatial_whitening:
        spatial_whitening = load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = load_data(params, 'temporal_whitening')

    if alignment:
        cdata = numpy.linspace(-jitter_range, jitter_range, nb_jitter)
        xdata = numpy.arange(-template_shift_2, template_shift_2 + 1)
        xoff  = len(cdata) / 2.
        duration = 2 * template_shift_2 + 1
        # if pos  == 'neg':
        #     weights = smoothing_factor / load_data(params, 'weights')
        # elif pos == 'pos':
        #     weights = smoothing_factor / load_data(params, 'weights-pos')
        align_factor = duration
        local_factor = align_factor*((smoothing_factor*mads[src])**2)
    else:
        xdata = numpy.arange(-template_shift, template_shift + 1)
        duration = N_t
    
    offset = duration // 2
    idx = numpy.where(neighs == src)[0]
    ydata = numpy.arange(len(neighs))

    count = 0
    for lb, time in zip(labels_i, times_i):

        local_chunk = data_file.get_snippet(time - offset, duration, nodes=nodes)

        if do_spatial_whitening:
            local_chunk = numpy.dot(local_chunk, spatial_whitening)
        if do_temporal_whitening:
            local_chunk = scipy.ndimage.filters.convolve1d(local_chunk, temporal_whitening, axis=0, mode='constant')

        local_chunk = numpy.take(local_chunk, neighs, axis=1)

        if alignment:
            if len(ydata) == 1:
                smoothed = True
                try:
                    f = scipy.interpolate.UnivariateSpline(xdata, local_chunk, s=local_factor, k=3)
                except Exception:
                    smoothed = False
                    f = scipy.interpolate.UnivariateSpline(xdata, local_chunk, k=3, s=0)
                if pos == 'neg':
                    rmin = (numpy.argmin(f(cdata)) - xoff) / over_factor
                elif pos == 'pos':
                    rmin = (numpy.argmax(f(cdata)) - xoff) / over_factor
                if smoothed:
                    f = scipy.interpolate.UnivariateSpline(xdata, local_chunk, s=0, k=3)
                ddata = numpy.linspace(rmin-template_shift, rmin+template_shift, N_t)
                local_chunk = f(ddata).astype(numpy.float32).reshape(N_t, 1)
            else:
                try:
                    f = scipy.interpolate.UnivariateSpline(xdata, local_chunk[:, idx], s=local_factor, k=3)
                except Exception:
                    f = scipy.interpolate.UnivariateSpline(xdata, local_chunk[:, idx], k=3, s=0)
                if pos == 'neg':
                    rmin = (numpy.argmin(f(cdata)) - xoff)/over_factor
                elif pos == 'pos':
                    rmin = (numpy.argmax(f(cdata)) - xoff)/over_factor
                ddata = numpy.linspace(rmin-template_shift, rmin+template_shift, N_t)
                f = scipy.interpolate.RectBivariateSpline(xdata, ydata, local_chunk, s=0, kx=3, ky=1)
                local_chunk = f(ddata, ydata).astype(numpy.float32)

        if all_labels:
            lc = numpy.where(nb_labels == lb)[0]
            stas[lc] += local_chunk.T
        else:
            if not mean_mode:
                stas[count, :, :] = local_chunk.T
                count += 1
            else:
                stas += local_chunk.T

    data_file.close()

    return stas


def get_dead_times(params):
    """
    Read the sampling points to be excluced from the data file from 
    'dead_file' in [triggers] section of the params file. If 'dead_unit'
    is 'ms', it will transform sampling points in ms.

    Parameters
    ----------
    params -- the parameter file.

    Returns
    -------
    A 2D NumPy array containing the start and stop sampling points 
    (or times) to be excluded from the data. 
    """

    def _get_dead_times(params):
        dead_times = numpy.loadtxt(fname=params.get('triggers', 'dead_file'), comments=['#', '//'])
        data_file = params.data_file
        if len(dead_times.shape) == 1:
            dead_times = dead_times.reshape(1, 2)
        dead_in_ms = params.getboolean('triggers', 'dead_in_ms')
        if dead_in_ms:
            dead_times *= numpy.int64(data_file.sampling_rate*1e-3)
        dead_times = dead_times.astype(numpy.int64)
        all_dead_times = indices_for_dead_times(dead_times[:, 0], dead_times[:, 1])
        return all_dead_times

    if not get_shared_memory_flag(params):
        return _get_dead_times(params)
    else:
        intsize = MPI.LONG_LONG.Get_size()
        nb_dead_times = 0
        local_rank = sub_comm.rank

        if local_rank == 0:
            dead_times = _get_dead_times(params)
            nb_dead_times = len(dead_times)

        sub_comm.Barrier()
        long_size = numpy.int64(sub_comm.bcast(numpy.array([nb_dead_times], dtype=numpy.uint32), root=0)[0])

        if local_rank == 0:
            data_bytes = long_size * intsize
        else:
            indptr_bytes = 0
            indices_bytes = 0
            data_bytes = 0

        win_data = MPI.Win.Allocate_shared(data_bytes, intsize, comm=sub_comm)
        buf_data, _ = win_data.Shared_query(0)
        buf_data = numpy.array(buf_data, dtype='B', copy=False)
        data = numpy.ndarray(buffer=buf_data, dtype=numpy.int64, shape=(long_size,))
        sub_comm.Barrier()

        if sub_comm.rank == 0:
            data[:] = dead_times

        sub_comm.Barrier()
        return data


def get_stas_memshared(
        params, times_i, labels_i, src, neighs, nodes=None, mean_mode=False, all_labels=False, auto_align=True
):

    # Load parameters.
    data_file = params.data_file
    data_file.open()
    N_t = params.getint('detection', 'N_t')
    N_total = params.nb_channels
    alignment = params.getboolean('detection', 'alignment') and auto_align
    over_factor = float(params.getint('detection', 'oversampling_factor'))
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening = params.getboolean('whitening', 'spatial')
    template_shift = params.getint('detection', 'template_shift')
    template_shift_2 = round(1.25 * template_shift)
    duration = 2 * N_t - 1

    # Calculate the sizes of the data structures to share.
    nb_triggers = 0
    nb_neighs = 0
    local_rank = sub_comm.rank
    nb_ts = 0
    if local_rank == 0:
        if not all_labels:
            if not mean_mode:
                nb_triggers = len(times_i)
            else:
                nb_triggers = 1
        else:
            nb_triggers = len(numpy.unique(labels_i))
        nb_neighs = len(neighs)
        nb_ts = N_t

    sub_comm.Barrier()

    # Broadcast the sizes of the data structures to share.
    triggers_size = numpy.int64(sub_comm.bcast(numpy.array([nb_triggers], dtype=numpy.uint32), root=0)[0])
    neighs_size = numpy.int64(sub_comm.bcast(numpy.array([nb_neighs], dtype=numpy.uint32), root=0)[0])
    ts_size = numpy.int64(sub_comm.bcast(numpy.array([nb_ts], dtype=numpy.uint32), root=0)[0])

    # Declare the data structures to share.
    if local_rank == 0:
        stas_bytes = triggers_size * neighs_size * ts_size * float_size
    else:
        stas_bytes = 0
    if triggers_size == 1:
        stas_shape = (neighs_size, ts_size)
    else:
        stas_shape = (triggers_size, neighs_size, ts_size)

    win_stas = MPI.Win.Allocate_shared(stas_bytes, float_size, comm=sub_comm)
    buf_stas, _ = win_stas.Shared_query(0)
    buf_stas = numpy.array(buf_stas, dtype='B', copy=False)
    stas = numpy.ndarray(buffer=buf_stas, dtype=numpy.float32, shape=stas_shape)

    sub_comm.Barrier()

    # Let master node initialize the data structures to share.
    if local_rank == 0:
        if do_spatial_whitening:
            spatial_whitening = load_data(params, 'spatial_whitening')
        if do_temporal_whitening:
            temporal_whitening = load_data(params, 'temporal_whitening')
        if alignment:
            cdata = numpy.linspace(-template_shift / 4, template_shift / 4, int(over_factor * template_shift / 2))
            xdata = numpy.arange(-template_shift_2, template_shift_2 + 1)
            xoff = len(cdata) / 2.0

        count = 0
        for lb, time in zip(labels_i, times_i):

            if alignment:
                local_chunk = data_file.get_snippet(time - template_shift_2, duration, nodes=nodes)
            else:
                local_chunk = data_file.get_snippet(time - template_shift, N_t, nodes=nodes)

            if do_spatial_whitening:
                local_chunk = numpy.dot(local_chunk, spatial_whitening)
            if do_temporal_whitening:
                local_chunk = scipy.ndimage.filters.convolve1d(local_chunk, temporal_whitening, axis=0, mode='constant')

            local_chunk = numpy.take(local_chunk, neighs, axis=1)

            if alignment:
                idx = numpy.where(neighs == src)[0]
                ydata = numpy.arange(len(neighs))
                if len(ydata) == 1:
                    f = scipy.interpolate.UnivariateSpline(xdata, local_chunk, k=3, s=0)
                    rmin = (numpy.argmin(f(cdata)) - xoff) / over_factor
                    ddata = numpy.linspace(rmin - template_shift, rmin + template_shift, N_t)
                    local_chunk = f(ddata).astype(numpy.float32).reshape(N_t, 1)
                else:
                    f = scipy.interpolate.RectBivariateSpline(xdata, ydata, local_chunk, ky=1, kx=3, s=0)
                    rmin = (numpy.argmin(f(cdata, idx)[:, 0]) - xoff) / over_factor
                    ddata = numpy.linspace(rmin - template_shift, rmin + template_shift, N_t)
                    local_chunk = f(ddata, ydata).astype(numpy.float32)
            if not all_labels:
                if not mean_mode:
                    stas[count, :, :] = local_chunk.T
                    count += 1
                else:
                    stas += local_chunk.T
            else:
                lc = numpy.where(nb_triggers == lb)[0]
                stas[lc] += local_chunk.T

    sub_comm.Barrier()

    # # Let each node wrap the data structures to share.
    # if not all_labels and mean_mode:
    #     stas_shape = (nb_neighs, nb_ts)
    # else:
    #     stas_shape = (nb_triggers, nb_neighs, nb_ts)
    # stas = numpy.reshape(stas, stas_shape)

    data_file.close()

    return stas


def get_artefact(params, times_i, tau, nodes):

    data_file = params.data_file
    data_file.open()

    dx, dy = len(nodes), int(tau)
    artefact = numpy.zeros((0, dx, dy), dtype=numpy.float32)
    for time in times_i:
        snippet = data_file.get_snippet(int(time), int(tau), nodes).T.reshape(1, dx, dy)
        artefact = numpy.vstack((artefact, snippet))

    artefact = numpy.median(artefact, 0)

    data_file.close()

    return artefact


def load_data_memshared(
        params, data, extension='', normalize=False, transpose=False,
        nb_cpu=1, nb_gpu=0, use_gpu=False):

    file_out = params.get('data', 'file_out')
    file_out_suff = params.get('data', 'file_out_suff')
    data_file_noext = params.get('data', 'data_file_noext')
    local_rank = sub_comm.rank
    intsize = MPI.INT.Get_size()
    floatsize = MPI.FLOAT.Get_size()

    data_file = params.data_file
    N_e = params.getint('data', 'N_e')
    N_t = params.getint('detection', 'N_t')

    if data == 'templates':

        file_name = file_out_suff + '.templates%s.hdf5' % extension
        if os.path.exists(file_name):

            nb_data = 0
            nb_ptr = 0
            indptr_bytes = 0
            indices_bytes = 0
            data_bytes = 0
            nb_templates = h5py.File(file_name, 'r', libver='earliest').get('norms').shape[0]

            if local_rank == 0:
                temp_x = h5py.File(file_name, 'r', libver='earliest').get('temp_x')[:].ravel()
                temp_y = h5py.File(file_name, 'r', libver='earliest').get('temp_y')[:].ravel()
                temp_data = h5py.File(file_name, 'r', libver='earliest').get('temp_data')[:].ravel()
                sparse_mat = scipy.sparse.csc_matrix((temp_data, (temp_x, temp_y)), shape=(N_e*N_t, nb_templates))
                if normalize:
                    norm_templates = load_data(params, 'norm-templates')
                    for idx in range(sparse_mat.shape[1]):
                        myslice = numpy.arange(sparse_mat.indptr[idx], sparse_mat.indptr[idx+1])
                        sparse_mat.data[myslice] /= norm_templates[idx]
                if transpose:
                    sparse_mat = sparse_mat.T

                nb_data = len(sparse_mat.data)
                nb_ptr = len(sparse_mat.indptr)

            long_size = numpy.int64(sub_comm.bcast(numpy.array([nb_data], dtype=numpy.int32), root=0)[0])
            short_size = numpy.int64(sub_comm.bcast(numpy.array([nb_ptr + nb_data], dtype=numpy.int32), root=0)[0])

            if local_rank == 0:
                indices_bytes = short_size * intsize
                data_bytes = long_size * floatsize

            win_data = MPI.Win.Allocate_shared(data_bytes, floatsize, comm=sub_comm)
            win_indices = MPI.Win.Allocate_shared(indices_bytes + indptr_bytes, intsize, comm=sub_comm)

            buf_data, _ = win_data.Shared_query(0)
            buf_indices, _ = win_indices.Shared_query(0)

            buf_data = numpy.array(buf_data, dtype='B', copy=False)
            buf_indices = numpy.array(buf_indices, dtype='B', copy=False)

            data = numpy.ndarray(buffer=buf_data, dtype=numpy.float32, shape=(long_size,))
            indices = numpy.ndarray(buffer=buf_indices, dtype=numpy.int32, shape=(short_size,))

            if local_rank == 0:
                data[:] = sparse_mat.data
                indices[:long_size] = sparse_mat.indices
                indices[long_size:] = sparse_mat.indptr
                del sparse_mat

            if not transpose:
                templates = scipy.sparse.csc_matrix((N_e * N_t, nb_templates), dtype=numpy.float32)
            else:
                templates = scipy.sparse.csr_matrix((nb_templates, N_e * N_t), dtype=numpy.float32)

            templates.data = data
            templates.indices = indices[:long_size]
            templates.indptr = indices[long_size:]

            return templates, (win_data, win_indices)
        else:
            if comm.rank == 0:
                print_and_log(["No templates found! Check suffix?"], 'error', logger)
            sys.exit(0)
    elif data == "overlaps":

        file_name = file_out_suff + '.overlap%s.hdf5' % extension
        if os.path.exists(file_name):

            c_overlap = h5py.File(file_name, 'r')
            
            over_shape = c_overlap.get('over_shape')[:]
            N_over = numpy.int64(numpy.sqrt(over_shape[0]))
            S_over = over_shape[1]
            c_overs = {}
            nb_data = 0

            if local_rank == 0:
                over_x = c_overlap.get('over_x')[:]
                over_y = c_overlap.get('over_y')[:]
                over_data = c_overlap.get('over_data')[:]
                nb_data = len(over_x) * 2

            c_overlap.close()

            nb_ptr = 0
            indptr_bytes = 0
            indices_bytes = 0
            data_bytes = 0

            nb_data = numpy.int64(sub_comm.bcast(numpy.array([nb_data], dtype=numpy.int32), root=0)[0])
            win_data = MPI.Win.Allocate_shared(nb_data * floatsize, floatsize, comm=sub_comm)
            buf_data, _ = win_data.Shared_query(0)
            buf_data = numpy.array(buf_data, dtype='B', copy=False)


            factor = 2 * int(max(nb_data, (N_over + 1) ** 2))
            win_indices = MPI.Win.Allocate_shared(factor * intsize, intsize, comm=sub_comm)
            buf_indices, _ = win_indices.Shared_query(0)
            buf_indices = numpy.array(buf_indices, dtype='B', copy=False)

            data = numpy.ndarray(buffer=buf_data, dtype=numpy.float32, shape=(nb_data,))
            indices = numpy.ndarray(buffer=buf_indices, dtype=numpy.int32, shape=(factor,))

            global_offset_data = 0
            global_offset_ptr = 0
            local_nb_data = 0
            local_nb_ptr = 0

            duration = over_shape[1] // 2

            res = []
            res2 = []
            for i in range(N_over):
                res += [i * N_over, (i + 1) * N_over]
                res2 += [i, i+1]

            if local_rank == 0:
                bounds = numpy.searchsorted(over_x, res, 'left')
                sub_over = numpy.mod(over_x, N_over)
                mask_duration = (over_y < duration)
                over_sorted = numpy.argsort(sub_over).astype(numpy.int32)
                bounds_2 = numpy.searchsorted(sub_over[over_sorted], res2, 'left')

            import gc

            for i in range(N_over):

                if local_rank == 0:
                    xmin, xmax = bounds[2*i:2*(i+1)]
                    local_x = over_x[xmin:xmax] - i * N_over
                    local_y = over_y[xmin:xmax]
                    local_data = over_data[xmin:xmax]

                    xmin, xmax = bounds_2[2*i:2*(i+1)]
                    nslice = over_sorted[xmin:xmax][mask_duration[over_sorted[xmin:xmax]]]

                    local_x = numpy.concatenate((local_x, over_x[nslice] // N_over))
                    local_y = numpy.concatenate((local_y, (over_shape[1] - 1) - over_y[nslice]))
                    local_data = numpy.concatenate((local_data, over_data[nslice]))

                    sparse_mat = scipy.sparse.csr_matrix((local_data, (local_x, local_y)), shape=(N_over, over_shape[1]))
                    local_nb_data = len(sparse_mat.data)
                    local_nb_ptr = len(sparse_mat.indptr)

                local_nb_data = numpy.int64(sub_comm.bcast(numpy.array([local_nb_data], dtype=numpy.int32), root=0)[0])
                local_nb_ptr = numpy.int64(sub_comm.bcast(numpy.array([local_nb_ptr], dtype=numpy.int32), root=0)[0])

                boundary_data = global_offset_data + local_nb_data
                boundary_ptr = global_offset_ptr + factor // 2

                if local_rank == 0:
                    data[global_offset_data:boundary_data] = sparse_mat.data
                    indices[global_offset_data:boundary_data] = sparse_mat.indices
                    indices[boundary_ptr:boundary_ptr + local_nb_ptr] = sparse_mat.indptr
                    del sparse_mat

                c_overs[i] = scipy.sparse.csr_matrix((N_over, S_over), dtype=numpy.float32)
                c_overs[i].data = data[global_offset_data:boundary_data]
                c_overs[i].indices = indices[global_offset_data:boundary_data]
                c_overs[i].indptr = indices[boundary_ptr:boundary_ptr + local_nb_ptr]
                global_offset_data += local_nb_data
                global_offset_ptr += local_nb_ptr

                if local_rank == 0:
                    del local_x, local_y, local_data, nslice

            if local_rank == 0:
                del over_x, over_y, over_data, over_sorted, sub_over

            gc.collect()

            sub_comm.Barrier()
            return c_overs, (win_data, win_indices)
        else:
            if comm.rank == 0:
                print_and_log(["No overlaps found! Check suffix?"], 'error', logger)
            sys.exit(0)

    elif data == "overlaps-raw":

        file_name = file_out_suff + '.overlap%s.hdf5' % extension
        if os.path.exists(file_name):

            c_overlap = h5py.File(file_name, 'r')
            over_shape = c_overlap.get('over_shape')[:]
            N_over = int(numpy.sqrt(over_shape[0]))
            S_over = over_shape[1]
            c_overs = {}
            indices_bytes = 0
            data_bytes = 0
            nb_data = 0

            if local_rank == 0:
                over_x = c_overlap.get('over_x')[:]
                over_y = c_overlap.get('over_y')[:]
                over_data = c_overlap.get('over_data')[:]
                nb_data = len(over_x)

            c_overlap.close()

            nb_data = numpy.int64(sub_comm.bcast(numpy.array([nb_data], dtype=numpy.int32), root=0)[0])

            if local_rank == 0:
                indices_bytes = nb_data * intsize
                data_bytes = nb_data * floatsize

            win_data = MPI.Win.Allocate_shared(data_bytes, floatsize, comm=sub_comm)
            win_indices_x = MPI.Win.Allocate_shared(indices_bytes, intsize, comm=sub_comm)
            win_indices_y = MPI.Win.Allocate_shared(indices_bytes, intsize, comm=sub_comm)
            win_indices_sub = MPI.Win.Allocate_shared(indices_bytes, intsize, comm=sub_comm)
            win_indices_sorted = MPI.Win.Allocate_shared(indices_bytes, intsize, comm=sub_comm)

            buf_data, _ = win_data.Shared_query(0)
            buf_indices_x, _ = win_indices_x.Shared_query(0)
            buf_indices_y, _ = win_indices_y.Shared_query(0)
            buf_indices_sub, _ = win_indices_sub.Shared_query(0)
            buf_indices_sorted, _ = win_indices_sorted.Shared_query(0)

            buf_data = numpy.array(buf_data, dtype='B', copy=False)
            buf_indices_x = numpy.array(buf_indices_x, dtype='B', copy=False)
            buf_indices_y = numpy.array(buf_indices_y, dtype='B', copy=False)
            buf_indices_sub = numpy.array(buf_indices_sub, dtype='B', copy=False)
            buf_indices_sorted = numpy.array(buf_indices_sorted, dtype='B', copy=False)

            data = numpy.ndarray(buffer=buf_data, dtype=numpy.float32, shape=(nb_data,))
            indices_x = numpy.ndarray(buffer=buf_indices_x, dtype=numpy.int32, shape=(nb_data,))
            indices_y = numpy.ndarray(buffer=buf_indices_y, dtype=numpy.int32, shape=(nb_data,))
            indices_sub = numpy.ndarray(buffer=buf_indices_sub, dtype=numpy.int32, shape=(nb_data,))
            indices_sorted = numpy.ndarray(buffer=buf_indices_sorted, dtype=numpy.int32, shape=(nb_data,))

            sub_comm.Barrier()

            if local_rank == 0:
                data[:] = over_data
                indices_x[:] = over_x
                indices_y[:] = over_y
                indices_sub[:] = numpy.mod(over_x, N_over)
                indices_sorted[:] = numpy.argsort(indices_sub).astype(numpy.int32)
                del over_x, over_y, over_data

            sub_comm.Barrier()

            pointers = (win_data, win_indices_x, win_indices_y, win_indices_sub, win_indices_sorted)

            return indices_x, indices_y, data, indices_sub, indices_sorted, over_shape, pointers
        else:
            if comm.rank == 0:
                print_and_log(["No overlaps found! Check suffix?"], 'error', logger)
            sys.exit(0)

    elif data == 'clusters-light':

        file_name = file_out_suff + '.clusters%s.hdf5' % extension
        if os.path.exists(file_name):

            myfile = h5py.File(file_name, 'r', libver='earliest')
            result = {}

            nb_data = 0

            for key in myfile.keys():

                if ('clusters_' in key) or (key == 'electrodes'):
                    if local_rank == 0:
                        locdata = myfile.get(key)[:]
                        nb_data = len(locdata)

                    data_size = numpy.int64(sub_comm.bcast(numpy.array([nb_data], dtype=numpy.int32), root=0)[0])
                    type_size = 0
                    data_bytes = 0

                    if local_rank == 0:
                        if locdata.dtype == 'int32':
                            type_size = 0
                        elif locdata.dtype == 'float32':
                            type_size = 1
                        data_bytes = data_size * 4
                        
                    type_size = numpy.int64(sub_comm.bcast(numpy.array([type_size], dtype=numpy.int32), root=0)[0])
                    empty = numpy.int64(sub_comm.bcast(numpy.array([data_bytes], dtype=numpy.int32), root=0)[0])
                    if empty > 0:
                        win_data = MPI.Win.Allocate_shared(data_bytes, 4, comm=sub_comm)
                        buf_data, _ = win_data.Shared_query(0)

                        buf_data = numpy.array(buf_data, dtype='B', copy=False)
                        if type_size == 0:
                            data = numpy.ndarray(buffer=buf_data, dtype=numpy.int32, shape=(data_size,))
                        elif type_size == 1:
                            data = numpy.ndarray(buffer=buf_data, dtype=numpy.float32, shape=(data_size,))

                        if local_rank == 0:
                            data[:] = locdata
                    else:
                        if type_size == 0:
                            data = numpy.zeros(0, dtype=numpy.int32)
                        elif type_size == 1:
                            data = numpy.zeros(0, dtype=numpy.float32)

                    sub_comm.Barrier()

                    result[str(key)] = data
            myfile.close()
            return result, (win_data, )
        else:
            if comm.rank == 0:
                print_and_log(["No clusters found! Check suffix?"], 'error', logger)
            sys.exit(0)


def load_data(params, data, extension=''):
    """
    Load data from a dataset.

    Parameters
    ----------
    data : {'thresholds', 'spatial_whitening', 'temporal_whitening', 'basis',
            'templates', 'norm-templates', 'spike-cluster', 'spikedetekt',
            'clusters', 'electrodes', 'results', 'overlaps', 'limits',
            'injected_spikes', 'triggers'}

    """

    file_out_suff = params.get('data', 'file_out_suff')
    data_file_noext = params.get('data', 'data_file_noext')

    if data == 'thresholds':
        filename = file_out_suff + '.basis.hdf5'
        spike_thresh = params.getfloat('detection', 'spike_thresh')
        if os.path.exists(filename):
            myfile = h5py.File(filename, 'r', libver='earliest')
            thresholds = myfile.get('thresholds')[:]
            myfile.close()
            return spike_thresh * thresholds
        else:
            if comm.rank == 0:
                print_and_log(["The whitening step should be launched first!"], 'error', logger)
            sys.exit(0)
    elif data == 'mads':
        filename = file_out_suff + '.basis.hdf5'
        if os.path.exists(filename):
            myfile = h5py.File(filename, 'r', libver='earliest')
            thresholds = myfile.get('thresholds')[:]
            myfile.close()
            return thresholds
        else:
            if comm.rank == 0:
                print_and_log(["The whitening step should be launched first!"], 'error', logger)
            sys.exit(0)
    elif data == 'stds':
        filename = file_out_suff + '.basis.hdf5'
        if os.path.exists(filename):
            myfile = h5py.File(filename, 'r', libver='earliest')
            thresholds = myfile.get('thresholds')[:]/0.674
            myfile.close()
            return thresholds
        else:
            if comm.rank == 0:
                print_and_log(["The whitening step should be launched first!"], 'error', logger)
            sys.exit(0)
    elif data == 'matched-thresholds':
        filename = file_out_suff + '.basis.hdf5'
        matched_thresh = params.getfloat('detection', 'matched_thresh')
        if os.path.exists(filename):
            myfile = h5py.File(filename, 'r', libver='earliest')
            thresholds = myfile.get('matched_thresholds')[:]
            myfile.close()
            return matched_thresh * thresholds
        else:
            if comm.rank == 0:
                print_and_log(["The whitening step should be launched first!"], 'error', logger)
            sys.exit(0)
    elif data == 'matched-thresholds-pos':
        filename = file_out_suff + '.basis.hdf5'
        matched_thresh = params.getfloat('detection', 'matched_thresh')
        if os.path.exists(filename):
            myfile = h5py.File(filename, 'r', libver='earliest')
            thresholds = myfile.get('matched_thresholds_pos')[:]
            myfile.close()
            return matched_thresh * thresholds
        else:
            if comm.rank == 0:
                print_and_log(["The whitening step should be launched first!"], 'error', logger)
            sys.exit(0)
    elif data == 'spatial_whitening':
        filename = file_out_suff + '.basis.hdf5'
        if os.path.exists(filename):
            try:
                myfile = h5py.File(filename, 'r', libver='earliest')
                spatial = numpy.ascontiguousarray(myfile.get('spatial')[:])
                myfile.close()
                return spatial
            except Exception:
                if comm.rank == 0:
                    print_and_log(["The whitening step should be launched first!"], 'error', logger)
                sys.exit(0)
        else:
            if comm.rank == 0:
                print_and_log(["The whitening step should be launched first!"], 'error', logger)
            sys.exit(0)
    elif data == 'temporal_whitening':
        filename = file_out_suff + '.basis.hdf5'
        if os.path.exists(filename):
            try:
                myfile = h5py.File(filename, 'r', libver='earliest')
                temporal = myfile.get('temporal')[:]
                myfile.close()
                return temporal
            except Exception:
                if comm.rank == 0:
                    print_and_log(["The whitening step should be launched first!"], 'error', logger)
                sys.exit(0)
        else:
            if comm.rank == 0:
                print_and_log(["The whitening step should be launched first!"], 'error', logger)
            sys.exit(0)
    elif data == 'basis':
        filename = file_out_suff + '.basis.hdf5'
        if os.path.exists(filename):
            try:
                myfile = h5py.File(filename, 'r', libver='earliest')
                basis_proj = numpy.ascontiguousarray(myfile.get('proj')[:])
                basis_rec = numpy.ascontiguousarray(myfile.get('rec')[:])
                myfile.close()
                return basis_proj, basis_rec
            except Exception:
                if comm.rank == 0:
                    print_and_log(["The whitening step should be launched first!"], 'error', logger)
                sys.exit(0)
        else:
            if comm.rank == 0:
                print_and_log(["The whitening step should be launched first!"], 'error', logger)
            sys.exit(0)
    elif data == 'basis-pos':
        filename = file_out_suff + '.basis.hdf5'
        if os.path.exists(filename):
            try:
                myfile = h5py.File(filename, 'r', libver='earliest')
                basis_proj = numpy.ascontiguousarray(myfile.get('proj_pos')[:])
                basis_rec = numpy.ascontiguousarray(myfile.get('rec_pos')[:])
                myfile.close()
                return basis_proj, basis_rec
            except Exception:
                if comm.rank == 0:
                    print_and_log(["The whitening step should be launched first!"], 'error', logger)
                sys.exit(0)
        else:
            if comm.rank == 0:
                print_and_log(["The whitening step should be launched first!"], 'error', logger)
            sys.exit(0)
    elif data == 'waveform':
        filename = file_out_suff + '.basis.hdf5'
        if os.path.exists(filename):
            myfile = h5py.File(filename, 'r', libver='earliest')
            waveforms = myfile.get('waveform')[:]
            myfile.close()
            return waveforms
        else:
            if comm.rank == 0:
                print_and_log(["The whitening step should be launched first!"], 'error', logger)
            sys.exit(0)
    elif data == 'waveforms':
        filename = file_out_suff + '.basis.hdf5'
        if os.path.exists(filename):
            myfile = h5py.File(filename, 'r', libver='earliest')
            waveforms = myfile.get('waveforms')[:]
            myfile.close()
            return waveforms
        else:
            if comm.rank == 0:
                print_and_log(["The whitening step should be launched first!"], 'error', logger)
            sys.exit(0)
    elif data == 'waveform-pos':
        filename = file_out_suff + '.basis.hdf5'
        if os.path.exists(filename):
            myfile = h5py.File(filename, 'r', libver='earliest')
            waveforms = myfile.get('waveform_pos')[:]
            myfile.close()
            return waveforms
        else:
            if comm.rank == 0:
                print_and_log(["The whitening step should be launched first!"], 'error', logger)
            sys.exit(0)
    elif data == 'waveforms-pos':
        filename = file_out_suff + '.basis.hdf5'
        if os.path.exists(filename):
            myfile = h5py.File(filename, 'r', libver='earliest')
            waveforms = myfile.get('waveforms_pos')[:]
            myfile.close()
            return waveforms
        else:
            if comm.rank == 0:
                print_and_log(["The whitening step should be launched first!"], 'error', logger)
            sys.exit(0)
    elif data == 'weights':
        filename = file_out_suff + '.basis.hdf5'
        if os.path.exists(filename):
            myfile = h5py.File(filename, 'r', libver='earliest')
            waveforms = myfile.get('waveforms')[:]
            myfile.close()
            u = numpy.median(waveforms, 0)
            tmp = numpy.median(numpy.abs(waveforms - u), 0)
            if params.getboolean('detection', 'alignment'):
                jitter_range = params.getint('detection', 'jitter_range')
                res = numpy.zeros(len(tmp) + 2 * jitter_range, dtype=numpy.float32)
                res[jitter_range:-jitter_range] = tmp
                res[:jitter_range] = tmp[0]
                res[-jitter_range:] = tmp[-1]
            else:
                res = tmp
            return res
        else:
            if comm.rank == 0:
                print_and_log(["The whitening step should be launched first!"], 'error', logger)
            sys.exit(0)
    elif data == 'weights-pos':
        filename = file_out_suff + '.basis.hdf5'
        if os.path.exists(filename):
            myfile = h5py.File(filename, 'r', libver='earliest')
            waveforms = myfile.get('waveforms_pos')[:]
            myfile.close()
            u = numpy.median(waveforms, 0)
            tmp = numpy.median(numpy.abs(waveforms - u), 0)
            if params.getboolean('detection', 'alignment'):
                jitter_range = params.getint('detection', 'jitter_range')
                res = numpy.zeros(len(tmp)+2*jitter_range, dtype=numpy.float32)
                res[jitter_range:-jitter_range] = tmp
                res[:jitter_range] = tmp[0]
                res[-jitter_range:] = tmp[-1]
            else:
                res = tmp
            return res
        else:
            if comm.rank == 0:
                print_and_log(["The whitening step should be launched first!"], 'error', logger)
            sys.exit(0)
    elif data == 'templates':
        filename = file_out_suff + '.templates%s.hdf5' % extension
        if os.path.exists(filename):
            myfile = h5py.File(filename, 'r', libver='earliest')
            temp_x = myfile.get('temp_x')[:].ravel()
            temp_y = myfile.get('temp_y')[:].ravel()
            temp_data = myfile.get('temp_data')[:].ravel()
            N_e, N_t, nb_templates = myfile.get('temp_shape')[:].ravel().astype(numpy.int32)
            myfile.close()
            return scipy.sparse.csc_matrix((temp_data, (temp_x, temp_y)), shape=(N_e * N_t, nb_templates))
        else:
            if comm.rank == 0:
                print_and_log(["No templates found! Check suffix?"], 'error', logger)
            sys.exit(0)
    elif data == 'nb_chances':
        filename = file_out_suff + '.templates%s.hdf5' % extension
        if os.path.exists(filename):
            myfile = h5py.File(filename, 'r', libver='earliest')
            if 'nb_chances' in myfile.keys():
                return myfile['nb_chances'][:]
            else:
                N_e, N_t, nb_templates = myfile.get('temp_shape')[:].ravel()
                nb_chances = params.getfloat('fitting', 'nb_chances')
                return nb_chances * numpy.ones(nb_templates//2, dtype=numpy.float32)
        else:
            if comm.rank == 0:
                print_and_log(["No templates found! Check suffix?"], 'error', logger)
            sys.exit(0)
    elif data == 'nb_templates':
        filename = file_out_suff + '.templates%s.hdf5' % extension
        if os.path.exists(filename):
            myfile = h5py.File(filename, 'r', libver='earliest')
            N_e, N_t, nb_templates = myfile.get('temp_shape')[:].ravel()
            myfile.close()
            return nb_templates
        else:
            if comm.rank == 0:
                print_and_log(["No templates found! Check suffix?"], 'error', logger)
            sys.exit(0)
    elif data == 'overlaps':
        filename = file_out_suff + '.overlap%s.hdf5' % extension
        if os.path.exists(filename):
            myfile = h5py.File(filename, 'r', libver='earliest')
            over_x = myfile.get('over_x')[:].ravel()
            over_y = myfile.get('over_y')[:].ravel()
            over_data = myfile.get('over_data')[:].ravel()
            over_shape = myfile.get('over_shape')[:].ravel()
            duration = over_shape[1] // 2
            myfile.close()

            c_overs = {}
            N_over = int(numpy.sqrt(over_shape[0]))

            res = []
            res2 = []
            for i in range(N_over):
                res += [i * N_over, (i + 1) * N_over]
                res2 += [i, i + 1]

            bounds = numpy.searchsorted(over_x, res, 'left')
            sub_over = numpy.mod(over_x, N_over)
            over_sorted = numpy.argsort(sub_over).astype(numpy.int32)

            bounds_2 = numpy.searchsorted(sub_over[over_sorted], res2, 'left')

            mask_duration = (over_y < duration)

            for i in range(N_over):

                xmin, xmax = bounds[2*i:2*(i+1)]
                local_x = over_x[xmin:xmax] - (i * N_over)
                local_y = over_y[xmin:xmax]
                local_data = over_data[xmin:xmax]

                xmin, xmax = bounds_2[2*i:2*(i+1)]
                nslice = over_sorted[xmin:xmax][mask_duration[over_sorted[xmin:xmax]]]

                local_x = numpy.concatenate((local_x, over_x[nslice] // N_over))
                local_y = numpy.concatenate((local_y, (over_shape[1] - 1) - over_y[nslice]))
                local_data = numpy.concatenate((local_data, over_data[nslice]))

                c_overs[i] = scipy.sparse.csr_matrix((local_data, (local_x, local_y)), shape=(N_over, over_shape[1]))

            del over_x, over_y, over_data, over_shape

            return c_overs
        else:
            if comm.rank == 0:
                print_and_log(["No overlaps found! Check suffix?"], 'error', logger)
            sys.exit(0)
    elif data == 'overlaps-raw':
        filename = file_out_suff + '.overlap%s.hdf5' % extension
        if os.path.exists(filename):
            myfile = h5py.File(filename, 'r', libver='earliest')
            over_x = myfile.get('over_x')[:].ravel()
            over_y = myfile.get('over_y')[:].ravel()
            over_data = myfile.get('over_data')[:].ravel()
            over_shape = myfile.get('over_shape')[:].ravel()
            over_sub = numpy.mod(over_x, int(numpy.sqrt(over_shape[0])))
            over_sorted = numpy.argsort(over_sub).astype(numpy.int32)
            myfile.close()
            return over_x, over_y, over_data, over_sub, over_sorted, over_shape
        else:
            if comm.rank == 0:
                print_and_log(["No overlaps found! Check suffix?"], 'error', logger)
            sys.exit(0)
    elif data == 'version':
        filename = file_out_suff + '.templates%s.hdf5' % extension
        if os.path.exists(filename):
            try:
                myfile = h5py.File(filename, 'r', libver='earliest')
                if myfile.get('version').dtype == numpy.dtype('S5'):
                    version = myfile.get('version')[0].decode('ascii')
                elif myfile.get('version').dtype == numpy.int32:
                    data = myfile.get('version')[:]
                    if len(data) == 3:
                        version = ".".join([str(int(i)) for i in data])
                    elif len(data) == 1:
                        version = ".".join([str(int(i)) for i in data[0]])
                myfile.close()
            except Exception:
                version = None
            return version
        else:
            if comm.rank == 0:
                print_and_log(["No templates found! Check suffix?"], 'error', logger)
            sys.exit(0)
    elif data == 'norm-templates':
        if os.path.exists(file_out_suff + '.templates%s.hdf5' % extension):
            myfile = h5py.File(file_out_suff + '.templates%s.hdf5' % extension, 'r', libver='earliest')
            norms = myfile.get('norms')[:]
            myfile.close()
            return norms
        else:
            if comm.rank == 0:
                print_and_log(["No norms found! Check suffix?"], 'error', logger)
            sys.exit(0)
    elif data == 'purity':
        if os.path.exists(file_out_suff + '.templates%s.hdf5' % extension):
            myfile = h5py.File(file_out_suff + '.templates%s.hdf5' % extension, 'r', libver='earliest')
            if 'purity' in myfile.keys():
                purity = myfile.get('purity')[:]
            else:
                N_e, N_t, nb_templates = myfile.get('temp_shape')[:].ravel()
                purity = numpy.zeros(nb_templates//2, dtype=numpy.float32)
            myfile.close()
            return purity
        else:
            if comm.rank == 0:
                print_and_log(["No templates found! Check suffix?"], 'error', logger)
            sys.exit(0)
    elif data == 'maxoverlap':
        if os.path.exists(file_out_suff + '.templates%s.hdf5' % extension):
            myfile = h5py.File(file_out_suff + '.templates%s.hdf5' % extension, 'r', libver='earliest')
            maxoverlap = myfile.get('maxoverlap')[:]
            myfile.close()
            return maxoverlap
        else:
            if comm.rank == 0:
                print_and_log(["No templates found! Check suffix?"], 'error', logger)
            sys.exit(0)
    elif data == 'supports':
        if os.path.exists(file_out_suff + '.templates%s.hdf5' % extension):
            myfile = h5py.File(file_out_suff + '.templates%s.hdf5' % extension, 'r', libver='earliest')
            supports = myfile.get('supports')[:]
            myfile.close()
            return supports
        else:
            if comm.rank == 0:
                print_and_log(["No supports found! Check suffix?"], 'error', logger)
            sys.exit(0)
    elif data == 'common-supports':
        if os.path.exists(file_out_suff + '.templates%s.hdf5' % extension):
            myfile = h5py.File(file_out_suff + '.templates%s.hdf5' % extension, 'r', libver='earliest')
            supports = myfile.get('supports')[:]
            myfile.close()
            nb_temp = len(supports)
            nb_elec = supports.shape[1]
            res = numpy.ones((nb_temp, nb_temp), dtype=numpy.float32)
            for i in range(nb_temp):
                res[i] = numpy.sum(numpy.logical_and(supports[i], supports), 1)
            return res
    elif data == 'spike-cluster':
        filename = params.get('data', 'data_file_noext') + '.spike-cluster.hdf5'
        if os.path.exists(filename):
            myfile = h5py.File(filename, 'r', libver='earliest')
            clusters = myfile.get('clusters')[:].ravel()
            N_clusters = len(numpy.unique(clusters))
            spiketimes = myfile.get('spikes')[:].ravel()
            myfile.close()
            return clusters, spiketimes, N_clusters
        else:
            raise Exception('Need to provide a spike-cluster file!')
    elif data == 'clusters':
        filename = file_out_suff + '.clusters%s.hdf5' % extension
        if os.path.exists(filename):
            myfile = h5py.File(filename, 'r', libver='earliest')
            result = {}
            for key in myfile.keys():
                result[str(key)] = myfile.get(key)[:]
            myfile.close()
            return result
        else:
            raise Exception('No clusters found! Check suffix or run clustering?')
    elif data == 'clusters-nodata':
        filename = file_out_suff + '.clusters%s.hdf5' % extension
        if os.path.exists(filename):
            myfile = h5py.File(filename, 'r', libver='earliest')
            result = {}
            for key in myfile.keys():
                if (key.find('data') == -1):
                    result[str(key)] = myfile.get(key)[:]
            myfile.close()
            return result
        else:
            raise Exception('No clusters found! Check suffix or run clustering?')
    elif data == 'clusters-light':
        filename = file_out_suff + '.clusters%s.hdf5' % extension
        if os.path.exists(filename):
            myfile = h5py.File(filename, 'r', libver='earliest')
            result = {}
            for key in myfile.keys():
                if ('clusters_' in key) or (key == 'electrodes'):
                    result[str(key)] = myfile.get(key)[:]
            myfile.close()
            return result
        else:
            raise Exception('No clusters found! Check suffix or run clustering?')
    elif data == 'electrodes':
        filename = file_out_suff + '.clusters%s.hdf5' % extension
        if os.path.exists(filename):
            myfile = h5py.File(filename, 'r', libver='earliest')
            electrodes = myfile.get('electrodes')[:].ravel().astype(numpy.int32)
            myfile.close()
            return electrodes
        else:
            raise Exception('No clusters found! Check suffix or run clustering?')
    elif data == 'results':
        try:
            return get_results(params, extension)
        except Exception:
            raise Exception('No results found! Check suffix or run the fitting?')
    elif data == 'mua':
        try:
            return get_mua(params, extension)
        except Exception:
            raise Exception('No MUA found! Check suffix or run the fitting?')
    elif data == 'duration':
        try:
            return get_duration(params, extension)
        except Exception:
            raise Exception('No results found! Check suffix or run the fitting?')
    elif data == 'garbage':
        try:
            return get_garbage(params, extension)
        except Exception:
            raise Exception('No results found! Check suffix or run the fitting?')
    elif data == 'overlaps':
        try:
            return get_overlaps(params, extension)
        except Exception:
            raise Exception('No overlaps found! Check suffix or run the fitting?')
    elif data == 'limits':
        myfile = file_out_suff + '.templates%s.hdf5' % extension
        if os.path.exists(myfile):
            myfile = h5py.File(myfile, 'r', libver='earliest')
            limits = myfile.get('limits')[:]
            myfile.close()
            return limits
        else:
            raise Exception('No templates found! Check suffix?')
    elif data == 'injected_spikes':
        try:
            spikes = h5py.File(data_file_noext + '/injected/result.hdf5').get('spiketimes')
            elecs = numpy.load(data_file_noext + '/injected/elecs.npy')
            N_tm = len(spikes)
            count = 0
            result = {}
            for i in range(N_tm):
                key = 'temp_' + str(i)
                if len(spikes[key]) > 0:
                    result['spikes_' + str(elecs[count])] = spikes[key]
                    count += 1
            return result
        except Exception:
            return None
    elif data == 'triggers':
        filename = file_out_suff + '.triggers%s.npy' % extension
        if os.path.exists(filename):
            triggers = numpy.load(filename)
            N_tr = triggers.shape[0]

            data_file = params.data_file
            data_file.open()

            N_total = params.nb_channels
            N_t = params.getint('detection', 'N_t')

            template_shift = params.getint('detection', 'template_shift')

            spikes = numpy.zeros((N_t, N_total, N_tr))
            for (count, idx) in enumerate(triggers):
                spikes[:, :, count] = data_file.get_snippet(idx - template_shift, N_t)
            data_file.close()
            return triggers, spikes
        else:
            raise Exception('No triggers found! Check suffix or check if file `%s` exists?' % filename)
    elif data == 'juxta-mad':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='earliest')
            try:
                juxta_mad = beer_file.get('juxta_mad').value
            finally:
                beer_file.close()
            return juxta_mad
        else:
            raise Exception('No median absolute deviation found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data == 'juxta-triggers':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='earliest')
            try:
                juxta_spike_times = beer_file.get('juxta_spiketimes/elec_0')[:]
            finally:
                beer_file.close()
            return juxta_spike_times
        else:
            raise Exception('No triggers found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data == 'juxta-values':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='earliest')
            try:
                juxta_spike_values = beer_file.get('juxta_spike_values/elec_0')[:]
            finally:
                beer_file.close()
            return juxta_spike_values
        else:
            raise Exception('No values found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data == 'extra-mads':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='earliest')
            try:
                extra_mads = beer_file.get('extra_mads')[:]
            finally:
                beer_file.close()
            return extra_mads
        else:
            raise Exception('No median absolute deviation found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data == 'extra-triggers':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='earliest')
            N_e = params.getint('data', 'N_e')
            extra_spike_times = N_e * [None]
            try:
                for e in range(0, N_e):
                    key = "extra_spiketimes/elec_{}".format(e)
                    extra_spike_times[e] = beer_file.get(key)[:]
            finally:
                beer_file.close()
            return extra_spike_times
        else:
            raise Exception('No triggers found! Check if file `{}` exists?'.format(filename))
    elif data == 'extra-values':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='earliest')
            N_e = params.getint('data', 'N_e')
            extra_spike_values = N_e * [None]
            try:
                for e in range(0, N_e):
                    key = "extra_spike_values/elec_{}".format(e)
                    extra_spike_values[e] = beer_file.get(key)[:]
            finally:
                beer_file.close()
            return extra_spike_values
        else:
            raise Exception('No values found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data == 'class-weights':
        filename = file_out_suff + '.beer.hdf5'
        if os.path.exists(filename):
            bfile = h5py.File(filename, 'r', libver='earliest')
            class_weights = bfile.get('class-weights')[:]
            bfile.close()
            return class_weights
        else:
            raise Exception('No class weights found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data == 'confusion-matrices':
        filename = file_out_suff + '.beer.hdf5'
        if os.path.exists(filename):
            bfile = h5py.File(filename, 'r', libver='earliest')
            confusion_matrices = bfile.get('confusion_matrices')[:]
            bfile.close()
            return confusion_matrices
        else:
            raise Exception('No confusion matrices found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data == 'proportion':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='earliest')
            try:
                proportion = beer_file.get('proportion').value
            finally:
                beer_file.close()
            return proportion
        else:
            raise Exception('No proportion found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data == 'threshold-false-negatives':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='earliest')
            try:
                threshold_false_negatives = beer_file.get('thresh_fn').value
            finally:
                beer_file.close()
            return threshold_false_negatives
        else:
            raise Exception('No threshold false negatives found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data in ['false-positive-rates', 'true-positive-rates',
                  'false-positive-error-rates', 'false-negative-error-rates']:
        # Retrieve saved data.
        confusion_matrices = load_data(params, 'confusion-matrices', extension)
        threshold_false_negatives = load_data(params, 'threshold-false-negatives', extension)
        # Correct counts of false negatives.
        for confusion_matrix in confusion_matrices:
            confusion_matrix[0, 1] += threshold_false_negatives
        # Compute the wanted statistics.
        if data == 'false-positive-rates':
            # Compute false positive rates (i.e. FP / (FP + TN)).
            results = [M[1, 0] / (M[1, 0] + M[1, 1]) for M in confusion_matrices]
            # Add false positive rate endpoints.
            results = [1.0] + results + [0.0]
        if data == 'true-positive-rates':
            # Compute true positive rates (i.e. TP / (TP + FN)).
            results = [M[0, 0] / (M[0, 0] + M[0, 1]) for M in confusion_matrices]
            # Add true positive rate endpoints.
            results = [1.0] + results + [0.0]
        if data == 'false-positive-error-rates':
            # Compute false positive error rates (i.e. FP / (TP + FP)).
            results = [M[1, 0] / (M[0, 0] + M[1, 0]) for M in confusion_matrices]
            # Add false positive error rate endpoints.
            results = [1.0] + results + [0.0]
        if data == 'false-negative-error-rates':
            # Compute false negative error rates (i.e. FN / (TP + FN)).
            results = [M[0, 1] / (M[0, 0] + M[0, 1]) for M in confusion_matrices]
            # Add false negative error rate endpoints.
            results = [0.0] + results + [1.0]
        results = numpy.array(results, dtype=numpy.float)
        return results
    elif data == 'sc-contingency-matrices':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='earliest')
            try:
                sc_contingency_matrices = beer_file.get('sc_contingency_matrices')[:]
            finally:
                beer_file.close()
            return sc_contingency_matrices
        else:
            raise Exception('No contingency matrices found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data in ['sc-false-positive-error-rates', 'sc-false-negative-error-rates']:
        # Retrieve saved data.
        sc_contingency_matrices = load_data(params, 'sc-contingency-matrices', extension)
        threshold_false_negatives = load_data(params, 'threshold-false-negatives', extension)
        # Correct counts of false negatives.
        for sc_contingency_matrix in sc_contingency_matrices:
            sc_contingency_matrix[0, 1] += threshold_false_negatives
        # Compute the wanted statistics.
        if data == 'sc-false-positive-error-rates':
            # Compute false positive error rates.
            results = [float(M[1, 1]) / float(M[1, 0] + M[1, 1]) if 0 < M[1, 0] + M[1, 1] else 0.0
                       for M in sc_contingency_matrices]
        if data == 'sc-false-negative-error-rates':
            # Compute false negative error rates.
            results = [float(M[0, 1]) / float(M[0, 0] + M[0, 1]) if 0 < M[0, 0] + M[0, 1] else 0.0
                       for M in sc_contingency_matrices]
        results = numpy.array(results, dtype=numpy.float)
        return results
    elif data == 'sc-contingency-matrix':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='earliest')
            try:
                sc_contingency_matrix = beer_file.get('sc_contingency_matrix')[:]
            finally:
                beer_file.close()
            return sc_contingency_matrix
        else:
            raise Exception('No contingency matrix found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data in ['sc-best-false-positive-error-rate', 'sc-best-false-negative-error-rate']:
        sc_contingency_matrix = load_data(params, 'sc-contingency-matrix', extension)
        threshold_false_negatives = load_data(params, 'threshold-false-negatives', extension)
        # Correct count of false negatives.
        sc_contingency_matrix[0, 1] += threshold_false_negatives
        # Compute the wanted statistics.
        if data == 'sc-best-false-positive-error-rate':
            # Compute best false positive error rate.
            M = sc_contingency_matrix
            result = float(M[1, 1]) / float(M[1, 0] + M[1, 1]) if 0 < M[1, 0] + M[1, 1] else 0.0
        if data == 'sc-best-false-negative-error-rate':
            # Compute best false negative error rate.
            M = sc_contingency_matrix
            result = float(M[0, 1]) / float(M[0, 0] + M[0, 1]) if 0 < M[0, 0] + M[0, 1] else 0.0
        return result
    elif data == 'selection':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='earliest')
            try:
                selection = beer_file.get('selection')[:]
            finally:
                beer_file.close()
            return selection
        else:
            raise Exception('No selection found! Check suffix or check if file `{}` exists?'.format(filename))


def write_datasets(h5file, to_write, result, electrode=None, compression=False):
    for key in to_write:
        if electrode is not None:
            mykey = key + str(electrode)
        else:
            mykey = key
        if compression:
            h5file.create_dataset(mykey, data=result[mykey], chunks=True, compression='gzip')
        else:
            h5file.create_dataset(mykey, data=result[mykey], chunks=True)


def collect_data(nb_threads, params, erase=False, with_real_amps=False, with_voltages=False, benchmark=False):

    # Retrieve the key parameters.
    data_file = params.data_file
    N_e = params.getint('data', 'N_e')
    N_t = params.getint('detection', 'N_t')
    file_out_suff = params.get('data', 'file_out_suff')
    max_chunk = params.getfloat('fitting', 'max_chunk')
    hdf5_compress = params.getboolean('data', 'hdf5_compress')
    data_length = data_stats(params, show=False)
    duration = data_length
    templates = load_data(params, 'norm-templates')
    refractory = params.getint('fitting', 'refractory')
    N_tm = len(templates)
    collect_all = params.getboolean('fitting', 'collect_all')
    debug = params.getboolean('fitting', 'debug')

    print_and_log(["Gathering spikes from %d nodes..." % nb_threads], 'default', logger)

    # Initialize data collection.
    result = {
        'spiketimes': {},
        'amplitudes': {},
        'info': {
            'duration': numpy.array([duration], dtype=numpy.uint64)
        }
    }
    if with_real_amps:
        result['real_amps'] = {}
    if with_voltages:
        result['voltages'] = {}
    if collect_all:
        result['gspikes'] = {}
        result['gtemps'] = {}

    for i in range(N_tm // 2):
        result['spiketimes']['temp_' + str(i)] = [numpy.empty(shape=0, dtype=numpy.uint32)]
        result['amplitudes']['temp_' + str(i)] = [numpy.empty(shape=(0, 2), dtype=numpy.float32)]
        if with_real_amps:
            result['real_amps']['temp_' + str(i)] = [numpy.empty(shape=0, dtype=numpy.float32)]
        if with_voltages:
            result['voltages']['temp_' + str(i)] = [numpy.empty(shape=0, dtype=numpy.float32)]

    if collect_all:
        for i in range(N_e):
            result['gspikes']['elec_' + str(i)] = [numpy.empty(shape=0, dtype=numpy.uint32)]

    if debug:
        result_debug = {
            'chunk_nbs': [numpy.empty(shape=0, dtype=numpy.uint32)],
            'iteration_nbs': [numpy.empty(shape=0, dtype=numpy.uint32)],
            'peak_nbs': [numpy.empty(shape=0, dtype=numpy.uint32)],
            'peak_local_time_steps': [numpy.empty(shape=0, dtype=numpy.uint32)],
            'peak_time_steps': [numpy.empty(shape=0, dtype=numpy.uint32)],
            'peak_scalar_products': [numpy.empty(shape=0, dtype=numpy.float32)],
            'peak_solved_flags': [numpy.empty(shape=0, dtype=numpy.float32)],
            'template_nbs': [numpy.empty(shape=0, dtype=numpy.uint32)],
            'success_flags': [numpy.empty(shape=0, dtype=numpy.bool)],
        }
    else:
        result_debug = None

    to_explore = range(nb_threads)

    if comm.rank == 0:
        to_explore = get_tqdm_progressbar(params, to_explore)

    # For each thread/process collect data.
    for count, node in enumerate(to_explore):
        spiketimes_file = file_out_suff + '.spiketimes-%d.data' % node
        amplitudes_file = file_out_suff + '.amplitudes-%d.data' % node
        templates_file = file_out_suff + '.templates-%d.data' % node
        if with_real_amps:
            real_amps_file = file_out_suff + '.real_amps-%d.data' % node
            real_amps = numpy.fromfile(real_amps_file, dtype=numpy.float32)
        if with_voltages:
            voltages_file = file_out_suff + '.voltages-%d.data' % node
            voltages = numpy.fromfile(voltages_file, dtype=numpy.float32)

        if collect_all:
            gspikes_file = file_out_suff + '.gspiketimes-%d.data' % node
            gspikes = numpy.fromfile(gspikes_file, dtype=numpy.uint32)
            gtemps_file = file_out_suff + '.gtemplates-%d.data' % node
            gtemps = numpy.fromfile(gtemps_file, dtype=numpy.uint32)

        if os.path.exists(amplitudes_file):

            amplitudes = numpy.fromfile(amplitudes_file, dtype=numpy.float32)
            spiketimes = numpy.fromfile(spiketimes_file, dtype=numpy.uint32)
            templates = numpy.fromfile(templates_file, dtype=numpy.uint32)
            N = len(amplitudes)
            amplitudes = amplitudes.reshape(N // 2, 2)
            min_size = min([amplitudes.shape[0], spiketimes.shape[0], templates.shape[0]])
            amplitudes = amplitudes[:min_size]
            spiketimes = spiketimes[:min_size]
            templates = templates[:min_size]
            if with_real_amps:
                real_amps = real_amps[:min_size]
            if with_voltages:
                voltages = voltages[:min_size]

            local_temp = numpy.unique(templates)

            for j in local_temp:
                idx = numpy.where(templates == j)[0]
                result['amplitudes']['temp_' + str(j)].append(amplitudes[idx])
                result['spiketimes']['temp_' + str(j)].append(spiketimes[idx])
                if with_real_amps:
                    result['real_amps']['temp_' + str(j)].append(real_amps[idx])
                if with_voltages:
                    result['voltages']['temp_' + str(j)].append(voltages[idx])

            if collect_all:
                for j in range(N_e):
                    idx = numpy.where(gtemps == j)[0]
                    result['gspikes']['elec_' + str(j)].append(gspikes[idx])

        if debug:
            for (key, filename_formatter, dtype) in [
                ('chunk_nbs', '.chunk_nbs_debug_%d.data', numpy.uint32),
                ('iteration_nbs', '.iteration_nbs_debug_%d.data', numpy.uint32),
                ('peak_nbs', '.peak_nbs_debug_%d.data', numpy.uint32),
                ('peak_local_time_steps', '.peak_local_time_steps_debug_%d.data', numpy.uint32),
                ('peak_time_steps', '.peak_time_steps_debug_%d.data', numpy.uint32),
                ('peak_scalar_products', '.peak_scalar_products_debug_%d.data', numpy.float32),
                ('peak_solved_flags', '.peak_solved_flags_debug_%d.data', numpy.float32),
                ('template_nbs', '.template_nbs_debug_%d.data', numpy.uint32),
                ('success_flags', '.success_flags_debug_%d.data', numpy.bool),
            ]:
                filename = file_out_suff + filename_formatter % node
                data = numpy.fromfile(filename, dtype=dtype)
                result_debug[key].append(data)
                # TODO avoid multiple concatenations (i.e. copies)?

    sys.stderr.flush()

    for key in result['spiketimes']:
        result['spiketimes'][key] = numpy.concatenate(result['spiketimes'][key]).astype(numpy.uint32)
        result['amplitudes'][key] = numpy.concatenate(result['amplitudes'][key]).astype(numpy.float32)

        idx = numpy.argsort(result['spiketimes'][key])
        result['spiketimes'][key] = result['spiketimes'][key][idx]
        result['amplitudes'][key] = result['amplitudes'][key][idx]
        if with_real_amps:
            result['real_amps'][key] = numpy.concatenate(result['real_amps'][key]).astype(numpy.float32)
            result['real_amps'][key] = result['real_amps'][key][idx]
        if with_voltages:
            result['voltages'][key] = numpy.concatenate(result['voltages'][key]).astype(numpy.float32)
            result['voltages'][key] = result['voltages'][key][idx]

        if refractory > 0:
            violations = numpy.where(numpy.diff(result['spiketimes'][key]) <= refractory)[0] + 1
            result['spiketimes'][key] = numpy.delete(result['spiketimes'][key], violations)
            result['amplitudes'][key] = numpy.delete(result['amplitudes'][key], violations, axis=0)
            if with_real_amps:
                result['real_amps'][key] = numpy.delete(result['real_amps'][key], violations)
            if with_voltages:
                result['voltages'][key] = numpy.delete(result['voltages'][key], violations)

    if collect_all:
        for key in result['gspikes']:
            result['gspikes'][key] = numpy.concatenate(result['gspikes'][key]).astype(numpy.uint32)
            idx = numpy.argsort(result['gspikes'][key])
            result['gspikes'][key] = result['gspikes'][key][idx]

    keys = ['spiketimes', 'amplitudes', 'info']
    if with_real_amps:
        keys += ['real_amps']
    if with_voltages:
        keys += ['voltages']
    if collect_all:
        keys += ['gspikes']

    # Save results into `<dataset>/<dataset>.result.hdf5`.
    mydata = h5py.File(file_out_suff + '.result.hdf5', mode='w', libver='earliest')
    for key in keys:
        mydata.create_group(key)
        for temp in result[key].keys():
            tmp_path = '%s/%s' % (key, temp)
            if hdf5_compress:
                mydata.create_dataset(tmp_path, data=result[key][temp], compression='gzip')
            else:
                mydata.create_dataset(tmp_path, data=result[key][temp])
    mydata.close()

    if debug:
        # Save debug data to debug HDF5 files.
        file = h5py.File(file_out_suff + '.result_debug.hdf5', mode='w', libver='earliest')
        names = [
            'chunk_nbs',
            'iteration_nbs',
            'peak_nbs',
            'peak_local_time_steps',
            'peak_time_steps',
            'peak_scalar_products',
            'peak_solved_flags',
            'template_nbs',
            'success_flags',
        ]
        for name in names:
            data = numpy.concatenate(result_debug[name])
            compression = 'gzip' if hdf5_compress else None
            file.create_dataset(name, data=data, compression=compression)
        file.close()

    # Count the number of spikes.
    count = 0
    for item in result['spiketimes'].keys():
        count += len(result['spiketimes'][item])
    if collect_all:
        gcount = 0
        for item in result['gspikes'].keys():
            gcount += len(result['gspikes'][item])

    # Print log message.
    if benchmark:
        to_print = "injected"
    else:
        to_print = "fitted"
    to_write = ["Number of spikes %s : %d" % (to_print, count)]
    if collect_all:
        to_write += ["Number of spikes not fitted (roughly): %d [%g percent]" % (gcount, 100 * gcount / float(count))]
    print_and_log(to_write, 'info', logger)

    if erase:
        purge(file_out_suff, '.data')


def get_accurate_thresholds(params, spike_thresh_min=1):

    thresholds = load_data(params, 'thresholds')

    if spike_thresh_min < 1:
        mads = load_data(params, 'mads')
        templates = load_data(params, 'templates')
        sign = params.get('detection', 'peaks')
        N_e = params.getint('data', 'N_e')
        N_t = params.getint('detection', 'N_t')
        amplitudes = load_data(params, 'limits')
        nb_temp = templates.shape[1] // 2
        spike_thresh = params.getfloat('detection', 'spike_thresh') * spike_thresh_min

        for idx in range(nb_temp):
            template = templates[:, idx].toarray().ravel().reshape(N_e, N_t)
            if sign == 'negative':
                a, b = numpy.unravel_index(template.argmin(), template.shape)
                value = -template[a, b]
            elif sign == 'positive':
                a, b = numpy.unravel_index(template.argmax(), template.shape)
                value = template[a, b]
            elif sign == 'both':
                a, b = numpy.unravel_index(numpy.abs(template).argmax(), template.shape)
                value = numpy.abs(template)[a, b]

            if thresholds[a] > value:
                thresholds[a] = max(spike_thresh * mads[a], value)

    return thresholds


def collect_mua(nb_threads, params, erase=False):

    # Retrieve the key parameters.
    data_file = params.data_file
    N_e = params.getint('data', 'N_e')
    N_t = params.getint('detection', 'N_t')
    file_out_suff = params.get('data', 'file_out_suff')
    max_chunk = params.getfloat('fitting', 'max_chunk')
    hdf5_compress = params.getboolean('data', 'hdf5_compress')
    data_length = data_stats(params, show=False)
    duration = int(data_length)
    print_and_log(["Gathering MUA from %d nodes..." % nb_threads], 'default', logger)

    # Initialize data collection.
    result = {
        'spiketimes': {},
        'amplitudes': {},
        'info': {
            'duration': numpy.array([duration], dtype=numpy.uint64)
        }
    }

    for i in range(N_e):
        result['spiketimes']['elec_' + str(i)] = [numpy.empty(shape=0, dtype=numpy.uint32)]
        result['amplitudes']['elec_' + str(i)] = [numpy.empty(shape=0, dtype=numpy.float32)]

    to_explore = range(nb_threads)

    if comm.rank == 0:
        to_explore = get_tqdm_progressbar(params, to_explore)

    # For each thread/process collect data.
    for count, node in enumerate(to_explore):
        spiketimes_file = file_out_suff + '.mua-%d.data' % node
        templates_file = file_out_suff + '.elec-%d.data' % node
        amplitudes_file = file_out_suff + '.amp-%d.data' % node

        if os.path.exists(templates_file):

            spiketimes = numpy.fromfile(spiketimes_file, dtype=numpy.uint32)
            templates = numpy.fromfile(templates_file, dtype=numpy.uint32)
            amplitudes = numpy.fromfile(amplitudes_file, dtype=numpy.float32)
            min_size = min([spiketimes.shape[0], templates.shape[0], amplitudes.shape[0]])
            spiketimes = spiketimes[:min_size]
            templates = templates[:min_size]
            amplitudes = amplitudes[:min_size]
            local_temp = numpy.unique(templates)

            for j in local_temp:
                idx = numpy.where(templates == j)[0]
                result['spiketimes']['elec_' + str(j)].append(spiketimes[idx])
                result['amplitudes']['elec_' + str(j)].append(amplitudes[idx])

    sys.stderr.flush()
    # TODO: find a programmer comment.
    for key in result['spiketimes']:
        result['spiketimes'][key] = numpy.concatenate(result['spiketimes'][key]).astype(numpy.uint32)
        result['amplitudes'][key] = numpy.concatenate(result['amplitudes'][key]).astype(numpy.float32)
        idx = numpy.argsort(result['spiketimes'][key])
        result['spiketimes'][key] = result['spiketimes'][key][idx]
        result['amplitudes'][key] = result['amplitudes'][key][idx]

    # Save results into `<dataset>/<dataset>.result.hdf5`.
    mydata = h5py.File(file_out_suff + '.mua.hdf5', 'w', libver='earliest')
    keys = ['spiketimes', 'amplitudes', 'info']
    for key in keys:
        mydata.create_group(key)
        for temp in result[key].keys():
            tmp_path = '%s/%s' % (key, temp)
            if hdf5_compress:
                mydata.create_dataset(tmp_path, data=result[key][temp], compression='gzip')
            else:
                mydata.create_dataset(tmp_path, data=result[key][temp])
    mydata.close()

    # Count and print the number of spikes.
    count = 0
    for item in result['spiketimes'].keys():
        count += len(result['spiketimes'][item])

    to_write = ["Number of threshold crossings : %d" % count]

    print_and_log(to_write, 'info', logger)

    if erase:
        purge(file_out_suff, '.data')


def get_results(params, extension=''):
    file_out_suff = params.get('data', 'file_out_suff')
    result = {}
    myfile = h5py.File(file_out_suff + '.result%s.hdf5' % extension, 'r', libver='earliest')
    for key in ['spiketimes', 'amplitudes']:
        result[str(key)] = {}
        for temp in myfile.get(key).keys():
            result[str(key)][str(temp)] = myfile.get(key).get(temp)[:]
            # Files has been saved with MATLAB and we need to be compatible with the default format
            if extension not in ['', '-merged'] and key == "spiketimes":
                result[str(key)][str(temp)] = result[str(key)][str(temp)].flatten()
    myfile.close()
    return result


def get_mua(params, extension=''):
    file_out_suff = params.get('data', 'file_out_suff')
    result = {}
    myfile = h5py.File(file_out_suff + '.mua%s.hdf5' % extension, 'r', libver='earliest')
    for key in ['spiketimes', 'amplitudes']:
        result[str(key)] = {}
        for temp in myfile.get(key).keys():
            result[str(key)][str(temp)] = myfile.get(key).get(temp)[:]
    myfile.close()
    return result


def get_duration(params, extension=''):
    file_out_suff = params.get('data', 'file_out_suff')
    result = {}
    myfile = h5py.File(file_out_suff + '.result%s.hdf5' % extension, 'r', libver='earliest')
    duration = myfile['info']['duration'][0]
    myfile.close()
    return duration


def get_garbage(params, extension=''):
    file_out_suff = params.get('data', 'file_out_suff')
    result = {}
    myfile = h5py.File(file_out_suff + '.result%s.hdf5' % extension, 'r', libver='earliest')
    for key in ['gspikes']:
        result[str(key)] = {}
        for temp in myfile.get(key).keys():
            result[str(key)][str(temp)] = myfile.get(key).get(temp)[:]
    myfile.close()
    return result


def get_overlaps(
        params, extension='', erase=False, normalize=True, maxoverlap=True,
        verbose=True, half=False, use_gpu=False, nb_cpu=1, nb_gpu=0, decimation=False
):

    parallel_hdf5 = get_parallel_hdf5_flag(params)
    data_file = params.data_file
    N_e = params.getint('data', 'N_e')
    N_t = params.getint('detection', 'N_t')
    hdf5_compress = params.getboolean('data', 'hdf5_compress')
    blosc_compress = params.getboolean('data', 'blosc_compress')
    N_total = params.nb_channels
    file_out_suff = params.get('data', 'file_out_suff')
    tmp_path = os.path.join(os.path.abspath(params.get('data', 'data_file_noext')), 'tmp')
    filename = file_out_suff + '.overlap%s.hdf5' % extension
    duration = 2 * N_t - 1
    n_scalar = N_e * N_t

    if os.path.exists(filename) and not erase:
        return h5py.File(filename, 'r')
    else:
        if os.path.exists(filename) and erase and (comm.rank == 0):
            os.remove(filename)

    SHARED_MEMORY = get_shared_memory_flag(params)

    if maxoverlap:
        if SHARED_MEMORY:
            templates, mpi_memory_1 = load_data_memshared(params, 'templates', extension=extension, normalize=normalize)
        else:
            templates = load_data(params, 'templates', extension=extension)
    else:
        if SHARED_MEMORY:
            templates, mpi_memory_1 = load_data_memshared(params, 'templates', normalize=normalize)
        else:
            templates = load_data(params, 'templates')

    if extension == '-merged':
        best_elec = load_data(params, 'electrodes', extension)
    else:
        best_elec = load_data(params, 'electrodes')
    nodes, edges = get_nodes_and_edges(params)
    N, N_tm = templates.shape

    norm_templates = load_data(params, 'norm-templates')

    if not SHARED_MEMORY and normalize:
        for idx in range(N_tm):
            myslice = numpy.arange(templates.indptr[idx], templates.indptr[idx+1])
            templates.data[myslice] /= norm_templates[idx]

    if half:
        N_tm //= 2

    comm.Barrier()
    inv_nodes = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.arange(len(nodes))

    if use_gpu:
        import cudamat as cmt
        if parallel_hdf5:
            if nb_gpu > nb_cpu:
                gpu_id = int(comm.rank // nb_cpu)
            else:
                gpu_id = 0
        else:
            gpu_id = 0
        cmt.cuda_set_device(gpu_id)
        cmt.init()
        cmt.cuda_sync_threads()

    all_delays = numpy.arange(1, N_t + 1)

    if half:
        upper_bounds = N_tm
    else:
        upper_bounds = N_tm // 2

    to_explore = range(comm.rank, N_e, comm.size)

    if comm.rank == 0:
        if verbose:
            print_and_log(["Pre-computing the overlaps of templates..."], 'default', logger)
        to_explore = get_tqdm_progressbar(params, to_explore)

    overlaps = {}
    overlaps['x'] = [numpy.zeros(0, dtype=numpy.uint32)]
    overlaps['y'] = [numpy.zeros(0, dtype=numpy.uint32)]
    overlaps['data'] = [numpy.zeros(0, dtype=numpy.float32)]
    overlaps['steps'] = []
    rows = numpy.arange(N_e*N_t)
    _srows = {'left': {}, 'right': {}}

    if decimation:
        nb_delays = len(all_delays) // 10
        indices = list(range(nb_delays)) + list(range(nb_delays, len(all_delays) - nb_delays, 3)) + list(range(len(all_delays) - nb_delays, len(all_delays)))
        all_delays = all_delays[indices]

    for idelay in all_delays:
        _srows['left'][idelay] = numpy.where(rows % N_t < idelay)[0]
        _srows['right'][idelay] = numpy.where(rows % N_t >= (N_t - idelay))[0]

    for ielec in to_explore:

        local_idx = numpy.where(best_elec == ielec)[0]
        len_local = len(local_idx)

        if not half:
            local_idx = numpy.concatenate((local_idx, local_idx + upper_bounds))

        if len_local > 0:

            to_consider = numpy.arange(upper_bounds)
            if not half:
                to_consider = numpy.concatenate((to_consider, to_consider + upper_bounds))

            loc_templates = templates[:, local_idx].tocsr()
            loc_templates2 = templates[:, to_consider].tocsr()

            for idelay in all_delays:

                tmp_1 = loc_templates[_srows['left'][idelay]].T.tocsr()
                tmp_2 = loc_templates2[_srows['right'][idelay]]

                if use_gpu:
                    tmp_1 = cmt.SparseCUDAMatrix(tmp_1, copy_on_host=False)
                    tmp_2 = cmt.CUDAMatrix(tmp_2.toarray(), copy_on_host=False)
                    data = cmt.sparse_dot(tmp_1, tmp_2).asarray()
                else:
                    data = tmp_1.dot(tmp_2)

                dx, dy = data.nonzero()
                ddx = numpy.take(local_idx, dx).astype(numpy.uint32)
                ddy = numpy.take(to_consider, dy).astype(numpy.uint32)
                ones = numpy.ones(len(dx), dtype=numpy.uint32)
                overlaps['x'].append(ddx*N_tm + ddy)
                overlaps['y'].append((idelay - 1)*ones)
                overlaps['data'].append(data.data)
                #if idelay < N_t:
                #    overlaps['x'].append(ddy*N_tm + ddx)
                #    overlaps['y'].append((duration - idelay)*ones)
                #    overlaps['data'].append(data.data)

    sys.stderr.flush()
    if comm.rank == 0:
        print_and_log(["Overlaps computed, now gathering data by MPI"], 'debug', logger)

    comm.Barrier()

    if comm.rank == 0:
        hfile = h5py.File(filename, 'w', libver='earliest')
        over_shape = numpy.array([N_tm**2, duration], dtype=numpy.int32)
        hfile.create_dataset('over_shape', data=over_shape)

    for key in ['x', 'y', 'data']:
        data = numpy.concatenate(overlaps.pop(key))
        if key in ['x', 'y']:
            data = gather_array(data, comm, dtype='uint32', compress=blosc_compress)
        else:
            data = gather_array(data, comm, dtype='float32')

        # We sort by x indices for faster retrieval later
        if comm.rank == 0:
            if key == 'x':
                indices = numpy.argsort(data).astype(numpy.int32)
            data = data[indices]

            if hdf5_compress:
                hfile.create_dataset('over_%s' %key, data=data, compression='gzip')
            else:
                hfile.create_dataset('over_%s' %key, data=data)
        del data

    # We need to gather the sparse arrays.
    if comm.rank == 0:
        del indices
        hfile.close()

    comm.Barrier()
    gc.collect()

    if SHARED_MEMORY:
        for memory in mpi_memory_1:
            memory.Free()

    if maxoverlap:

        sys.stderr.flush()
        if comm.rank == 0:
            print_and_log(["Overlaps gathered, now computing overlaps/lags"], 'debug', logger)

        assert not half, "Error"
        N_half = N_tm // 2

        if not SHARED_MEMORY:
            over_x, over_y, over_data, sub_over, over_sorted, over_shape = load_data(params, 'overlaps-raw', extension=extension)
        else:
            over_x, over_y, over_data, sub_over, over_sorted, over_shape, mpi_memory_2 = load_data_memshared(
                params, 'overlaps-raw', extension=extension, use_gpu=use_gpu, nb_cpu=nb_cpu, nb_gpu=nb_gpu
            )

        to_explore = numpy.arange(N_half)[comm.rank::comm.size]

        maxlags = numpy.zeros((len(to_explore), N_half), dtype=numpy.int32)
        maxoverlaps = numpy.zeros((len(to_explore), N_half), dtype=numpy.float32)

        res = []
        res2 = []
        for i in to_explore:
            res += [i * N_tm, i * N_tm + N_half]
            res2 += [i, i+1]

        bounds = numpy.searchsorted(over_x, res, 'left')
        bounds_2 = numpy.searchsorted(sub_over[over_sorted], res2, 'left')

        duration = over_shape[1] // 2
        mask_duration = over_y < duration

        for count, i in enumerate(to_explore):

            xmin, xmax = bounds[2*count:2*(count+1)]

            local_x = over_x[xmin:xmax] - i * N_tm
            local_y = over_y[xmin:xmax]
            local_data = over_data[xmin:xmax]

            xmin, xmax = bounds_2[2*count:2*(count+1)]
            nslice = over_sorted[xmin:xmax][mask_duration[over_sorted[xmin:xmax]]]

            local_x = numpy.concatenate((local_x, over_x[nslice] // N_tm))
            local_y = numpy.concatenate((local_y, (over_shape[1] - 1) - over_y[nslice]))
            local_data = numpy.concatenate((local_data, over_data[nslice]))

            data = scipy.sparse.csr_matrix((local_data, (local_x, local_y)), shape=(N_tm, over_shape[1]), dtype=numpy.float32)
            maxoverlaps[count, :] = data.max(1).toarray().flatten()[:N_half]
            maxlags[count, :] = N_t - numpy.array(data.argmax(1)).flatten()[:N_half]
            del local_x, local_y, local_data, data, nslice
            gc.collect()

        # Now we need to sync everything across nodes.
        maxlags = gather_array(maxlags, comm, 0, 1, 'int32', compress=blosc_compress)
        line = numpy.arange(N_half)

        if comm.rank == 0:
            indices = []
            for idx in range(comm.size):
                indices += list(numpy.arange(idx, N_half, comm.size))
            indices = numpy.argsort(indices).astype(numpy.int32)

            maxlags = maxlags[indices, :]
            maxlags[line, line] = 0

            #maxlags = numpy.maximum(maxlags, maxlags.T)
            #mask = numpy.tril(numpy.ones((N_half, N_half)), -1) > 0
            #maxlags[mask] *= -1
        else:
            del maxlags

        gc.collect()

        maxoverlaps = gather_array(maxoverlaps, comm, 0, 1, 'float32', compress=blosc_compress)
        if comm.rank == 0:
            indices = []
            for idx in range(comm.size):
                indices += list(numpy.arange(idx, N_half, comm.size))
            indices = numpy.argsort(indices).astype(numpy.int32)

            maxoverlaps = maxoverlaps[indices, :]
            maxoverlaps[line, line] = 0
            #maxoverlaps = numpy.maximum(maxoverlaps, maxoverlaps.T)
        else:
            del maxoverlaps

        gc.collect()

        if comm.rank == 0:
            myfile2 = h5py.File(file_out_suff + '.templates%s.hdf5' % extension, 'r+', libver='earliest')

            for key in ['maxoverlap', 'maxlag', 'version']:
                if key in myfile2.keys():
                    myfile2.pop(key)

            if not normalize:
                maxoverlaps /= norm_templates[: N_half]
                maxoverlaps /= norm_templates[: N_half][:, numpy.newaxis]

            myfile2.create_dataset('version', data=numpy.array(circus.__version__.split('.'), dtype=numpy.int32))
            if hdf5_compress:
                myfile2.create_dataset('maxlag',  data=maxlags, compression='gzip')
                myfile2.create_dataset('maxoverlap', data=maxoverlaps, compression='gzip')
            else:
                myfile2.create_dataset('maxlag',  data=maxlags)
                myfile2.create_dataset('maxoverlap', data=maxoverlaps)
            myfile2.close()
            del maxoverlaps, maxlags

    comm.Barrier()
    gc.collect()

    if SHARED_MEMORY and maxoverlap:
        for memory in mpi_memory_2:
            memory.Free()

    return h5py.File(filename, 'r')


def load_sp_memshared(file_name, nb_temp):

    intsize = MPI.INT.Get_size()
    floatsize = MPI.FLOAT.Get_size()
    local_rank = sub_comm.rank

    if os.path.exists(file_name):

        c_overlap = h5py.File(file_name, 'r')
        
        results = {}

        if local_rank == 0:
            over_x = c_overlap.get('all/over_x')[:]
            over_data = c_overlap.get('all/over_data')[:]
            noise_x = c_overlap.get('noise/over_x')[:]
            noise_data = c_overlap.get('noise/over_data')[:]
            nb_data = len(over_x)
            nb_noise_data = len(noise_x)
            
            indices_bytes = nb_data * intsize
            data_bytes = nb_data * floatsize
            indices_noise_bytes = nb_noise_data * intsize
            data_noise_bytes = nb_noise_data * floatsize
        else:
            indices_bytes = 0
            data_bytes = 0
            nb_data = 0
            nb_noise_data = 0
            data_noise_bytes = 0
            indices_noise_bytes = 0


        c_overlap.close()

        nb_data = numpy.int64(sub_comm.bcast(numpy.array([nb_data], dtype=numpy.int32), root=0)[0])
        nb_noise_data = numpy.int64(sub_comm.bcast(numpy.array([nb_noise_data], dtype=numpy.int32), root=0)[0])

        win_data = MPI.Win.Allocate_shared(nb_data * floatsize, floatsize, comm=sub_comm)
        buf_data, _ = win_data.Shared_query(0)
        buf_data = numpy.array(buf_data, dtype='B', copy=False)

        win_indices = MPI.Win.Allocate_shared(nb_data * intsize, intsize, comm=sub_comm)
        buf_indices, _ = win_indices.Shared_query(0)
        buf_indices = numpy.array(buf_indices, dtype='B', copy=False)

        data = numpy.ndarray(buffer=buf_data, dtype=numpy.float32, shape=(nb_data,))
        indices = numpy.ndarray(buffer=buf_indices, dtype=numpy.int32, shape=(nb_data,))

        win_data_noise = MPI.Win.Allocate_shared(nb_noise_data * floatsize, floatsize, comm=sub_comm)
        buf_data_noise, _ = win_data_noise.Shared_query(0)
        buf_data_noise = numpy.array(buf_data_noise, dtype='B', copy=False)

        win_indices_noise = MPI.Win.Allocate_shared(nb_noise_data * intsize, intsize, comm=sub_comm)
        buf_indices_noise, _ = win_indices_noise.Shared_query(0)
        buf_indices_noise = numpy.array(buf_indices_noise, dtype='B', copy=False)

        data_noise = numpy.ndarray(buffer=buf_data_noise, dtype=numpy.float32, shape=(nb_noise_data,))
        indices_noise = numpy.ndarray(buffer=buf_indices_noise, dtype=numpy.int32, shape=(nb_noise_data,))

        sub_comm.Barrier()

        if local_rank == 0:
            data[:] = over_data
            indices[:] = over_x
            data_noise[:] = noise_data
            indices_noise[:] = noise_x

        sub_comm.Barrier()

        bounds = numpy.searchsorted(indices, numpy.arange(nb_temp**2 + 1), 'left')

        for c in range(nb_temp**2):

            x_min, x_max = bounds[c], bounds[c+1]
            i = c // nb_temp
            j = numpy.mod(c, nb_temp)
            
            results[i, j] = data[x_min:x_max]

        bounds = numpy.searchsorted(indices_noise, numpy.arange(nb_temp + 1), 'left')

        for c in range(nb_temp):

            x_min, x_max = bounds[c], bounds[c+1]
            
            results[c, 'noise'] = data_noise[x_min:x_max]


        sub_comm.Barrier()
        return results, (win_data, win_indices, win_data_noise, win_indices_noise)

def load_sp(file_name, nb_temp):

    if os.path.exists(file_name):

        c_overlap = h5py.File(file_name, 'r')
        results = {}

        over_x = c_overlap.get('all/over_x')[:]
        over_data = c_overlap.get('all/over_data')[:]
        noise_x = c_overlap.get('noise/over_x')[:]
        noise_data = c_overlap.get('noise/over_data')[:]

        bounds = numpy.searchsorted(over_x, numpy.arange(nb_temp**2 + 1), 'left')

        for c in range(nb_temp**2):

            x_min, x_max = bounds[c], bounds[c+1]
            i = c // nb_temp
            j = numpy.mod(c, nb_temp)
            results[i, j] = over_data[x_min:x_max]

        bounds = numpy.searchsorted(noise_x, numpy.arange(nb_temp + 1), 'left')

        for c in range(nb_temp):

            x_min, x_max = bounds[c], bounds[c+1]
            results[c, 'noise'] = noise_data[x_min:x_max]

        return results