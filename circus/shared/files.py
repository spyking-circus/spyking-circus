from __future__ import division
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

from circus.shared.utils import get_tqdm_progressbar
import numpy, h5py, os, platform, re, sys, scipy, logging
import sys
from colorama import Fore
from mpi import all_gather_array, gather_array, SHARED_MEMORY, comm
from mpi4py import MPI
from circus.shared.probes import get_nodes_and_edges
from circus.shared.messages import print_and_log
from circus.shared.utils import purge

logger = logging.getLogger(__name__)


def data_stats(params, show=True, export_times=False):


    data_file   = params.get_data_file(source=True, has_been_created=False)
    stream_mode = data_file.is_stream
    chunk_size  = 60 * data_file.sampling_rate
    nb_chunks   = data_file.duration // chunk_size
    last_chunk_len = data_file.duration - nb_chunks * chunk_size

    nb_seconds      = last_chunk_len//params.rate
    last_chunk_len -= (nb_seconds*params.rate)
    if nb_seconds > 60:
      nb_extra_seconds = nb_seconds // 60
      nb_chunks  += nb_extra_seconds
      nb_seconds -= 60*nb_extra_seconds
    last_chunk_len  = int(1000*last_chunk_len/params.rate)

    N_t = params.getint('detection', 'N_t')
    N_t = numpy.round(1000.*N_t/params.rate, 1)

    lines = ["Number of recorded channels : %d" %params.nb_channels,
             "Number of analyzed channels : %d" %params.getint('data', 'N_e'),
             "File format                 : %s" %params.get('data', 'file_format').upper(),
             "Data type                   : %s" %str(data_file.data_dtype),
             "Sampling rate               : %d kHz" %(params.rate//1000.),
             "Duration of the recording   : %d min %s s %s ms" %(nb_chunks, int(nb_seconds), last_chunk_len),
             "Width of the templates      : %d ms" %N_t,
             "Spatial radius considered   : %d um" %params.getint('detection', 'radius'),
             "Threshold crossing          : %s" %params.get('detection', 'peaks'),
             "Waveform alignment          : %s" %params.getboolean('detection', 'alignment'),
             "Matched filters             : %s" %params.getboolean('detection', 'matched-filter'),
             "Template Extraction         : %s" %params.get('clustering', 'extraction'),
             "Overwrite                   : %s" %params.get('data', 'overwrite'),
             "Collect all spikes          : %s" %params.getboolean('fitting', 'collect_all'),
             "Smart Search                : %s" %params.getboolean('clustering', 'smart_search'),
             "Smart Selection             : %s" %params.getboolean('clustering', 'smart_select')]

    if stream_mode:
        lines += ["Streams                     : %s (%d found)" %(params.get('data', 'stream_mode'), data_file.nb_streams)]

    if show:
        print_and_log(lines, 'info', logger)

    if not export_times:
        return nb_chunks*60 + nb_seconds + last_chunk_len/1000.
    else:
        return times



def get_stas(params, times_i, labels_i, src, neighs, nodes=None, mean_mode=False, all_labels=False, pos='neg', auto_align=True):

    data_file    = params.data_file
    data_file.open()
    N_t          = params.getint('detection', 'N_t')
    if not all_labels:
        if not mean_mode:
            stas = numpy.zeros((len(times_i), len(neighs), N_t), dtype=numpy.float32)
        else:
            stas = numpy.zeros((len(neighs), N_t), dtype=numpy.float32)
    else:
        nb_labels = numpy.unique(labels_i)
        stas      = numpy.zeros((len(nb_labels), len(neighs), N_t), dtype=numpy.float32)

    alignment     = params.getboolean('detection', 'alignment') and auto_align

    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    template_shift        = params.getint('detection', 'template_shift')

    if do_spatial_whitening:
        spatial_whitening  = load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = load_data(params, 'temporal_whitening')

    if alignment:
        cdata = numpy.linspace(-template_shift, template_shift, 5*N_t)
        xdata = numpy.arange(-2*template_shift, 2*template_shift+1)

    count = 0
    for lb, time in zip(labels_i, times_i):
        if alignment:
            local_chunk = data_file.get_snippet(time - 2*template_shift, 2*N_t - 1, nodes=nodes)
        else:
            local_chunk = data_file.get_snippet(time - template_shift, N_t, nodes=nodes)

        if do_spatial_whitening:
            local_chunk = numpy.dot(local_chunk, spatial_whitening)
        if do_temporal_whitening:
            local_chunk = scipy.ndimage.filters.convolve1d(local_chunk, temporal_whitening, axis=0, mode='constant')

        local_chunk = numpy.take(local_chunk, neighs, axis=1)

        if alignment:
            idx   = numpy.where(neighs == src)[0]
            ydata = numpy.arange(len(neighs))
            if len(ydata) == 1:
                f           = scipy.interpolate.UnivariateSpline(xdata, local_chunk, s=0)
                if pos == 'neg':
                    rmin    = (numpy.argmin(f(cdata)) - len(cdata)/2.)/5.
                elif pos =='pos':
                    rmin    = (numpy.argmax(f(cdata)) - len(cdata)/2.)/5.
                ddata       = numpy.linspace(rmin-template_shift, rmin+template_shift, N_t)
                local_chunk = f(ddata).astype(numpy.float32).reshape(N_t, 1)
            else:
                f           = scipy.interpolate.RectBivariateSpline(xdata, ydata, local_chunk, s=0, ky=min(len(ydata)-1, 3))
                if pos == 'neg':
                    rmin    = (numpy.argmin(f(cdata, idx)[:, 0]) - len(cdata)/2.)/5.
                elif pos == 'pos':
                    rmin    = (numpy.argmax(f(cdata, idx)[:, 0]) - len(cdata)/2.)/5.
                ddata       = numpy.linspace(rmin-template_shift, rmin+template_shift, N_t)
                local_chunk = f(ddata, ydata).astype(numpy.float32)

        if all_labels:
            lc        = numpy.where(nb_labels == lb)[0]
            stas[lc] += local_chunk.T
        else:
            if not mean_mode:
                stas[count, :, :] = local_chunk.T
                count            += 1
            else:
                stas += local_chunk.T

    data_file.close()

    return stas


def get_stas_memshared(params, times_i, labels_i, src, neighs, nodes=None,
                       mean_mode=False, all_labels=False, auto_align=True):

    # First we need to identify machines in the MPI ring.
    from uuid import getnode as get_mac
    myip = numpy.int64(get_mac()) % 100000
    ##### TODO: remove quarantine zone
    # intsize = MPI.INT.Get_size()
    ##### end quarantine zone
    float_size = MPI.FLOAT.Get_size()
    sub_comm = comm.Split(myip, 0)

    # Load parameters.
    data_file    = params.data_file
    data_file.open()
    N_t          = params.getint('detection', 'N_t')
    N_total      = params.nb_channels
    alignment    = params.getboolean('detection', 'alignment') and auto_align
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening = params.getboolean('whitening', 'spatial')
    template_shift = params.getint('detection', 'template_shift')

    # Calculate the sizes of the data structures to share.
    nb_triggers = 0
    nb_neighs = 0
    nb_ts = 0
    if sub_comm.Get_rank() == 0:
        if not all_labels:
            if not mean_mode:
                ##### TODO: clean quarantine zone
                # nb_times = len(times_i)
                ##### end quarantine zone
                nb_triggers = len(times_i)
            else:
                nb_triggers = 1
        else:
            ##### TODO: remove quarantine zone
            # nb_labels = len(numpy.unique(labels_i))
            ##### end quarantine zone
            nb_triggers = len(numpy.unique(labels_i))
        nb_neighs = len(neighs)
        nb_ts = N_t

    sub_comm.Barrier()

    # Broadcast the sizes of the data structures to share.
    triggers_size = numpy.int64(sub_comm.bcast(numpy.array([nb_triggers], dtype=numpy.int32), root=0)[0])
    neighs_size = numpy.int64(sub_comm.bcast(numpy.array([nb_neighs], dtype=numpy.int32), root=0)[0])
    ts_size = numpy.int64(sub_comm.bcast(numpy.array([nb_ts], dtype=numpy.int32), root=0)[0])

    # Declare the data structures to share.
    if sub_comm.Get_rank() == 0:
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
    if sub_comm.Get_rank() == 0:
        if do_spatial_whitening:
            spatial_whitening = load_data(params, 'spatial_whitening')
        if do_temporal_whitening:
            temporal_whitening = load_data(params, 'temporal_whitening')
        if alignment:
            cdata = numpy.linspace(- template_shift, template_shift, 5 * N_t)
            xdata = numpy.arange(- 2 * template_shift, 2 * template_shift + 1)
        count = 0
        for lb, time in zip(labels_i, times_i):

            if alignment:
                local_chunk = data_file.get_snippet(time - 2*template_shift, 2*N_t - 1, nodes=nodes)
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
                    f = scipy.interpolate.UnivariateSpline(xdata, local_chunk, s=0)
                    rmin = (numpy.argmin(f(cdata)) - len(cdata) / 2.0) / 5.0
                    ddata = numpy.linspace(rmin - template_shift, rmin + template_shift, N_t)
                    local_chunk = f(ddata).astype(numpy.float32).reshape(N_t, 1)
                else:
                    f = scipy.interpolate.RectBivariateSpline(xdata, ydata, local_chunk, s=0, ky=min(len(ydata) - 1, 3))
                    rmin = (numpy.argmin(f(cdata, idx)[:, 0]) - len(cdata) / 2.0) / 5.0
                    ddata = numpy.linspace(rmin - template_shift, rmin + template_shift, N_t)
                    local_chunk = f(ddata, ydata).astype(numpy.float32)
            if not all_labels:
                if not mean_mode:
                    # #####
                    # print(stas.shape)
                    # print(count)
                    # #####
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

    sub_comm.Free()
    data_file.close()

    return stas

##### end working zone


def get_artefact(params, times_i, tau, nodes, normalize=True):

    data_file    = params.data_file
    data_file.open()

    artefact     = numpy.zeros((len(nodes), tau), dtype=numpy.float32)
    for time in times_i:
        artefact += data_file.get_snippet(time, tau, nodes).T

    if normalize:
        artefact /= len(times_i)

    data_file.close()

    return artefact



def load_data_memshared(params, data, extension='', normalize=False, transpose=False, nb_cpu=1, nb_gpu=0, use_gpu=False):

    file_out        = params.get('data', 'file_out')
    file_out_suff   = params.get('data', 'file_out_suff')
    data_file_noext = params.get('data', 'data_file_noext')

    ## First we need to identify machines in the MPI ring
    from uuid import getnode as get_mac
    myip = numpy.int64(get_mac()) % 100000

    intsize   = MPI.INT.Get_size()
    floatsize = MPI.FLOAT.Get_size()
    sub_comm  = comm.Split(myip, 0)

    data_file = params.data_file
    N_e       = params.getint('data', 'N_e')
    N_t       = params.getint('detection', 'N_t')

    if data == 'templates':
        if os.path.exists(file_out_suff + '.templates%s.hdf5' %extension):
            nb_data = 0
            nb_ptr  = 0
            nb_templates = h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r', libver='latest').get('norms').shape[0]

            if sub_comm.rank == 0:
                temp_x       = h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r', libver='latest').get('temp_x')[:]
                temp_y       = h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r', libver='latest').get('temp_y')[:]
                temp_data    = h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r', libver='latest').get('temp_data')[:]
                sparse_mat = scipy.sparse.csc_matrix((temp_data, (temp_x, temp_y)), shape=(N_e*N_t, nb_templates))
                if normalize:
                    norm_templates = load_data(params, 'norm-templates')
                    for idx in xrange(sparse_mat.shape[1]):
                        myslice = numpy.arange(sparse_mat.indptr[idx], sparse_mat.indptr[idx+1])
                        sparse_mat.data[myslice] /= norm_templates[idx]
                if transpose:
                    sparse_mat = sparse_mat.T

                nb_data = len(sparse_mat.data)
                nb_ptr  = len(sparse_mat.indptr)

            sub_comm.Barrier()
            long_size  = numpy.int64(sub_comm.bcast(numpy.array([nb_data], dtype=numpy.int32), root=0)[0])
            short_size = numpy.int64(sub_comm.bcast(numpy.array([nb_ptr], dtype=numpy.int32), root=0)[0])

            if sub_comm.rank == 0:
                indptr_bytes  = short_size * intsize
                indices_bytes = long_size * intsize
                data_bytes    = long_size * floatsize
            else:
                indptr_bytes  = 0
                indices_bytes = 0
                data_bytes    = 0

            win_data    = MPI.Win.Allocate_shared(data_bytes, floatsize, comm=sub_comm)
            win_indices = MPI.Win.Allocate_shared(indices_bytes, intsize, comm=sub_comm)
            win_indptr  = MPI.Win.Allocate_shared(indptr_bytes, intsize, comm=sub_comm)

            buf_data, _    = win_data.Shared_query(0)
            buf_indices, _ = win_indices.Shared_query(0)
            buf_indptr, _  = win_indptr.Shared_query(0)

            buf_data    = numpy.array(buf_data, dtype='B', copy=False)
            buf_indices = numpy.array(buf_indices, dtype='B', copy=False)
            buf_indptr  = numpy.array(buf_indptr, dtype='B', copy=False)

            data    = numpy.ndarray(buffer=buf_data, dtype=numpy.float32, shape=(long_size,))
            indices = numpy.ndarray(buffer=buf_indices, dtype=numpy.int32, shape=(long_size,))
            indptr  = numpy.ndarray(buffer=buf_indptr, dtype=numpy.int32, shape=(short_size,))

            sub_comm.Barrier()

            if sub_comm.rank == 0:
                data[:nb_data]    = sparse_mat.data
                indices[:nb_data] = sparse_mat.indices
                indptr[:nb_data]  = sparse_mat.indptr
                del sparse_mat

            sub_comm.Barrier()
            if not transpose:
                templates = scipy.sparse.csc_matrix((N_e*N_t, nb_templates), dtype=numpy.float32)
            else:
                templates = scipy.sparse.csr_matrix((nb_templates, N_e*N_t), dtype=numpy.float32)
            templates.data    = data
            templates.indices = indices
            templates.indptr  = indptr

            sub_comm.Free()

            return templates
        else:
            raise Exception('No templates found! Check suffix?')
    elif data == "overlaps":

        c_overlap  = get_overlaps(params, nb_cpu=nb_cpu, nb_gpu=nb_gpu, use_gpu=use_gpu)
        over_shape = c_overlap.get('over_shape')[:]
        N_over     = numpy.int64(numpy.sqrt(over_shape[0]))
        S_over     = over_shape[1]
        c_overs    = {}

        if sub_comm.rank == 0:
            over_x     = c_overlap.get('over_x')[:]
            over_y     = c_overlap.get('over_y')[:]
            over_data  = c_overlap.get('over_data')[:]
            c_overlap.close()

            # To be faster, we rearrange the overlaps into a dictionnary
            overlaps  = scipy.sparse.csr_matrix((over_data, (over_x, over_y)), shape=(over_shape[0], over_shape[1]))
            del over_x, over_y, over_data

        sub_comm.Barrier()

        nb_data = 0
        nb_ptr  = 0

        for i in xrange(N_over):

            if sub_comm.rank == 0:
                sparse_mat = overlaps[i*N_over:(i+1)*N_over]
                nb_data    = len(sparse_mat.data)
                nb_ptr     = len(sparse_mat.indptr)

            long_size  = numpy.int64(sub_comm.bcast(numpy.array([nb_data], dtype=numpy.int32), root=0)[0])
            short_size = numpy.int64(sub_comm.bcast(numpy.array([nb_ptr], dtype=numpy.int32), root=0)[0])

            if sub_comm.rank == 0:
                indptr_bytes  = short_size * intsize
                indices_bytes = long_size * intsize
                data_bytes    = long_size * floatsize
            else:
                indptr_bytes  = 0
                indices_bytes = 0
                data_bytes    = 0

            win_data    = MPI.Win.Allocate_shared(data_bytes, floatsize, comm=sub_comm)
            win_indices = MPI.Win.Allocate_shared(indices_bytes, intsize, comm=sub_comm)
            win_indptr  = MPI.Win.Allocate_shared(indptr_bytes, intsize, comm=sub_comm)

            buf_data, _    = win_data.Shared_query(0)
            buf_indices, _ = win_indices.Shared_query(0)
            buf_indptr, _  = win_indptr.Shared_query(0)

            buf_data    = numpy.array(buf_data, dtype='B', copy=False)
            buf_indices = numpy.array(buf_indices, dtype='B', copy=False)
            buf_indptr  = numpy.array(buf_indptr, dtype='B', copy=False)

            data    = numpy.ndarray(buffer=buf_data, dtype=numpy.float32, shape=(long_size,))
            indices = numpy.ndarray(buffer=buf_indices, dtype=numpy.int32, shape=(long_size,))
            indptr  = numpy.ndarray(buffer=buf_indptr, dtype=numpy.int32, shape=(short_size,))

            sub_comm.Barrier()

            if sub_comm.rank == 0:
                data[:]    = sparse_mat.data
                indices[:] = sparse_mat.indices
                indptr[:]  = sparse_mat.indptr
                del sparse_mat

            c_overs[i]         = scipy.sparse.csr_matrix((N_over, over_shape[1]), dtype=numpy.float32)
            c_overs[i].data    = data
            c_overs[i].indices = indices
            c_overs[i].indptr  = indptr

            sub_comm.Barrier()

        if sub_comm.rank == 0:
            del overlaps

        sub_comm.Free()

        return c_overs

    elif data == 'clusters-light':

        if os.path.exists(file_out_suff + '.clusters%s.hdf5' %extension):
            myfile = h5py.File(file_out_suff + '.clusters%s.hdf5' %extension, 'r', libver='latest')
            result = {}

            nb_data = 0

            for key in myfile.keys():

                if ('clusters_' in key) or (key == 'electrodes'):
                    if sub_comm.rank == 0:
                        locdata = myfile.get(key)[:]
                        nb_data = len(locdata)

                    data_size  = numpy.int64(sub_comm.bcast(numpy.array([nb_data], dtype=numpy.int32), root=0)[0])

                    if sub_comm.rank == 0:
                        if locdata.dtype == 'int32':
                            type_size = 0
                        elif locdata.dtype == 'float32':
                            type_size = 1
                        data_bytes = data_size * 4
                    else:
                        type_size  = 0
                        data_bytes = 0

                    type_size  = numpy.int64(sub_comm.bcast(numpy.array([type_size], dtype=numpy.int32), root=0)[0])

                    win_data    = MPI.Win.Allocate_shared(data_bytes, 4, comm=sub_comm)
                    buf_data, _ = win_data.Shared_query(0)

                    buf_data    = numpy.array(buf_data, dtype='B', copy=False)

                    if type_size == 0:
                        data = numpy.ndarray(buffer=buf_data, dtype=numpy.int32, shape=(data_size,))
                    elif type_size == 1:
                        data = numpy.ndarray(buffer=buf_data, dtype=numpy.float32, shape=(data_size,))

                    if sub_comm.rank == 0:
                        data[:]    = locdata

                    sub_comm.Barrier()

                    result[str(key)] = data

            sub_comm.Free()

            myfile.close()
            return result



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

    file_out_suff   = params.get('data', 'file_out_suff')
    data_file_noext = params.get('data', 'data_file_noext')

    if data == 'thresholds':
        spike_thresh = params.getfloat('detection', 'spike_thresh')
        if os.path.exists(file_out_suff + '.basis.hdf5'):
            myfile     = h5py.File(file_out_suff + '.basis.hdf5', 'r', libver='latest')
            thresholds = myfile.get('thresholds')[:]
            myfile.close()
            return spike_thresh * thresholds
    elif data == 'matched-thresholds':
        matched_thresh = params.getfloat('detection', 'matched_thresh')
        if os.path.exists(file_out_suff + '.basis.hdf5'):
            myfile     = h5py.File(file_out_suff + '.basis.hdf5', 'r', libver='latest')
            thresholds = myfile.get('matched_thresholds')[:]
            myfile.close()
            return matched_thresh * thresholds
    elif data == 'matched-thresholds-pos':
        matched_thresh = params.getfloat('detection', 'matched_thresh')
        if os.path.exists(file_out_suff + '.basis.hdf5'):
            myfile     = h5py.File(file_out_suff + '.basis.hdf5', 'r', libver='latest')
            thresholds = myfile.get('matched_thresholds_pos')[:]
            myfile.close()
            return matched_thresh * thresholds
    elif data == 'spatial_whitening':
        if os.path.exists(file_out_suff + '.basis.hdf5'):
            myfile  = h5py.File(file_out_suff + '.basis.hdf5', 'r', libver='latest')
            spatial = numpy.ascontiguousarray(myfile.get('spatial')[:])
            myfile.close()
            return spatial
        else:
            raise Exception('Whitening matrix has to be computed first!')
    elif data == 'temporal_whitening':
        if os.path.exists(file_out_suff + '.basis.hdf5'):
            myfile   = h5py.File(file_out_suff + '.basis.hdf5', 'r', libver='latest')
            temporal = myfile.get('temporal')[:]
            myfile.close()
            return temporal
        else:
            raise Exception('Whitening matrix has to be computed first!')
    elif data == 'basis':
        myfile     = h5py.File(file_out_suff + '.basis.hdf5', 'r', libver='latest')
        basis_proj = numpy.ascontiguousarray(myfile.get('proj')[:])
        basis_rec  = numpy.ascontiguousarray(myfile.get('rec')[:])
        myfile.close()
        return basis_proj, basis_rec
    elif data == 'basis-pos':
        myfile     = h5py.File(file_out_suff + '.basis.hdf5', 'r', libver='latest')
        basis_proj = numpy.ascontiguousarray(myfile.get('proj_pos')[:])
        basis_rec  = numpy.ascontiguousarray(myfile.get('rec_pos')[:])
        myfile.close()
        return basis_proj, basis_rec
    elif data == 'waveform':
        myfile     = h5py.File(file_out_suff + '.basis.hdf5', 'r', libver='latest')
        waveforms  = myfile.get('waveform')[:]
        myfile.close()
        return waveforms
    elif data == 'waveforms':
        myfile     = h5py.File(file_out_suff + '.basis.hdf5', 'r', libver='latest')
        waveforms  = myfile.get('waveforms')[:]
        myfile.close()
        return waveforms
    elif data == 'waveform-pos':
        myfile     = h5py.File(file_out_suff + '.basis.hdf5', 'r', libver='latest')
        waveforms  = myfile.get('waveform_pos')[:]
        myfile.close()
        return waveforms
    elif data == 'waveforms-pos':
        myfile     = h5py.File(file_out_suff + '.basis.hdf5', 'r', libver='latest')
        waveforms  = myfile.get('waveforms_pos')[:]
        myfile.close()
        return waveforms
    elif data == 'templates':
        if os.path.exists(file_out_suff + '.templates%s.hdf5' %extension):
            temp_x = h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r', libver='latest').get('temp_x')[:]
            temp_y = h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r', libver='latest').get('temp_y')[:]
            temp_data = h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r', libver='latest').get('temp_data')[:]
            N_e, N_t, nb_templates = h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r', libver='latest').get('temp_shape')[:]
            return scipy.sparse.csc_matrix((temp_data, (temp_x, temp_y)), shape=(N_e*N_t, nb_templates))
        else:
            raise Exception('No templates found! Check suffix?')
    elif data == 'norm-templates':
        if os.path.exists(file_out_suff + '.templates%s.hdf5' %extension):
            return h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r', libver='latest').get('norms')[:]
        else:
            raise Exception('No templates found! Check suffix?')
    elif data == 'spike-cluster':
        file_name = params.get('data', 'data_file_noext') + '.spike-cluster.hdf5'
        if os.path.exists(file_name):
            data       = h5py.File(file_name, 'r')
            clusters   = data.get('clusters').ravel()
            N_clusters = len(numpy.unique(clusters))
            spiketimes = data.get('spikes').ravel()
            return clusters, spiketimes, N_clusters
        else:
            raise Exception('Need to provide a spike-cluster file!')
    elif data == 'clusters':
        if os.path.exists(file_out_suff + '.clusters%s.hdf5' %extension):
            myfile = h5py.File(file_out_suff + '.clusters%s.hdf5' %extension, 'r', libver='latest')
            result = {}
            for key in myfile.keys():
                result[str(key)] = myfile.get(key)[:]
            myfile.close()
            return result
        else:
            raise Exception('No clusters found! Check suffix or run clustering?')
    elif data == 'clusters-light':
        if os.path.exists(file_out_suff + '.clusters%s.hdf5' %extension):
            myfile = h5py.File(file_out_suff + '.clusters%s.hdf5' %extension, 'r', libver='latest')
            result = {}
            for key in myfile.keys():
                if ('clusters_' in key) or (key == 'electrodes'):
                    result[str(key)] = myfile.get(key)[:]
            myfile.close()
            return result
        else:
            raise Exception('No clusters found! Check suffix or run clustering?')
    elif data == 'electrodes':
        if os.path.exists(file_out_suff + '.clusters%s.hdf5' %extension):
            myfile     = h5py.File(file_out_suff + '.clusters%s.hdf5' %extension, 'r', libver='latest')
            electrodes = myfile.get('electrodes')[:]
            myfile.close()
            return electrodes
        else:
            raise Exception('No clusters found! Check suffix or run clustering?')
    elif data == 'results':
        try:
            return get_results(params, extension)
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
        if os.path.exists(file_out_suff + '.templates%s.hdf5' %extension):
            myfile = h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r', libver='latest')
            limits = myfile.get('limits')[:]
            myfile.close()
            return limits
        else:
            raise Exception('No templates found! Check suffix?')
    elif data == 'injected_spikes':
        try:
            spikes = h5py.File(data_file_noext + '/injected/result.hdf5').get('spiketimes')
            elecs  = numpy.load(data_file_noext + '/injected/elecs.npy')
            N_tm   = len(spikes)
            count  = 0
            result = {}
            for i in xrange(N_tm):
                key = 'temp_' + str(i)
                if len(spikes[key]) > 0:
                    result['spikes_' + str(elecs[count])] = spikes[key]
                    count += 1
            return result
        except Exception:
            return None
    elif data == 'triggers':
        filename = file_out_suff + '.triggers%s.npy' %extension
        if os.path.exists(filename):
            triggers = numpy.load(filename)
            N_tr     = triggers.shape[0]

            data_file = params.data_file
            data_file.open()

            N_total = params.nb_channels
            N_t     = params.getint('detection', 'N_t')

            template_shift = params.getint('detection', 'template_shift')

            spikes = numpy.zeros((N_t, N_total, N_tr))
            for (count, idx) in enumerate(triggers):
                spikes[:, :, count] = data_file.get_snippet(idx - template_shift, N_t)
            data_file.close()
            return triggers, spikes
        else:
            raise Exception('No triggers found! Check suffix or check if file `%s` exists?' %filename)
    elif data == 'juxta-mad':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='latest')
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
            beer_file = h5py.File(filename, 'r', libver='latest')
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
            beer_file = h5py.File(filename, 'r', libver='latest')
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
            beer_file = h5py.File(filename, 'r', libver='latest')
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
            beer_file = h5py.File(filename, 'r', libver='latest')
            N_e = params.getint('data', 'N_e')
            extra_spike_times = N_e * [None]
            try:
                for e in xrange(0, N_e):
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
            beer_file = h5py.File(filename, 'r', libver='latest')
            N_e = params.getint('data', 'N_e')
            extra_spike_values = N_e * [None]
            try:
                for e in xrange(0, N_e):
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
            bfile = h5py.File(filename, 'r', libver='latest')
            class_weights = bfile.get('class-weights')[:]
            bfile.close()
            return class_weights
        else:
            raise Exception('No class weights found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data == 'confusion-matrices':
        filename = file_out_suff + '.beer.hdf5'
        if os.path.exists(filename):
            bfile = h5py.File(filename, 'r', libver='latest')
            confusion_matrices = bfile.get('confusion_matrices')[:]
            bfile.close()
            return confusion_matrices
        else:
            raise Exception('No confusion matrices found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data == 'proportion':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='latest')
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
            beer_file = h5py.File(filename, 'r', libver='latest')
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
        confusion_matrices = load_data(params, 'confusion-matrices')
        threshold_false_negatives = load_data(params, 'threshold-false-negatives')
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
            beer_file = h5py.File(filename, 'r', libver='latest')
            try:
                sc_contingency_matrices = beer_file.get('sc_contingency_matrices')[:]
            finally:
                beer_file.close()
            return sc_contingency_matrices
        else:
            raise Exception('No contingency matrices found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data in ['sc-false-positive-error-rates', 'sc-false-negative-error-rates']:
        # Retrieve saved data.
        sc_contingency_matrices = load_data(params, 'sc-contingency-matrices')
        threshold_false_negatives = load_data(params, 'threshold-false-negatives')
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
            beer_file = h5py.File(filename, 'r', libver='latest')
            try:
                sc_contingency_matrix = beer_file.get('sc_contingency_matrix')[:]
            finally:
                beer_file.close()
            return sc_contingency_matrix
        else:
            raise Exception('No contingency matrix found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data in ['sc-best-false-positive-error-rate', 'sc-best-false-negative-error-rate']:
        sc_contingency_matrix = load_data(params, 'sc-contingency-matrix')
        threshold_false_negatives = load_data(params, 'threshold-false-negatives')
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
            beer_file = h5py.File(filename, 'r', libver='latest')
            try:
                selection = beer_file.get('selection')[:]
            finally:
                beer_file.close()
            return selection
        else:
            raise Exception('No selection found! Check suffix or check if file `{}` exists?'.format(filename))


def write_datasets(h5file, to_write, result, electrode=None):
    for key in to_write:
        if electrode is not None:
            mykey = key + str(electrode)
        else:
            mykey = key
        h5file.create_dataset(mykey, shape=result[mykey].shape, dtype=result[mykey].dtype, chunks=True)
        h5file.get(mykey)[:] = result[mykey]

def collect_data(nb_threads, params, erase=False, with_real_amps=False, with_voltages=False, benchmark=False):

    # Retrieve the key parameters.
    data_file      = params.data_file
    N_e            = params.getint('data', 'N_e')
    N_t            = params.getint('detection', 'N_t')
    file_out_suff  = params.get('data', 'file_out_suff')
    max_chunk      = params.getfloat('fitting', 'max_chunk')
    chunks         = params.getfloat('fitting', 'chunk_size')
    data_length    = data_stats(params, show=False)
    duration       = int(min(chunks*max_chunk, data_length))
    templates      = load_data(params, 'norm-templates')
    refractory     = params.getfloat('fitting', 'refractory')
    N_tm           = len(templates)
    collect_all    = params.getboolean('fitting', 'collect_all')
    print_and_log(["Gathering data from %d nodes..." %nb_threads], 'default', logger)

    # Initialize data collection.
    result = {'spiketimes' : {}, 'amplitudes' : {}, 'info' : {'duration' : numpy.array([duration], dtype=numpy.uint64)}}
    if with_real_amps:
        result['real_amps'] = {}
    if with_voltages:
        result['voltages'] = {}
    if collect_all:
        result['gspikes'] = {}
        result['gtemps']  = {}

    for i in xrange(N_tm//2):
        result['spiketimes']['temp_' + str(i)]  = numpy.empty(shape=0, dtype=numpy.uint32)
        result['amplitudes']['temp_' + str(i)]  = numpy.empty(shape=(0, 2), dtype=numpy.float32)
        if with_real_amps:
            result['real_amps']['temp_' + str(i)] = numpy.empty(shape=0, dtype=numpy.float32)
        if with_voltages:
            result['voltages']['temp_' + str(i)] = numpy.empty(shape=0, dtype=numpy.float32)

    if collect_all:
        for i in xrange(N_e):
            result['gspikes']['elec_' + str(i)] = numpy.empty(shape=0, dtype=numpy.uint32)

    to_explore = xrange(nb_threads)

    if comm.rank == 0:
        to_explore = get_tqdm_progressbar(to_explore)

    # For each thread/process collect data.
    for count, node in enumerate(to_explore):
        spiketimes_file = file_out_suff + '.spiketimes-%d.data' %node
        amplitudes_file = file_out_suff + '.amplitudes-%d.data' %node
        templates_file  = file_out_suff + '.templates-%d.data' %node
        if with_real_amps:
            real_amps_file = file_out_suff + '.real_amps-%d.data' %node
            real_amps      = numpy.fromfile(real_amps_file, dtype=numpy.float32)
        if with_voltages:
            voltages_file  = file_out_suff + '.voltages-%d.data' %node
            voltages       = numpy.fromfile(voltages_file, dtype=numpy.float32)

        if collect_all:
            gspikes_file = file_out_suff + '.gspiketimes-%d.data' %node
            gspikes      = numpy.fromfile(gspikes_file, dtype=numpy.uint32)
            gtemps_file  = file_out_suff + '.gtemplates-%d.data' %node
            gtemps       = numpy.fromfile(gtemps_file, dtype=numpy.int32)

        if os.path.exists(amplitudes_file):

            amplitudes = numpy.fromfile(amplitudes_file, dtype=numpy.float32)
            spiketimes = numpy.fromfile(spiketimes_file, dtype=numpy.uint32)
            templates  = numpy.fromfile(templates_file, dtype=numpy.int32)
            N          = len(amplitudes)
            amplitudes = amplitudes.reshape(N//2, 2)
            min_size   = min([amplitudes.shape[0], spiketimes.shape[0], templates.shape[0]])
            amplitudes = amplitudes[:min_size]
            spiketimes = spiketimes[:min_size]
            templates  = templates[:min_size]
            if with_real_amps:
                real_amps = real_amps[:min_size]
            if with_voltages:
                voltages = voltages[:min_size]

            local_temp = numpy.unique(templates)

            for j in local_temp:
                idx = numpy.where(templates == j)[0]
                result['amplitudes']['temp_' + str(j)] = numpy.concatenate((amplitudes[idx], result['amplitudes']['temp_' + str(j)]))
                result['spiketimes']['temp_' + str(j)] = numpy.concatenate((result['spiketimes']['temp_' + str(j)], spiketimes[idx]))
                if with_real_amps:
                    result['real_amps']['temp_' + str(j)] = numpy.concatenate((result['real_amps']['temp_' + str(j)], real_amps[idx]))
                if with_voltages:
                    result['voltages']['temp_' + str(j)] = numpy.concatenate((result['voltages']['temp_' + str(j)], voltages[idx]))

            if collect_all:
                for j in xrange(N_e):
                    idx = numpy.where(gtemps == j)[0]
                    result['gspikes']['elec_' + str(j)] = numpy.concatenate((result['gspikes']['elec_' + str(j)], gspikes[idx]))

    # TODO: find a programmer comment.
    for key in result['spiketimes']:
        result['spiketimes'][key] = numpy.array(result['spiketimes'][key], dtype=numpy.uint32)
        idx                       = numpy.argsort(result['spiketimes'][key])
        result['amplitudes'][key] = numpy.array(result['amplitudes'][key], dtype=numpy.float32)
        result['spiketimes'][key] = result['spiketimes'][key][idx]
        result['amplitudes'][key] = result['amplitudes'][key][idx]
        if with_real_amps:
            result['real_amps'][key] = result['real_amps'][key][idx]
        if with_voltages:
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
            result['gspikes'][key] = numpy.array(result['gspikes'][key], dtype=numpy.uint32)
            idx                    = numpy.argsort(result['gspikes'][key])
            result['gspikes'][key] = result['gspikes'][key][idx]

    keys = ['spiketimes', 'amplitudes', 'info']
    if with_real_amps:
        keys += ['real_amps']
    if with_voltages:
        keys += ['voltages']
    if collect_all:
        keys += ['gspikes']

    # Save results into `<dataset>/<dataset>.result.hdf5`.
    mydata = h5py.File(file_out_suff + '.result.hdf5', 'w', libver='latest')
    for key in keys:
        mydata.create_group(key)
        for temp in result[key].keys():
            tmp_path = '%s/%s' %(key, temp)
            mydata.create_dataset(tmp_path, data=result[key][temp])
    mydata.close()

    # Count and print the number of spikes.
    count = 0
    for item in result['spiketimes'].keys():
        count += len(result['spiketimes'][item])

    if collect_all:
        gcount = 0
        for item in result['gspikes'].keys():
            gcount += len(result['gspikes'][item])

    if benchmark:
        to_print = "injected"
    else:
        to_print = "fitted"

    to_write = ["Number of spikes %s : %d" %(to_print, count)]

    if collect_all:
        to_write += ["Number of spikes not fitted (roughly): %d [%g percent]" %(gcount, 100*gcount/float(count))]

    print_and_log(to_write, 'info', logger)

    # TODO: find a programmer comment
    if erase:
        purge(file_out_suff, '.data')

def get_results(params, extension=''):
    file_out_suff        = params.get('data', 'file_out_suff')
    result               = {}
    myfile               = h5py.File(file_out_suff + '.result%s.hdf5' %extension, 'r', libver='latest')
    for key in ['spiketimes', 'amplitudes']:
        result[str(key)] = {}
        for temp in myfile.get(key).keys():
            result[str(key)][str(temp)] = myfile.get(key).get(temp)[:]
    myfile.close()
    return result

def get_garbage(params, extension=''):
    file_out_suff        = params.get('data', 'file_out_suff')
    result               = {}
    myfile               = h5py.File(file_out_suff + '.result%s.hdf5' %extension, 'r', libver='latest')
    for key in ['gspikes']:
        result[str(key)] = {}
        for temp in myfile.get(key).keys():
            result[str(key)][str(temp)] = myfile.get(key).get(temp)[:]
    myfile.close()
    return result

def get_overlaps(params, extension='', erase=False, normalize=True, maxoverlap=True, verbose=True, half=False, use_gpu=False, nb_cpu=1, nb_gpu=0):

    import h5py
    parallel_hdf5  = h5py.get_config().mpi
    data_file      = params.data_file
    N_e            = params.getint('data', 'N_e')
    N_t            = params.getint('detection', 'N_t')
    N_total        = params.nb_channels
    file_out_suff  = params.get('data', 'file_out_suff')
    tmp_path       = os.path.join(os.path.abspath(params.get('data', 'data_file_noext')), 'tmp')
    filename       = file_out_suff + '.overlap%s.hdf5' %extension

    if os.path.exists(filename) and not erase:
        return h5py.File(filename, 'r')
    else:
        if os.path.exists(filename) and erase and (comm.rank == 0):
            os.remove(filename)

    if maxoverlap:
        if SHARED_MEMORY:
            templates  = load_data_memshared(params, 'templates', extension=extension, normalize=normalize)
        else:
            templates  = load_data(params, 'templates', extension=extension)
    else:
        if SHARED_MEMORY:
            templates  = load_data_memshared(params, 'templates', normalize=normalize)
        else:
            templates  = load_data(params, 'templates')

    if extension == '-merged':
        best_elec  = load_data(params, 'electrodes', extension)
    else:
        best_elec  = load_data(params, 'electrodes')
    nodes, edges   = get_nodes_and_edges(params)
    N,        N_tm = templates.shape

    if not SHARED_MEMORY and normalize:
        norm_templates = load_data(params, 'norm-templates')[:N_tm]
        for idx in xrange(N_tm):
            myslice = numpy.arange(templates.indptr[idx], templates.indptr[idx+1])
            templates.data[myslice] /= norm_templates[idx]

    if half:
        N_tm //= 2

    comm.Barrier()
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)

    cuda_string = 'using %d CPU...' %comm.size

    if use_gpu:
        import cudamat as cmt
        if parallel_hdf5:
            if nb_gpu > nb_cpu:
                gpu_id = int(comm.rank//nb_cpu)
            else:
                gpu_id = 0
        else:
            gpu_id = 0
        cmt.cuda_set_device(gpu_id)
        cmt.init()
        cmt.cuda_sync_threads()

    if use_gpu:
        cuda_string = 'using %d GPU...' %comm.size

    all_delays      = numpy.arange(1, N_t+1)

    local_templates = numpy.zeros(0, dtype=numpy.int32)
    for ielec in range(comm.rank, N_e, comm.size):
        local_templates = numpy.concatenate((local_templates, numpy.where(best_elec == ielec)[0]))

    if half:
        nb_total     = len(local_templates)
        upper_bounds = N_tm
    else:
        nb_total     = 2*len(local_templates)
        upper_bounds = N_tm//2

    to_explore = xrange(comm.rank, N_e, comm.size)

    if comm.rank == 0:
        if verbose:
            print_and_log(["Pre-computing the overlaps of templates %s" %cuda_string], 'default', logger)
        N_0  = len(range(comm.rank, N_e, comm.size))
        to_explore = get_tqdm_progressbar(to_explore)


    over_x    = numpy.zeros(0, dtype=numpy.int32)
    over_y    = numpy.zeros(0, dtype=numpy.int32)
    over_data = numpy.zeros(0, dtype=numpy.float32)
    rows      = numpy.arange(N_e*N_t)

    for count, ielec in enumerate(to_explore):

        local_idx = numpy.where(best_elec == ielec)[0]
        len_local = len(local_idx)

        if not half:
            local_idx = numpy.concatenate((local_idx, local_idx + upper_bounds))

        if len_local > 0:

            to_consider   = numpy.arange(upper_bounds)
            if not half:
                to_consider = numpy.concatenate((to_consider, to_consider + upper_bounds))

            loc_templates  = templates[:, local_idx].tocsr()
            loc_templates2 = templates[:, to_consider].tocsr()

            for idelay in all_delays:

                srows = numpy.where(rows % N_t < idelay)[0]
                tmp_1 = loc_templates[srows]

                srows = numpy.where(rows % N_t >= (N_t - idelay))[0]
                tmp_2 = loc_templates2[srows]

                if use_gpu:
                    tmp_1 = cmt.SparseCUDAMatrix(tmp_1.T.tocsr(), copy_on_host=False)
                    tmp_2 = cmt.CUDAMatrix(tmp_2.toarray(), copy_on_host=False)
                    data  = cmt.sparse_dot(tmp_1, tmp_2).asarray()
                else:
                    data  = tmp_1.T.dot(tmp_2)
                    data  = data.toarray()

                dx, dy     = data.nonzero()
                ddx        = numpy.take(local_idx, dx).astype(numpy.int32)
                ddy        = numpy.take(to_consider, dy).astype(numpy.int32)
                data       = data.ravel()
                dd         = data.nonzero()[0].astype(numpy.int32)
                over_x     = numpy.concatenate((over_x, ddx*N_tm + ddy))
                over_y     = numpy.concatenate((over_y, (idelay-1)*numpy.ones(len(dx), dtype=numpy.int32)))
                over_data  = numpy.concatenate((over_data, numpy.take(data, dd)))
                if idelay < N_t:
                    over_x     = numpy.concatenate((over_x, ddy*N_tm + ddx))
                    over_y     = numpy.concatenate((over_y, (2*N_t-idelay-1)*numpy.ones(len(dx), dtype=numpy.int32)))
                    over_data  = numpy.concatenate((over_data, numpy.take(data, dd)))

    if comm.rank == 0:
        print_and_log(["Overlaps computed, now gathering data by MPI"], 'debug', logger)

    comm.Barrier()

    #We need to gather the sparse arrays
    over_x    = gather_array(over_x, comm, dtype='int32')
    over_y    = gather_array(over_y, comm, dtype='int32')
    over_data = gather_array(over_data, comm)

    #We need to add the transpose matrices

    if comm.rank == 0:
        hfile      = h5py.File(filename, 'w', libver='latest')
        hfile.create_dataset('over_x', data=over_x)
        hfile.create_dataset('over_y', data=over_y)
        hfile.create_dataset('over_data', data=over_data)
        hfile.create_dataset('over_shape', data=numpy.array([N_tm**2, 2*N_t - 1], dtype=numpy.int32))
        hfile.close()

        if maxoverlap:

            overlap    = scipy.sparse.csr_matrix((over_data, (over_x, over_y)), shape=(N_tm**2, 2*N_t - 1))
            myfile     = h5py.File(filename, 'r+', libver='latest')
            myfile2    = h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r+', libver='latest')
            if 'maxoverlap' in myfile2.keys():
                maxoverlap = myfile2.get('maxoverlap')
            else:
                maxoverlap = myfile2.create_dataset('maxoverlap', shape=(N_tm, N_tm), dtype=numpy.float32)
            if 'maxlag' in myfile2.keys():
                maxlag = myfile2.get('maxlag')
            else:
                maxlag = myfile2.create_dataset('maxlag', shape=(N_tm, N_tm), dtype=numpy.int32)

            for i in xrange(N_tm-1):
                data                = overlap[i*N_tm+i+1:(i+1)*N_tm].toarray()
                maxlag[i, i+1:]     = N_t - numpy.argmax(data, 1)
                maxlag[i+1:, i]     = maxlag[i, i+1:]
                maxoverlap[i, i+1:] = numpy.max(data, 1)
                maxoverlap[i+1:, i] = maxoverlap[i, i+1:]
            myfile.close()
            myfile2.close()

    comm.Barrier()
    return h5py.File(filename, 'r')
