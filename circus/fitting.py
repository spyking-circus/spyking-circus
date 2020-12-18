from circus.shared.utils import *
import circus.shared.files as io
from circus.shared.files import get_dead_times
from circus.shared.probes import get_nodes_and_edges
from circus.shared.messages import print_and_log, init_logging
from circus.shared.mpi import detect_memory
import time

def main(params, nb_cpu, nb_gpu, use_gpu):

    #################################################################
    # params = detect_memory(params)
    _ = init_logging(params.logfile)
    SHARED_MEMORY = get_shared_memory_flag(params)
    logger = logging.getLogger('circus.fitting')
    data_file = params.data_file
    n_e = params.getint('data', 'N_e')
    n_total = params.nb_channels
    n_t = params.getint('detection', 'N_t')
    template_shift = params.getint('detection', 'template_shift')
    # file_out = params.get('data', 'file_out')
    file_out_suff = params.get('data', 'file_out_suff')
    sign_peaks = params.get('detection', 'peaks')
    matched_filter = params.getboolean('detection', 'matched-filter')
    # spike_thresh = params.getfloat('detection', 'spike_thresh')
    ratio_thresh = params.getfloat('fitting', 'ratio_thresh')
    two_components = params.getboolean('fitting', 'two_components')
    sparse_threshold = params.getfloat('fitting', 'sparse_thresh')
    # spike_width = params.getfloat('detection', 'spike_width')
    # dist_peaks = params.getint('detection', 'dist_peaks')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening = params.getboolean('whitening', 'spatial')
    templates_normalization = params.getboolean('clustering', 'templates_normalization')  # TODO test, switch, test!
    chunk_size = detect_memory(params, fitting=True)
    gpu_only = params.getboolean('fitting', 'gpu_only')
    nodes, edges = get_nodes_and_edges(params)
    tmp_limits = params.get('fitting', 'amp_limits').replace('(', '').replace(')', '').split(',')
    tmp_limits = [float(v) for v in tmp_limits]
    amp_auto = params.getboolean('fitting', 'amp_auto')
    max_chunk = params.getfloat('fitting', 'max_chunk')
    # noise_thr = params.getfloat('clustering', 'noise_thr')
    collect_all = params.getboolean('fitting', 'collect_all')
    min_second_component = params.getfloat('fitting', 'min_second_component')
    debug = params.getboolean('fitting', 'debug')
    ignore_dead_times = params.getboolean('triggers', 'ignore_times')
    inv_nodes = numpy.zeros(n_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.arange(len(nodes))
    data_file.open()
    supports = io.load_data(params, 'supports')
    low_channels_thr = params.getint('detection', 'low_channels_thr')
    median_channels = numpy.median(numpy.sum(supports, 1))
    fixed_amplitudes = params.getboolean('clustering', 'fixed_amplitudes')

    if not fixed_amplitudes:
        nb_amp_bins = params.getint('clustering', 'nb_amp_bins')
        splits = np.linspace(0, params.data_file.duration, nb_amp_bins)
        interpolated_times = np.zeros(len(splits) - 1, dtype=numpy.float32)
        for count in range(0, len(splits) - 1):
            interpolated_times[count] = (splits[count] + splits[count + 1])/2
        interpolated_times = numpy.concatenate(([0], interpolated_times, [params.data_file.duration]))
        nb_amp_times = len(splits) + 1

    mse_error = params.getboolean('fitting', 'mse_error')
    if mse_error:
        stds = io.load_data(params, 'stds')
        stds_norm = numpy.linalg.norm(stds)
    # if median_channels < low_channels_thr:
    #     normalization = False
    #     if comm.rank == 0:
    #         print_and_log(['Templates defined on few channels (%g), turning off normalization' %median_channels], 'debug', logger)

    #################################################################

    if SHARED_MEMORY:
        templates, mpi_memory_1 = io.load_data_memshared(params, 'templates', normalize=templates_normalization, transpose=True, sparse_threshold=sparse_threshold)
        N_tm, x = templates.shape
        is_sparse = not isinstance(templates, numpy.ndarray)
    else:
        templates = io.load_data(params, 'templates')
        x, N_tm = templates.shape
        if N_tm > 0:
            sparsity = templates.nnz / (x * N_tm)
            is_sparse = sparsity < sparse_threshold
        else:
            is_sparse = True
        if not is_sparse:
            if comm.rank == 0:
                print_and_log(['Templates sparsity is low (%g): densified to speedup the algorithm' %sparsity], 'debug', logger)
            templates = templates.toarray()

    temp_2_shift = 2 * template_shift
    temp_3_shift = 3 * template_shift
    n_tm = N_tm // 2
    n_scalar = n_e * n_t

    temp_window = numpy.arange(-template_shift, template_shift + 1)
    size_window = n_e * (2 * template_shift + 1)

    if not amp_auto:
        amp_limits = numpy.zeros((n_tm, 2))
        amp_limits[:, 0] = tmp_limits[0]
        amp_limits[:, 1] = tmp_limits[1]
    else:
        amp_limits = io.load_data(params, 'limits')

    norm_templates = io.load_data(params, 'norm-templates')
    sub_norm_templates = n_scalar * norm_templates[:n_tm]
    sub_norm_templates_full = n_scalar * norm_templates[:, numpy.newaxis]
    if not templates_normalization:
        norm_templates_2 = (norm_templates ** 2.0) * n_scalar
        sub_norm_templates_2 = norm_templates_2[:n_tm]
        sub_norm_templates_2_full = n_scalar*(norm_templates ** 2.0)[:, numpy.newaxis]


    if not SHARED_MEMORY:
        # Normalize templates (if necessary).
        if templates_normalization:
            if is_sparse:
                for idx in range(templates.shape[1]):
                    myslice = numpy.arange(templates.indptr[idx], templates.indptr[idx+1])
                    templates.data[myslice] /= norm_templates[idx]
            else:
                for idx in range(templates.shape[1]):
                    templates[:, idx] /= norm_templates[idx]
        # Transpose templates.
        templates = templates.T

    maxoverlap = io.load_data(params, 'maxoverlap')/n_scalar
    similar = np.where(maxoverlap > 0.5)

    idx = similar[0] < similar[1]
    similar = similar[0][idx], similar[1][idx]
    nb_mixtures = len(similar[0])

    waveform_neg = numpy.empty(0)  # default assignment (for PyCharm code inspection)
    matched_thresholds_neg = None  # default assignment (for PyCharm code inspection)
    waveform_pos = numpy.empty(0)  # default assignment (for PyCharm code inspection)
    matched_thresholds_pos = None  # default assignment (for PyCharm code inspection)
    if matched_filter:
        if sign_peaks in ['negative', 'both']:
            waveform_neg = io.load_data(params, 'waveform')[::-1]
            waveform_neg /= (numpy.abs(numpy.sum(waveform_neg)) * len(waveform_neg))
            matched_thresholds_neg = io.load_data(params, 'matched-thresholds')
        if sign_peaks in ['positive', 'both']:
            waveform_pos = io.load_data(params, 'waveform-pos')[::-1]
            waveform_pos /= (numpy.abs(numpy.sum(waveform_pos)) * len(waveform_pos))
            matched_thresholds_pos = io.load_data(params, 'matched-thresholds-pos')

    if ignore_dead_times:
        if SHARED_MEMORY:
            all_dead_times, mpi_memory_3 = get_dead_times(params)
        else:
            all_dead_times = get_dead_times(params)
    else:
        all_dead_times = None  # default assignment (for PyCharm code inspection)

    thresholds = io.get_accurate_thresholds(params, ratio_thresh)

    neighbors = {}
    if collect_all:
        is_sparse = not isinstance(templates, numpy.ndarray)
        for i in range(0, n_tm):
            if is_sparse:
                tmp = templates[i, :].toarray().reshape(n_e, n_t)
            else:
                tmp = templates[i].reshape(n_e, n_t)
            if templates_normalization:
                tmp = tmp * norm_templates[i]
            neighbors[i] = numpy.where(numpy.sum(tmp, axis=1) != 0.0)[0]

    #N_tm, x = templates.shape
    #sparsity_factor = templates.nnz / (N_tm * x)
    #if sparsity_factor > sparse_threshold:
    #    if comm.rank == 0:
    #        print_and_log(['Templates are not sparse enough, we densify them for'], 'default', logger)
    #    templates = templates.toarray()

    info_string = ''

    if comm.rank == 0:
        info_string = "using %d CPUs" % comm.size

    comm.Barrier()

    c_overlap = io.get_overlaps(params, nb_cpu=nb_cpu, nb_gpu=nb_gpu, use_gpu=use_gpu)
    over_shape = c_overlap.get('over_shape')[:]
    n_over = int(numpy.sqrt(over_shape[0]))
    s_over = over_shape[1]
    s_center = s_over // 2
    # # If the number of overlaps is different from templates, we need to recompute them.
    if n_over != N_tm:
        if comm.rank == 0:
            print_and_log(['Templates have been modified, recomputing the overlaps...'], 'default', logger)
        c_overlap = io.get_overlaps(params, erase=True, nb_cpu=nb_cpu, nb_gpu=nb_gpu, use_gpu=use_gpu)
        over_shape = c_overlap.get('over_shape')[:]
        n_over = int(numpy.sqrt(over_shape[0]))
        s_over = over_shape[1]

    if SHARED_MEMORY:
        c_overs, mpi_memory_2 = io.load_data_memshared(params, 'overlaps')
    else:
        c_overs = io.load_data(params, 'overlaps')

    comm.Barrier()

    if n_tm == 0:
        if comm.rank == 0:
            print_and_log(["No templates present. Redo clustering?"], 'default', logger)

        sys.exit(0)

    if comm.rank == 0:
        print_and_log(["Here comes the SpyKING CIRCUS %s and %d templates..." % (info_string, n_tm)], 'default', logger)
        purge(file_out_suff, '.data')

    if do_spatial_whitening:
        spatial_whitening = io.load_data(params, 'spatial_whitening')
    else:
        spatial_whitening = None  # default assignment (for PyCharm code inspection)
    if do_temporal_whitening:
        temporal_whitening = io.load_data(params, 'temporal_whitening')
    else:
        temporal_whitening = None  # default assignment (for PyCharm code inspection)

    nb_chunks, last_chunk_len = data_file.analyze(chunk_size)
    processed_chunks = int(min(nb_chunks, max_chunk))

    comm.Barrier()
    spiketimes_file = open(file_out_suff + '.spiketimes-%d.data' % comm.rank, 'wb')
    comm.Barrier()
    amplitudes_file = open(file_out_suff + '.amplitudes-%d.data' % comm.rank, 'wb')
    comm.Barrier()
    templates_file = open(file_out_suff + '.templates-%d.data' % comm.rank, 'wb')
    comm.Barrier()


    if mse_error:
        mse_file = open(file_out_suff + '.mses-%d.data' % comm.rank, 'wb')
        comm.Barrier()

    if collect_all:
        garbage_times_file = open(file_out_suff + '.gspiketimes-%d.data' % comm.rank, 'wb')
        comm.Barrier()
        garbage_temp_file = open(file_out_suff + '.gtemplates-%d.data' % comm.rank, 'wb')
        comm.Barrier()
    else:
        garbage_times_file = None  # default assignment (for PyCharm code inspection)
        garbage_temp_file = None  # default assignment (for PyCharm code inspection)

    if debug:
        # Open debug files.
        chunk_nbs_debug_file = open(file_out_suff + '.chunk_nbs_debug_%d.data' % comm.rank, mode='wb')
        comm.Barrier()
        iteration_nbs_debug_file = open(file_out_suff + '.iteration_nbs_debug_%d.data' % comm.rank, mode='wb')
        comm.Barrier()
        peak_nbs_debug_file = open(file_out_suff + '.peak_nbs_debug_%d.data' % comm.rank, mode='wb')
        comm.Barrier()
        peak_local_time_steps_debug_file = open(
            file_out_suff + '.peak_local_time_steps_debug_%d.data' % comm.rank, mode='wb'
        )
        comm.Barrier()
        peak_time_steps_debug_file = open(file_out_suff + '.peak_time_steps_debug_%d.data' % comm.rank, mode='wb')
        comm.Barrier()
        peak_scalar_products_debug_file = open(
            file_out_suff + '.peak_scalar_products_debug_%d.data' % comm.rank, mode='wb'
        )
        comm.Barrier()
        peak_solved_flags_debug_file = open(file_out_suff + '.peak_solved_flags_debug_%d.data' % comm.rank, mode='wb')
        comm.Barrier()
        template_nbs_debug_file = open(file_out_suff + '.template_nbs_debug_%d.data' % comm.rank, mode='wb')
        comm.Barrier()
        success_flags_debug_file = open(file_out_suff + '.success_flags_debug_%d.data' % comm.rank, mode='wb')
        comm.Barrier()
    else:
        chunk_nbs_debug_file = None  # default assignment (for PyCharm code inspection)
        iteration_nbs_debug_file = None  # default assignment (for PyCharm code inspection)
        peak_nbs_debug_file = None  # default assignment (for PyCharm code inspection)
        peak_local_time_steps_debug_file = None  # default assignment (for PyCharm code inspection)
        peak_time_steps_debug_file = None  # default assignment (for PyCharm code inspection)
        peak_scalar_products_debug_file = None  # default assignment (for PyCharm code inspection)
        peak_solved_flags_debug_file = None  # default assignment (for PyCharm code inspection)
        template_nbs_debug_file = None  # default assignment (for PyCharm code inspection)
        success_flags_debug_file = None  # default assignment (for PyCharm code inspection)

    last_chunk_size = 0
    slice_indices = numpy.zeros(0, dtype=numpy.int32)

    to_explore = range(comm.rank, processed_chunks, comm.size)

    if comm.rank == 0:
        to_explore = get_tqdm_progressbar(params, to_explore)

    if fixed_amplitudes:
        min_scalar_products = amp_limits[:,0][:, numpy.newaxis]
        max_scalar_products = amp_limits[:,1][:, numpy.newaxis]

    if templates_normalization:
        min_sps = sub_norm_templates[:, numpy.newaxis]
    else:
        min_sps = sub_norm_templates_2[:, numpy.newaxis]

    for gcount, gidx in enumerate(to_explore):
        # print "Node", comm.rank, "is analyzing chunk", gidx, "/", nb_chunks, " ..."
        # # We need to deal with the borders by taking chunks of size [0, chunck_size + template_shift].

        is_first = data_file.is_first_chunk(gidx, nb_chunks)
        is_last = data_file.is_last_chunk(gidx, nb_chunks)

        if not (is_first and is_last):
            if is_last:
                padding = (-temp_3_shift, 0)
            elif is_first:
                padding = (0, temp_3_shift)
            else:
                padding = (-temp_3_shift, temp_3_shift)
        else:
            padding = (0, 0)

        result = {
            'spiketimes': [],
            'amplitudes': [],
            'templates': [],
        }

        if mse_error:
            mse_fit = {
            'spiketimes': [],
            'amplitudes': [],
            'templates': [],
        }

        result_debug = {
            'chunk_nbs': [],
            'iteration_nbs': [],
            'peak_nbs': [],
            'peak_local_time_steps': [],
            'peak_time_steps': [],
            'peak_scalar_products': [],
            'peak_solved_flags': [],
            'template_nbs': [],
            'success_flags': [],
        }

        local_chunk, t_offset = data_file.get_data(gidx, chunk_size, padding, nodes=nodes)           
        len_chunk = len(local_chunk)
        if is_last:
            my_chunk_size = last_chunk_size
        else:
            my_chunk_size = chunk_size

        if do_spatial_whitening:
            local_chunk = numpy.dot(local_chunk, spatial_whitening)
        if do_temporal_whitening:
            local_chunk = scipy.ndimage.filters.convolve1d(local_chunk, temporal_whitening, axis=0, mode='constant')

        # Extracting peaks.

        all_found_spikes = {}
        if collect_all:
            for i in range(n_e):
                all_found_spikes[i] = []

        local_peaktimes = [numpy.empty(0, dtype=numpy.uint32)]

        if matched_filter:
            if sign_peaks in ['positive', 'both']:
                filter_chunk = scipy.ndimage.filters.convolve1d(local_chunk, waveform_pos, axis=0, mode='constant')
                for i in range(n_e):
                    peaktimes = scipy.signal.find_peaks(filter_chunk[:, i], height=matched_thresholds_pos[i])[0]
                    local_peaktimes.append(peaktimes)
                    if collect_all:
                        all_found_spikes[i] += peaktimes.tolist()
            if sign_peaks in ['negative', 'both']:
                filter_chunk = scipy.ndimage.filters.convolve1d(local_chunk, waveform_neg, axis=0, mode='constant')
                for i in range(n_e):
                    peaktimes = scipy.signal.find_peaks(filter_chunk[:, i], height=matched_thresholds_neg[i])[0]
                    local_peaktimes.append(peaktimes)
                    if collect_all:
                        all_found_spikes[i] += peaktimes.tolist()
            local_peaktimes = numpy.concatenate(local_peaktimes)
        else:
            for i in range(n_e):
                if sign_peaks == 'negative':
                    peaktimes = scipy.signal.find_peaks(-local_chunk[:, i], height=thresholds[i])[0]
                elif sign_peaks == 'positive':
                    peaktimes = scipy.signal.find_peaks(local_chunk[:, i], height=thresholds[i])[0]
                elif sign_peaks == 'both':
                    peaktimes = scipy.signal.find_peaks(numpy.abs(local_chunk[:, i]), height=thresholds[i])[0]
                else:
                    raise ValueError("Unexpected value %s" % sign_peaks)
                local_peaktimes.append(peaktimes)
                if collect_all:
                    all_found_spikes[i] += peaktimes.tolist()
            local_peaktimes = numpy.concatenate(local_peaktimes)

        local_peaktimes = numpy.unique(local_peaktimes)

        g_offset = t_offset + padding[0]

        if ignore_dead_times:
            dead_indices = numpy.searchsorted(all_dead_times, [t_offset, t_offset + my_chunk_size])
            if dead_indices[0] != dead_indices[1]:
                is_included = numpy.in1d(local_peaktimes + g_offset, all_dead_times[dead_indices[0]:dead_indices[1]])
                local_peaktimes = local_peaktimes[~is_included]
                local_peaktimes = numpy.sort(local_peaktimes)
        else:
            dead_indices = None  # default assignment (for PyCharm code inspection)

        # print "Removing the useless borders..."
        local_borders = (template_shift, len_chunk - template_shift)
        idx = (local_peaktimes >= local_borders[0]) & (local_peaktimes < local_borders[1])
        local_peaktimes = numpy.compress(idx, local_peaktimes)

        if collect_all:
            for i in range(n_e):
                all_found_spikes[i] = numpy.array(all_found_spikes[i], dtype=numpy.uint32)

                if ignore_dead_times:
                    if dead_indices[0] != dead_indices[1]:
                        is_included = numpy.in1d(
                            all_found_spikes[i] + g_offset, all_dead_times[dead_indices[0]:dead_indices[1]]
                        )
                        all_found_spikes[i] = all_found_spikes[i][~is_included]
                        all_found_spikes[i] = numpy.sort(all_found_spikes[i])

                idx = (all_found_spikes[i] >= local_borders[0]) & (all_found_spikes[i] < local_borders[1])
                all_found_spikes[i] = numpy.compress(idx, all_found_spikes[i])

        nb_local_peak_times = len(local_peaktimes)

        if nb_local_peak_times > 0:
            # print "Computing the b (should full_gpu by putting all chunks on GPU if possible?)..."

            if collect_all or mse_error:
                c_local_chunk = local_chunk.copy()
            else:
                c_local_chunk = None  # default assignment (for PyCharm code inspection)

            sub_mat = local_chunk[local_peaktimes[:, None] + temp_window]
            sub_mat = sub_mat.transpose(2, 1, 0).reshape(size_window, nb_local_peak_times)

            del local_chunk

            b = templates.dot(sub_mat)
    
            del sub_mat

            local_restriction = (t_offset, t_offset + my_chunk_size)
            all_spikes = local_peaktimes + g_offset

            if collect_all:
                c_all_times = numpy.zeros((len_chunk, n_e), dtype=numpy.bool)
                c_min_times = numpy.maximum(numpy.arange(len_chunk) - template_shift, 0)
                c_max_times = numpy.minimum(numpy.arange(len_chunk) + template_shift + 1, len_chunk)
                for i in range(n_e):
                    c_all_times[all_found_spikes[i], i] = True
            else:
                c_all_times = None  # default assignment (for PyCharm code inspection)
                c_min_times = None  # default assignment (for PyCharm code inspection)
                c_max_times = None  # default assignment (for PyCharm code inspection)

            iteration_nb = 0
            data = b[:n_tm, :]
            amplitudes = numpy.zeros(b.shape, dtype=numpy.float32)

            if not fixed_amplitudes:
                amp_index = numpy.searchsorted(splits, local_restriction[0], 'right')
                scaling = 1/(splits[amp_index] - splits[amp_index - 1])
                min_scalar_products = amp_limits[:, amp_index, 0] + (amp_limits[:, amp_index, 0] - amp_limits[:, amp_index+1, 0])*scaling
                max_scalar_products = amp_limits[:, amp_index, 1] + (amp_limits[:, amp_index, 1] - amp_limits[:, amp_index+1, 0])*scaling
                    
                min_scalar_products = min_scalar_products[:, numpy.newaxis]
                max_scalar_products = max_scalar_products[:, numpy.newaxis]

            is_valid = numpy.ones(data.shape, dtype=numpy.bool)
            valid_indices = numpy.where(is_valid)

            M = scipy.sparse.csr_matrix((0, 0), dtype=numpy.float32)
            selection = numpy.zeros((0, 2), dtype=numpy.int32)
            res_sps = numpy.zeros(0, dtype=numpy.float32)

            full_sps = b.copy()

            peaks_times = local_peaktimes - local_peaktimes[:, numpy.newaxis]
            all_neighbors = numpy.abs(peaks_times) <= temp_2_shift
            neighbors = {}
            for i in range(len(all_neighbors)):
                idx = numpy.where(all_neighbors[i])[0]
                neighbors[i] = {'idx' : idx, 'tdx' : peaks_times[i][idx] + temp_2_shift }


            while True:

                best_amplitude_idx = data[is_valid].argmax()
                best_template_index, peak_index = valid_indices[0][best_amplitude_idx], valid_indices[1][best_amplitude_idx]
                    
                to_add = numpy.array([[best_template_index, peak_index], [best_template_index+n_tm, peak_index]])
                selection = numpy.vstack((selection, to_add))

                res_sps = numpy.concatenate((res_sps, full_sps[selection[-2:, 0], selection[-2:, 1]]))

                b[best_template_index, peak_index] = -numpy.inf

                nb_selection = len(selection)
    
                delta_t = local_peaktimes[selection[:, 1]] - local_peaktimes[selection[-1, 1]]
                idx = numpy.where(numpy.abs(delta_t) <= temp_2_shift)[0]

                M.resize(nb_selection, nb_selection)
                line = temp_2_shift + delta_t[idx]

                M[-2, idx] = c_overs[selection[-2, 0]][selection[idx, 0], line]
                M[:, -2] = M[-2].T

                M[-1, idx] = c_overs[selection[-1, 0]][selection[idx, 0], line]
                M[:, -1] = M[-1].T

                all_amplitudes = scipy.sparse.linalg.spsolve(M, res_sps)/norm_templates[selection[:, 0]]

                diff_amplitudes   = (all_amplitudes[::2] - amplitudes[selection[::2,0], selection[::2, 1]])
                diff_amplitudes_2 = (all_amplitudes[1::2] - amplitudes[selection[1::2,0], selection[1::2, 1]])

                modified = numpy.where(numpy.abs(diff_amplitudes) > 1e-3)[0]
                
                amplitudes[selection[:,0], selection[:,1]] = all_amplitudes
                for i in modified:
                    
                    tmp_best = selection[2*i, 0]
                    tmp_peak = selection[2*i, 1]

                    best_amp = diff_amplitudes[i]*norm_templates[tmp_best]
                    best_amp2 = diff_amplitudes_2[i]*norm_templates[tmp_best + n_tm]

                    idx = neighbors[tmp_peak]['idx']
                    tdx = neighbors[tmp_peak]['tdx']

                    tmp1 = c_overs[tmp_best].multiply(-best_amp)
                    tmp1 += c_overs[tmp_best + n_tm].multiply(-best_amp2)

                    to_add = tmp1.toarray()[:, tdx]
                    b[:, idx] += to_add

                is_valid = data > 0.25*min_sps
                valid_indices = numpy.where(is_valid)

                if len(valid_indices[0]) == 0:
                    break

            are_valid = (amplitudes[:n_tm] > min_scalar_products)*(amplitudes[:n_tm] < max_scalar_products)
            valid_indices = numpy.where(are_valid)

            if len(valid_indices[0]) > 0:
            
                for i,j in zip(valid_indices[0], valid_indices[1]):
                    # Add matching to the result.
                    t_spike = all_spikes[j]

                    if (t_spike >= local_restriction[0]) and (t_spike < local_restriction[1]):
                        result['spiketimes'] += [t_spike]
                        result['amplitudes'] += [(amplitudes[i,j], amplitudes[i + n_tm,j])]
                        result['templates'] += [i]
                    elif mse_error:
                        mse_fit['spiketimes'] += [t_spike]
                        mse_fit['amplitudes'] += [(amplitudes[i,j], amplitudes[i + n_tm,j])]
                        mse_fit['templates'] += [i]

                    # Save debug data.
                    if debug:
                        result_debug['chunk_nbs'] += [gidx]
                        result_debug['iteration_nbs'] += [iteration_nb]
                        result_debug['peak_nbs'] += [peak_index]
                        result_debug['peak_local_time_steps'] += [local_peaktimes[peak_index]]
                        result_debug['peak_time_steps'] += [all_spikes[peak_index]]
                        result_debug['peak_scalar_products'] += [peak_scalar_product]
                        result_debug['peak_solved_flags'] += [b[best_template_index, peak_index]]
                        result_debug['template_nbs'] += [best_template_index]
                        result_debug['success_flags'] += [True]

            iteration_nb += 1

            spikes_to_write = numpy.array(result['spiketimes'], dtype=numpy.uint32)
            amplitudes_to_write = numpy.array(result['amplitudes'], dtype=numpy.float32)
            templates_to_write = numpy.array(result['templates'], dtype=numpy.uint32)

            spiketimes_file.write(spikes_to_write.tostring())
            amplitudes_file.write(amplitudes_to_write.tostring())
            templates_file.write(templates_to_write.tostring())

            if mse_error:
                curve = numpy.zeros((len_chunk, n_e), dtype=numpy.float32)
                for spike, temp_id, amplitude in zip(result['spiketimes'], result['templates'], result['amplitudes']):
                    spike = spike - t_offset - padding[0]
                    if is_sparse:
                        tmp1 = templates[temp_id].toarray().reshape(n_e, n_t)
                        tmp2 = templates[temp_id + n_tm].toarray().reshape(n_e, n_t)
                    else:
                        tmp1 = templates[temp_id].reshape(n_e, n_t)
                        tmp2 = templates[temp_id + n_tm].reshape(n_e, n_t)

                    curve[spike - template_shift:spike + template_shift + 1, :] += (amplitude[0] * tmp1 + amplitude[1] * tmp2).T

                for spike, temp_id, amplitude in zip(mse_fit['spiketimes'], mse_fit['templates'], mse_fit['amplitudes']):
                    spike = spike - t_offset + padding[0]
                    if is_sparse:
                        tmp1 = templates[temp_id].toarray().reshape(n_e, n_t)
                        tmp2 = templates[temp_id + n_tm].toarray().reshape(n_e, n_t)
                    else:
                        tmp1 = templates[temp_id].reshape(n_e, n_t)
                        tmp2 = templates[temp_id + n_tm].reshape(n_e, n_t)
                    try:
                        curve[int(spike) - template_shift:int(spike) + template_shift + 1, :] += (amplitude[0] * tmp1 + amplitude[1] * tmp2).T
                    except Exception:
                        pass
                mse = numpy.linalg.norm((curve - c_local_chunk)[-padding[0]:-padding[1]])
                nb_points = len(curve) - (padding[1] - padding[0])
                mse_ratio = mse/(numpy.sqrt(nb_points)*stds_norm)
                mse_to_write = numpy.array([g_offset, mse_ratio], dtype=numpy.float32)
                mse_file.write(mse_to_write.tostring())

            if collect_all:

                for temp, spike in zip(templates_to_write, spikes_to_write - g_offset):
                    c_all_times[c_min_times[spike]:c_max_times[spike], neighbors[temp]] = False

                gspikes = numpy.where(numpy.sum(c_all_times, 1) > 0)[0]
                c_all_times = numpy.take(c_all_times, gspikes, axis=0)
                c_local_chunk = numpy.take(c_local_chunk, gspikes, axis=0) * c_all_times                

                if sign_peaks == 'negative':
                    bestlecs = numpy.argmin(c_local_chunk, 1)
                    if matched_filter:
                        threshs = -matched_thresholds_neg[bestlecs]
                    else:
                        threshs = -thresholds[bestlecs]
                    idx = numpy.where(numpy.min(c_local_chunk, 1) < threshs)[0]
                elif sign_peaks == 'positive':
                    bestlecs = numpy.argmax(c_local_chunk, 1)
                    if matched_filter:
                        threshs = matched_thresholds_pos[bestlecs]
                    else:
                        threshs = thresholds[bestlecs]
                    idx = numpy.where(numpy.max(c_local_chunk, 1) > threshs)[0]
                elif sign_peaks == 'both':
                    c_local_chunk = numpy.abs(c_local_chunk)
                    bestlecs = numpy.argmax(c_local_chunk, 1)
                    if matched_filter:
                        threshs = numpy.minimum(matched_thresholds_neg[bestlecs], matched_thresholds_pos[bestlecs])
                    else:
                        threshs = thresholds[bestlecs]
                    idx = numpy.where(numpy.max(c_local_chunk, 1) > threshs)[0]
                else:
                    raise ValueError("Unexpected value %s" % sign_peaks)

                gspikes = numpy.take(gspikes, idx)
                bestlecs = numpy.take(bestlecs, idx)
                gspikes_to_write = numpy.array(gspikes + g_offset, dtype=numpy.uint32)
                gtemplates_to_write = numpy.array(bestlecs, dtype=numpy.uint32)

                garbage_times_file.write(gspikes_to_write.tostring())
                garbage_temp_file.write(gtemplates_to_write.tostring())

            if debug:
                # Write debug data to debug files.
                for field_label, field_dtype, field_file in [
                    ('chunk_nbs', numpy.uint32, chunk_nbs_debug_file),
                    ('iteration_nbs', numpy.uint32, iteration_nbs_debug_file),
                    ('peak_nbs', numpy.uint32, peak_nbs_debug_file),
                    ('peak_local_time_steps', numpy.uint32, peak_local_time_steps_debug_file),
                    ('peak_time_steps', numpy.uint32, peak_time_steps_debug_file),
                    ('peak_scalar_products', numpy.float32, peak_scalar_products_debug_file),
                    ('peak_solved_flags', numpy.float32, peak_solved_flags_debug_file),
                    ('template_nbs', numpy.uint32, template_nbs_debug_file),
                    ('success_flags', numpy.bool, success_flags_debug_file),
                ]:
                    field_to_write = numpy.array(result_debug[field_label], dtype=field_dtype)
                    field_file.write(field_to_write.tostring())

    sys.stderr.flush()

    spiketimes_file.flush()
    os.fsync(spiketimes_file.fileno())
    spiketimes_file.close()

    amplitudes_file.flush()
    os.fsync(amplitudes_file.fileno())
    amplitudes_file.close()

    templates_file.flush()
    os.fsync(templates_file.fileno())
    templates_file.close()

    if collect_all:

        garbage_temp_file.flush()
        os.fsync(garbage_temp_file.fileno())
        garbage_temp_file.close()
        
        garbage_times_file.flush()
        os.fsync(garbage_times_file.fileno())
        garbage_times_file.close()

    if mse_error:
        mse_file.flush()
        os.fsync(mse_file.fileno())
        mse_file.close()

    if debug:
        # Close debug files.
        for field_file in [
            chunk_nbs_debug_file,
            iteration_nbs_debug_file,
            peak_nbs_debug_file,
            peak_local_time_steps_debug_file,
            peak_time_steps_debug_file,
            peak_scalar_products_debug_file,
            peak_solved_flags_debug_file,
            template_nbs_debug_file,
            success_flags_debug_file,
        ]:
            field_file.flush()
            os.fsync(field_file.fileno())
            field_file.close()

    comm.Barrier()

    if SHARED_MEMORY:
        for memory in mpi_memory_1 + mpi_memory_2:
            memory.Free()
        if ignore_dead_times:
            mpi_memory_3.Free()

    if comm.rank == 0:
        io.collect_data(comm.size, params, erase=True)

    data_file.close()
