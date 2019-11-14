import circus.shared.algorithms as algo
from .shared.utils import *
from .shared.files import get_dead_times
from .shared.probes import get_nodes_and_edges
from circus.shared.messages import print_and_log, init_logging
from circus.shared.mpi import detect_memory

def main(params, nb_cpu, nb_gpu, use_gpu):

    #################################################################
    #params         = detect_memory(params)
    logger         = init_logging(params.logfile)
    SHARED_MEMORY  = get_shared_memory_flag(params)
    logger         = logging.getLogger('circus.fitting')
    data_file      = params.data_file
    N_e            = params.getint('data', 'N_e')
    N_total        = params.nb_channels
    N_t            = params.getint('detection', 'N_t')
    template_shift = params.getint('detection', 'template_shift')
    file_out       = params.get('data', 'file_out')
    file_out_suff  = params.get('data', 'file_out_suff')
    sign_peaks     = params.get('detection', 'peaks')
    matched_filter = params.getboolean('detection', 'matched-filter')
    spike_thresh   = params.getfloat('detection', 'spike_thresh')
    spike_width    = params.getfloat('detection', 'spike_width')
    dist_peaks     = params.getint('detection', 'dist_peaks')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    chunk_size     = detect_memory(params, fitting=True)
    gpu_only       = params.getboolean('fitting', 'gpu_only')
    nodes, edges   = get_nodes_and_edges(params)
    tmp_limits     = params.get('fitting', 'amp_limits').replace('(', '').replace(')', '').split(',')
    tmp_limits     = map(float, tmp_limits)
    amp_auto       = params.getboolean('fitting', 'amp_auto')
    nb_chances     = params.getint('fitting', 'nb_chances')
    max_chunk      = params.getfloat('fitting', 'max_chunk')
    noise_thr      = params.getfloat('clustering', 'noise_thr')
    collect_all    = params.getboolean('fitting', 'collect_all')
    ignore_dead_times = params.getboolean('triggers', 'ignore_times')
    inv_nodes         = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes]  = numpy.arange(len(nodes))
    data_file.open()
    #################################################################

    if use_gpu:
        import cudamat as cmt
        ## Need to properly handle multi GPU per MPI nodes?
        if nb_gpu > nb_cpu:
            gpu_id = int(comm.rank//nb_cpu)
        else:
            gpu_id = 0
        cmt.cuda_set_device(gpu_id)
        cmt.init()
        cmt.cuda_sync_threads()
        
    if SHARED_MEMORY:
        templates  = io.load_data_memshared(params, 'templates', normalize=True, transpose=True)
        N_tm, x    = templates.shape
    else:
        templates  = io.load_data(params, 'templates')
        x, N_tm    = templates.shape

    temp_2_shift   = 2*template_shift
    temp_3_shift   = 3*template_shift
    full_gpu       = use_gpu and gpu_only
    n_tm           = N_tm//2
    n_scalar       = N_e*N_t

    temp_window    = numpy.arange(-template_shift, template_shift+1)
    size_window    = N_e*(2*template_shift+1)

    if not amp_auto:
        amp_limits       = numpy.zeros((n_tm, 2))
        amp_limits[:, 0] = tmp_limits[0]
        amp_limits[:, 1] = tmp_limits[1]
    else:
        amp_limits       = io.load_data(params, 'limits')

    norm_templates = io.load_data(params, 'norm-templates')

    if not SHARED_MEMORY:
        for idx in xrange(templates.shape[1]):
            myslice = numpy.arange(templates.indptr[idx], templates.indptr[idx+1])
            templates.data[myslice] /= norm_templates[idx]
        templates = templates.T

    if matched_filter:
        if sign_peaks in ['negative', 'both']:
            waveform_neg  = io.load_data(params, 'waveform')[::-1]
            waveform_neg /= (numpy.abs(numpy.sum(waveform_neg))* len(waveform_neg))
            matched_tresholds_neg = io.load_data(params, 'matched-thresholds')
        if sign_peaks in ['positive', 'both']:
            waveform_pos  = io.load_data(params, 'waveform-pos')[::-1]
            waveform_pos /= (numpy.abs(numpy.sum(waveform_pos))* len(waveform_pos))
            matched_tresholds_pos = io.load_data(params, 'matched-thresholds-pos')

    if ignore_dead_times:
        all_dead_times = get_dead_times(params)

    thresholds = io.load_data(params, 'thresholds')


    if collect_all:
        neighbors = {}
        for i in xrange(n_tm):
            tmp  = templates[i, :].toarray().reshape(N_e, N_t) * norm_templates[i]
            neighbors[i] = numpy.where(numpy.sum(tmp, 1) != 0)[0]

    if use_gpu:
        templates = cmt.SparseCUDAMatrix(templates, copy_on_host=False)

    info_string   = ''

    
    if comm.rank == 0:
        if use_gpu:
            info_string = "using %d GPUs" %(comm.size)
        else:
            info_string = "using %d CPUs" %(comm.size)

    comm.Barrier()

    c_overlap  = io.get_overlaps(params, nb_cpu=nb_cpu, nb_gpu=nb_gpu, use_gpu=use_gpu)
    over_shape = c_overlap.get('over_shape')[:]
    N_over     = int(numpy.sqrt(over_shape[0]))
    S_over     = over_shape[1]
    ## If the number of overlaps is different from templates, we need to recompute them
    if N_over != N_tm:
        if comm.rank == 0:
            print_and_log(['Templates have been modified, recomputing the overlaps...'], 'default', logger)
        c_overlap  = io.get_overlaps(params, erase=True, nb_cpu=nb_cpu, nb_gpu=nb_gpu, use_gpu=use_gpu)
        over_shape = c_overlap.get('over_shape')[:]
        N_over     = int(numpy.sqrt(over_shape[0]))
        S_over     = over_shape[1]

    if SHARED_MEMORY:
        c_overs    = io.load_data_memshared(params, 'overlaps')
    else:
        c_overs    = io.load_data(params, 'overlaps')

    comm.Barrier()

    if n_tm == 0:
        if comm.rank == 0:
            print_and_log(["No templates present. Redo clustering?"], 'default', logger)

        sys.exit(0)

    if comm.rank == 0:
        print_and_log(["Here comes the SpyKING CIRCUS %s and %d templates..." %(info_string, n_tm)], 'default', logger)
        purge(file_out_suff, '.data')

    if do_spatial_whitening:
        spatial_whitening  = io.load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = io.load_data(params, 'temporal_whitening')

    if full_gpu:
        try:
            # If memory on the GPU is large enough, we load the overlaps onto it
            for i in xrange(N_over):
                c_overs[i] = cmt.SparseCUDAMatrix(c_overs[i], copy_on_host=False)
        except Exception:
            if comm.rank == 0:
                print_and_log(["Not enough memory on GPUs: GPUs are used for projection only"], 'info', logger)
            for i in xrange(N_over):
                if c_overs.has_key(i):
                    del c_overs[i]
            full_gpu = False

    nb_chunks, last_chunk_len = data_file.analyze(chunk_size)
    processed_chunks          = int(min(nb_chunks, max_chunk))

    comm.Barrier()
    spiketimes_file = open(file_out_suff + '.spiketimes-%d.data' %comm.rank, 'wb')
    comm.Barrier()
    amplitudes_file = open(file_out_suff + '.amplitudes-%d.data' %comm.rank, 'wb')
    comm.Barrier()
    templates_file  = open(file_out_suff + '.templates-%d.data' %comm.rank, 'wb')
    comm.Barrier()

    if collect_all:
        garbage_times_file = open(file_out_suff + '.gspiketimes-%d.data' %comm.rank, 'wb')
        comm.Barrier()
        garbage_temp_file  = open(file_out_suff + '.gtemplates-%d.data' %comm.rank, 'wb')
        comm.Barrier()


    if use_gpu and do_spatial_whitening:
        spatial_whitening = cmt.CUDAMatrix(spatial_whitening, copy_on_host=False)

    last_chunk_size = 0

    to_explore = xrange(comm.rank, processed_chunks, comm.size)

    if comm.rank == 0:
        to_explore = get_tqdm_progressbar(to_explore)

    for gcount, gidx in enumerate(to_explore):
        #print "Node", comm.rank, "is analyzing chunk", gidx, "/", nb_chunks, " ..."
        ## We need to deal with the borders by taking chunks of size [0, chunck_size+template_shift]

        is_first = data_file.is_first_chunk(gidx, nb_chunks)
        is_last  = data_file.is_last_chunk(gidx, nb_chunks)

        if is_last:
            padding = (-temp_2_shift, 0)
        elif is_first:
            padding = (0, temp_2_shift)
        else:
            padding = (-temp_3_shift, temp_3_shift)

        result       = {'spiketimes' : [], 'amplitudes' : [], 'templates' : []}

        local_chunk, t_offset = data_file.get_data(gidx, chunk_size, padding, nodes=nodes)           
        len_chunk             = len(local_chunk)

        if do_spatial_whitening:
            if use_gpu:
                local_chunk = cmt.CUDAMatrix(local_chunk, copy_on_host=False)
                local_chunk = local_chunk.dot(spatial_whitening).asarray()
            else:
                local_chunk = numpy.dot(local_chunk, spatial_whitening)
        if do_temporal_whitening:
            local_chunk = scipy.ndimage.filters.convolve1d(local_chunk, temporal_whitening, axis=0, mode='constant')

        #print "Extracting the peaks..."

        if collect_all:
            all_found_spikes = {}
            for i in xrange(N_e):
                all_found_spikes[i] = []

        local_peaktimes = numpy.zeros(0, dtype=numpy.uint32)

        if matched_filter:
            if sign_peaks in ['positive', 'both']:
                filter_chunk = scipy.ndimage.filters.convolve1d(local_chunk, waveform_pos, axis=0, mode='constant')
                for i in xrange(N_e):
                    peaktimes = algo.detect_peaks(filter_chunk[:, i], matched_tresholds_pos[i])
                    local_peaktimes = numpy.concatenate((local_peaktimes, peaktimes))
                    if collect_all:
                        all_found_spikes[i] += peaktimes.tolist()
            if sign_peaks in ['negative', 'both']:
                filter_chunk = scipy.ndimage.filters.convolve1d(local_chunk, waveform_neg, axis=0, mode='constant')
                for i in xrange(N_e):
                    peaktimes = algo.detect_peaks(filter_chunk[:, i], matched_tresholds_neg[i])
                    local_peaktimes = numpy.concatenate((local_peaktimes, peaktimes))
                    if collect_all:
                        all_found_spikes[i] += peaktimes.tolist()
        else:
            for i in xrange(N_e):
                if sign_peaks == 'negative':
                    peaktimes = scipy.signal.find_peaks(-local_chunk[:, i], height=thresholds[i])[0]
                elif sign_peaks == 'positive':
                    peaktimes = scipy.signal.find_peaks(local_chunk[:, i], height=thresholds[i])[0]
                elif sign_peaks == 'both':
                    peaktimes = scipy.signal.find_peaks(numpy.abs(local_chunk[:, i]), height=thresholds[i])[0]
                local_peaktimes = numpy.concatenate((local_peaktimes, peaktimes)) 
                if collect_all:
                    all_found_spikes[i] += peaktimes.tolist()

        local_peaktimes = numpy.unique(local_peaktimes)

        g_offset = t_offset + padding[0]

        if ignore_dead_times:
            dead_indices = numpy.searchsorted(all_dead_times, [t_offset, t_offset + chunk_size])
            if dead_indices[0] != dead_indices[1]:
                is_included = numpy.in1d(local_peaktimes + g_offset, all_dead_times[dead_indices[0]:dead_indices[1]])
                local_peaktimes = local_peaktimes[~is_included]
                local_peaktimes = numpy.sort(local_peaktimes)

        #print "Removing the useless borders..."
        local_borders   = (template_shift, len_chunk - template_shift)
        idx             = (local_peaktimes >= local_borders[0]) & (local_peaktimes < local_borders[1])
        local_peaktimes = numpy.compress(idx, local_peaktimes)

        if collect_all:
            for i in xrange(N_e):
                all_found_spikes[i] = numpy.array(all_found_spikes[i], dtype=numpy.uint32)

                if ignore_dead_times:
                    if dead_indices[0] != dead_indices[1]:
                        is_included = numpy.in1d(all_found_spikes[i] + g_offset, all_dead_times[dead_indices[0]:dead_indices[1]])
                        all_found_spikes[i] = all_found_spikes[i][~is_included]
                        all_found_spikes[i] = numpy.sort(all_found_spikes[i])

                idx                 = (all_found_spikes[i] >= local_borders[0]) & (all_found_spikes[i] < local_borders[1])
                all_found_spikes[i] = numpy.compress(idx, all_found_spikes[i])

        n_t             = len(local_peaktimes)

        if full_gpu:
        #   all_indices = cmt.CUDAMatrix(all_indices)
            tmp_gpu = cmt.CUDAMatrix(local_peaktimes.reshape((1, n_t)), copy_on_host=False)


        if n_t > 0:
            #print "Computing the b (should full_gpu by putting all chunks on GPU if possible?)..."     

            if collect_all:
                c_local_chunk = local_chunk.copy()

            local_chunk = local_chunk.T.ravel()
            sub_mat     = numpy.zeros((size_window, n_t), dtype=numpy.float32)

            if len_chunk != last_chunk_size:
                slice_indices = numpy.zeros(0, dtype=numpy.int32)
                for idx in xrange(N_e):
                    slice_indices = numpy.concatenate((slice_indices, len_chunk*idx + temp_window))
                last_chunk_size = len_chunk

            for count, idx in enumerate(local_peaktimes):
                sub_mat[:, count] = numpy.take(local_chunk, slice_indices + idx)

            #snippet_norm = numpy.sum(sub_mat**2, 0)/n_scalar
            #sub_mat /= snippet_norm

            del local_chunk

            if use_gpu: 
                sub_mat = cmt.CUDAMatrix(sub_mat, copy_on_host=False)
                b       = cmt.sparse_dot(templates, sub_mat)
            else:
                b       = templates.dot(sub_mat)

            del sub_mat

            local_restriction = (t_offset, t_offset + chunk_size)
            all_spikes   = local_peaktimes + g_offset

            # Because for GPU, slicing by columns is more efficient, we need to transpose b
            #b           = b.transpose()
            if use_gpu and not full_gpu:
                b = b.asarray()           

            failure     = numpy.zeros(n_t, dtype=numpy.int32)

            if full_gpu:
                mask     = numpy.zeros((2*n_tm, n_t), dtype=numpy.float32)
                mask[:n_tm, :] = 1
                data     = cmt.empty(mask.shape)
                patch_gpu= b.shape[1] == 1
            else:
                mask     = numpy.ones((n_tm, n_t), dtype=numpy.bool)
                sub_b    = b[:n_tm, :]

            if collect_all:
                c_all_times = numpy.zeros((len_chunk, N_e), dtype=numpy.bool)
                c_min_times = numpy.maximum(numpy.arange(len_chunk) - template_shift, 0)
                c_max_times = numpy.minimum(numpy.arange(len_chunk) + template_shift + 1, len_chunk)
                for i in xrange(N_e):
                    c_all_times[all_found_spikes[i], i] = True
                    
            while (numpy.mean(failure) < nb_chances):

                if full_gpu:
                    b_array = b.asarray()
                    sub_b   = b_array[:n_tm, :]

                # Is there a way to update sub_b * mask at the same time?
                data        = sub_b * mask
                best_template_index, peak_index = numpy.unravel_index(data.argmax(), data.shape)
                best_template2_index = best_template_index + n_tm

                if full_gpu:
                    best_amp  = sub_b[best_template_index, peak_index]/n_scalar
                    best_amp2 = b_array[best_template_index, peak_index]/n_scalar
                else:
                    best_amp  = sub_b[best_template_index, peak_index]/n_scalar
                    best_amp2 = b[best_template2_index, peak_index]/n_scalar

                best_amp_n   = best_amp/norm_templates[best_template_index]
                best_amp2_n  = best_amp2/norm_templates[best_template2_index]

                # Verify amplitude constraint.
                a_min = amp_limits[best_template_index, 0]
                a_max = amp_limits[best_template_index, 1]
                
                if (a_min <= best_amp_n) & (best_amp_n <= a_max):
                    # Keep the matching.
                    peak_time_step = local_peaktimes[peak_index]
                     
                    mydata       = (local_peaktimes - peak_time_step).astype(numpy.int32)
                    is_neighbor  = np.where(np.abs(mydata) <= temp_2_shift)[0]
                    idx_neighbor = mydata[is_neighbor] + temp_2_shift
                    nb_neighbors = len(is_neighbor)
                    indices      = np.zeros((S_over, nb_neighbors), dtype=np.int32)
                    indices[idx_neighbor, np.arange(nb_neighbors)] = 1
                        
                    if full_gpu: 
                        indices  = cmt.CUDAMatrix(indices, copy_on_host=False)
                        if patch_gpu:
                            b_lines  = b.get_col_slice(0, b.shape[0])
                        else:
                            b_lines  = b.get_col_slice(is_neighbor[0], is_neighbor[-1]+1)
 
                        tmp1 = cmt.sparse_dot(c_overs[best_template_index], indices, mult=-best_amp)
                        tmp2 = cmt.sparse_dot(c_overs[best_template2_index], indices, mult=-best_amp2)
                        b_lines.add(tmp1.add(tmp2))
                        del tmp1, tmp2
                    else:
                        tmp1 = c_overs[best_template_index].multiply(-best_amp)
                        tmp2 = c_overs[best_template2_index].multiply(-best_amp2)
                        b[:, is_neighbor] += (tmp1 + tmp2).dot(indices)

                    # Add matching to the result.
                    t_spike               = all_spikes[peak_index]

                    if (t_spike >= local_restriction[0]) and (t_spike < local_restriction[1]):
                        result['spiketimes'] += [t_spike]
                        result['amplitudes'] += [(best_amp_n, best_amp2_n)]
                        result['templates']  += [best_template_index]
                    # Mark current matching as tried.
                    mask[best_template_index, peak_index] = False
                else:
                    # Reject the matching.
                    # Update failure counter of the peak.
                    failure[peak_index] += 1
                    # If the maximal number of failures is reached then mark peak as solved (i.e. not fitted).
                    if failure[peak_index] == nb_chances:
                        mask[:, peak_index] = False
                    else:
                        mask[best_template_index, peak_index] = False

            spikes_to_write     = numpy.array(result['spiketimes'], dtype=numpy.uint32)
            amplitudes_to_write = numpy.array(result['amplitudes'], dtype=numpy.float32)
            templates_to_write  = numpy.array(result['templates'], dtype=numpy.uint32)

            spiketimes_file.write(spikes_to_write.tostring())
            amplitudes_file.write(amplitudes_to_write.tostring())
            templates_file.write(templates_to_write.tostring())

            if collect_all:

                for temp, spike in zip(templates_to_write, spikes_to_write - g_offset):
                    c_all_times[c_min_times[spike]:c_max_times[spike], neighbors[temp]] = False

                gspikes       = numpy.where(numpy.sum(c_all_times, 1) > 0)[0]
                c_all_times   = numpy.take(c_all_times, gspikes, axis=0)
                c_local_chunk = numpy.take(c_local_chunk, gspikes, axis=0) * c_all_times                

                if sign_peaks == 'negative':
                    bestlecs = numpy.argmin(c_local_chunk, 1)
                    if matched_filter:
                        threshs = -matched_tresholds_neg[bestlecs]
                    else:
                        threshs = -thresholds[bestlecs]
                    idx      = numpy.where(numpy.min(c_local_chunk, 1) < threshs)[0]
                elif sign_peaks == 'positive':
                    bestlecs = numpy.argmax(c_local_chunk, 1)
                    if matched_filter:
                        threshs = matched_tresholds_pos[bestlecs]
                    else:
                        threshs = thresholds[bestlecs]
                    idx      = numpy.where(numpy.max(c_local_chunk, 1) > threshs)[0]
                elif sign_peaks == 'both':
                    c_local_chunk = numpy.abs(c_local_chunk)
                    bestlecs = numpy.argmax(c_local_chunk, 1)
                    if matched_filter:
                        threshs = numpy.minimum(matched_tresholds_neg[bestlecs], matched_tresholds_pos[bestlecs])
                    else:
                        threshs = thresholds[bestlecs]
                    idx      = numpy.where(numpy.max(c_local_chunk, 1) > threshs)[0]
                
                gspikes  = numpy.take(gspikes, idx)
                bestlecs = numpy.take(bestlecs, idx)
                gspikes_to_write     = numpy.array(gspikes + g_offset, dtype=numpy.uint32)
                gtemplates_to_write  = numpy.array(bestlecs, dtype=numpy.uint32)

                garbage_times_file.write(gspikes_to_write.tostring())
                garbage_temp_file.write(gtemplates_to_write.tostring())
            

            if full_gpu:
                del gpu_mask, b, data

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


    comm.Barrier()
    
    if comm.rank == 0:
        io.collect_data(comm.size, params, erase=True)

    data_file.close()