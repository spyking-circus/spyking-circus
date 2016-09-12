import circus.shared.algorithms as algo
from .shared.utils import *

def main(filename, params, nb_cpu, nb_gpu, use_gpu):

    try:
        SHARED_MEMORY = True
        MPI.Win.Allocate_shared(1, 1, MPI.INFO_NULL, MPI.COMM_SELF).Free()
    except NotImplementedError:
        SHARED_MEMORY = False

    #################################################################
    sampling_rate  = params.getint('data', 'sampling_rate')
    N_e            = params.getint('data', 'N_e')
    N_t            = params.getint('data', 'N_t')
    N_total        = params.getint('data', 'N_total')
    template_shift = params.getint('data', 'template_shift')
    file_out       = params.get('data', 'file_out')
    file_out_suff  = params.get('data', 'file_out_suff')
    sign_peaks     = params.get('detection', 'peaks')
    matched_filter = params.getboolean('detection', 'matched-filter')
    spike_thresh   = params.getfloat('detection', 'spike_thresh')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    chunk_size     = int(params.getfloat('fitting', 'chunk')*sampling_rate)
    gpu_only       = params.getboolean('fitting', 'gpu_only')
    nodes, edges   = io.get_nodes_and_edges(params)
    tmp_limits     = params.get('fitting', 'amp_limits').replace('(', '').replace(')', '').split(',')
    tmp_limits     = map(float, tmp_limits)
    amp_auto       = params.getboolean('fitting', 'amp_auto')
    space_explo    = params.getfloat('fitting', 'space_explo')
    nb_chances     = params.getint('fitting', 'nb_chances')
    max_chunk      = params.getfloat('fitting', 'max_chunk')
    collect_all    = params.getboolean('fitting', 'collect_all')
    if collect_all:
        collect_zone = int(0.5*sampling_rate*1e-3)
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)
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
        templates  = io.load_data_memshared(params, comm, 'templates', normalize=True, transpose=True)
        N_tm, x    = templates.shape
    else:
        templates  = io.load_data(params, 'templates')
        x, N_tm    = templates.shape

    N_e            = params.getint('data', 'N_e')
    N_t            = params.getint('data', 'N_t')
    template_shift = int((N_t-1)//2)
    temp_2_shift   = 2*template_shift
    full_gpu       = use_gpu and gpu_only
    n_tm           = N_tm//2
    n_scalar       = N_e*N_t
    last_spikes    = numpy.zeros((n_tm, 1), dtype=numpy.int32)
    temp_window    = numpy.arange(-template_shift, template_shift+1)

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
            waveform_neg  = io.load_data(params, 'waveform')
            waveform_neg /= (numpy.abs(numpy.sum(waveform_neg))* len(waveform_neg))
            matched_tresholds_neg = io.load_data(params, 'matched-thresholds')
        if sign_peaks in ['positive', 'both']:
            waveform_pos  = io.load_data(params, 'waveform-pos')
            waveform_pos /= (numpy.abs(numpy.sum(waveform_pos))* len(waveform_pos))
            matched_tresholds_pos = io.load_data(params, 'matched-thresholds-pos')

    thresholds = io.load_data(params, 'thresholds')


    if collect_all:
        neigbors = {}
        for i in xrange(n_tm):
            tmp  = templates[i, :].toarray().reshape(N_e, N_t) * norm_templates[i]
            if sign_peaks == 'negative':
                if matched_filter:
                    threshs = -matched_tresholds_neg
                else:
                    threshs = -thresholds
                idx      = numpy.where(numpy.min(tmp, 1) <= amp_limits[i, 0]*threshs)[0]
            elif sign_peaks == 'positive':
                if matched_filter:
                    threshs = matched_tresholds_pos
                else:
                    threshs = thresholds
                idx      = numpy.where(numpy.max(tmp, 1) >= amp_limits[i, 0]*threshs)[0]
            elif sign_peaks == 'both':
                if matched_filter:
                    threshs = numpy.minimum(matched_tresholds_neg, matched_tresholds_pos)
                else:
                    threshs = thresholds
                idx      = numpy.where(numpy.max(numpy.abs(tmp), 1) >= amp_limits[i, 0]*threshs)[0]
            neigbors[i] = idx

    if use_gpu:
        templates = cmt.SparseCUDAMatrix(templates, copy_on_host=False)

    info_string   = ''

    
    if comm.rank == 0:
        if use_gpu:
            info_string = "using %d GPUs" %(comm.size)
        else:
            info_string = "using %d CPUs" %(comm.size)

    comm.Barrier()

    
    if SHARED_MEMORY:
        c_overs    = io.load_data_memshared(params, comm, 'overlaps', nb_cpu=nb_cpu, nb_gpu=nb_gpu, use_gpu=use_gpu)        
        c_overlap  = io.get_overlaps(comm, params, nb_cpu=nb_cpu, nb_gpu=nb_gpu, use_gpu=use_gpu)
        over_shape = c_overlap.get('over_shape')[:]
        N_over     = int(numpy.sqrt(over_shape[0]))
        S_over     = over_shape[1]
    else:
        c_overlap  = io.get_overlaps(comm, params, nb_cpu=nb_cpu, nb_gpu=nb_gpu, use_gpu=use_gpu)
        over_x     = c_overlap.get('over_x')[:]
        over_y     = c_overlap.get('over_y')[:]
        over_data  = c_overlap.get('over_data')[:]
        over_shape = c_overlap.get('over_shape')[:]
        N_over     = int(numpy.sqrt(over_shape[0]))
        S_over     = over_shape[1]
        c_overlap.close()

        # To be faster, we rearrange the overlaps into a dictionnary. This has a cost: twice the memory usage for 
        # a short period of time
        c_overs   = {}
        overlaps  = scipy.sparse.csr_matrix((over_data, (over_x, over_y)), shape=(over_shape[0], over_shape[1]))
        del over_x, over_y, over_data
        
        for i in xrange(N_over):
            c_overs[i] = overlaps[i*N_over:(i+1)*N_over]
        
        del overlaps

    comm.Barrier()

    if comm.rank == 0:
        io.print_and_log(["Here comes the SpyKING CIRCUS %s and %d templates..." %(info_string, n_tm)], 'default', params)
        io.purge(file_out_suff, '.data')

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
                io.print_and_log(["Not enough memory on GPUs: GPUs are used for projection only"], 'info', params)
            for i in xrange(N_over):
                if c_overs.has_key(i):
                    del c_overs[i]
            full_gpu = False

    borders, nb_chunks, chunk_len, last_chunk_len = io.analyze_data(params, chunk_size)
    nb_chunks                                     = int(min(nb_chunks, max_chunk))

    if comm.rank == 0:
        pbar = get_progressbar(int(nb_chunks//comm.size))


    spiketimes_file = open(file_out_suff + '.spiketimes-%d.data' %comm.rank, 'wb')
    amplitudes_file = open(file_out_suff + '.amplitudes-%d.data' %comm.rank, 'wb')
    templates_file  = open(file_out_suff + '.templates-%d.data' %comm.rank, 'wb')

    if collect_all:
        garbage_times_file = open(file_out_suff + '.gspiketimes-%d.data' %comm.rank, 'wb')
        garbage_temp_file  = open(file_out_suff + '.gtemplates-%d.data' %comm.rank, 'wb')


    comm.Barrier()

    if use_gpu and do_spatial_whitening:
        spatial_whitening = cmt.CUDAMatrix(spatial_whitening, copy_on_host=False)


    last_chunk_size = 0

    for gcount, gidx in enumerate(xrange(comm.rank, nb_chunks, comm.size)):
        #print "Node", comm.rank, "is analyzing chunk", gidx, "/", nb_chunks, " ..."
        ## We need to deal with the borders by taking chunks of size [0, chunck_size+template_shift]
        if gidx == (nb_chunks - 1):
            padding = (-2*borders, 0)
        elif gidx == 0:
            padding = (0, 2*borders)
        else:
            padding = (-2*borders, 2*borders)

        result       = {'spiketimes' : [], 'amplitudes' : [], 'templates' : []}

        local_chunk, local_shape = io.load_chunk(params, gidx, chunk_len, chunk_size, padding, nodes=nodes)           

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

        local_peaktimes = numpy.zeros(0, dtype=numpy.int32)

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
                    peaktimes = algo.detect_peaks(local_chunk[:, i], thresholds[i], valley=True)
                elif sign_peaks == 'positive':
                    peaktimes = algo.detect_peaks(local_chunk[:, i], thresholds[i], valley=False)
                elif sign_peaks == 'both':
                    peaktimes = algo.detect_peaks(numpy.abs(local_chunk[:, i]), thresholds[i], valley=False)                    
                local_peaktimes = numpy.concatenate((local_peaktimes, peaktimes)) 
                if collect_all:
                    all_found_spikes[i] += peaktimes.tolist()


            
        local_peaktimes = numpy.unique(local_peaktimes)
        
        #print "Removing the useless borders..."
        local_borders   = (template_shift, local_shape - template_shift)
        idx             = (local_peaktimes >= local_borders[0]) & (local_peaktimes < local_borders[1])
        local_peaktimes = numpy.compress(idx, local_peaktimes)

        if collect_all:
            for i in xrange(N_e):
                all_found_spikes[i] = numpy.array(all_found_spikes[i], dtype=numpy.int32)
                idx                 = (all_found_spikes[i] >= local_borders[0]) & (all_found_spikes[i] < local_borders[1])
                all_found_spikes[i] = numpy.compress(idx, all_found_spikes[i])

        n_t             = len(local_peaktimes)
        len_chunk       = local_chunk.shape[0]
        all_indices     = numpy.arange(n_t)
                            

        if full_gpu:
        #   all_indices = cmt.CUDAMatrix(all_indices)
            tmp_gpu = cmt.CUDAMatrix(local_peaktimes.reshape((1, n_t)), copy_on_host=False)


        if n_t > 0:
            #print "Computing the b (should full_gpu by putting all chunks on GPU if possible?)..."     

            if collect_all:
                c_local_chunk = local_chunk.copy()

            local_chunk = local_chunk.T.ravel()
            sub_mat     = numpy.zeros((N_e*(2*template_shift+1), n_t), dtype=numpy.float32)

            if len_chunk != last_chunk_size:
                slice_indices = numpy.zeros(0, dtype=numpy.int32)
                for idx in xrange(N_e):
                    slice_indices = numpy.concatenate((slice_indices, len_chunk*idx + temp_window))
                last_chunk_size = len_chunk

            for count, idx in enumerate(local_peaktimes):
                sub_mat[:, count] = numpy.take(local_chunk, slice_indices + idx)

            del local_chunk

            if use_gpu: 
                sub_mat = cmt.CUDAMatrix(sub_mat, copy_on_host=False)
                b       = cmt.sparse_dot(templates, sub_mat)
            else:
                b       = templates.dot(sub_mat)                

            del sub_mat

            local_offset = gidx*chunk_size+padding[0]//N_total
            local_bounds = (temp_2_shift, local_shape - temp_2_shift)
            all_spikes   = local_peaktimes + local_offset

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
                mask     = numpy.ones((n_tm, n_t), dtype=numpy.float32)
                sub_b    = b[:n_tm, :]

            min_time     = local_peaktimes.min()
            max_time     = local_peaktimes.max()
            local_len    = max_time - min_time + 1
            min_times    = numpy.maximum(local_peaktimes - min_time - temp_2_shift, 0)
            max_times    = numpy.minimum(local_peaktimes - min_time + temp_2_shift + 1, max_time - min_time)
            max_n_t      = int(space_explo*(max_time-min_time+1)//(2*temp_2_shift + 1))

            if collect_all:
                c_all_times = numpy.zeros((N_e, local_len), dtype=numpy.bool)
                c_min_times = numpy.maximum(numpy.arange(0, local_len) - collect_zone, 0)
                c_max_times = numpy.minimum(numpy.arange(0, local_len) + collect_zone + 1, max_time - min_time)
                for i in xrange(N_e):
                    c_all_times[i, all_found_spikes[i] - min_time] = True
                    
            while (numpy.mean(failure) < nb_chances):

                if full_gpu:
                    gpu_mask    = cmt.CUDAMatrix(mask, copy_on_host=False)
                    b.mult(gpu_mask, data)
                    tmp_mat     = data.max(0)
                    argmax_bi   = numpy.argsort(tmp_mat.asarray()[0, :])[::-1]
                    del tmp_mat
                else:
                    data        = sub_b * mask
                    argmax_bi   = numpy.argsort(numpy.max(data, 0))[::-1]

                while (len(argmax_bi) > 0):

                    subset          = []
                    indices         = []
                    all_times       = numpy.zeros(local_len, dtype=numpy.bool)

                    for count, idx in enumerate(argmax_bi):
                        myslice = all_times[min_times[idx]:max_times[idx]]
                        if not myslice.any():
                            subset  += [idx]
                            indices += [count]
                            all_times[min_times[idx]:max_times[idx]] = True
                        if len(subset) > max_n_t:
                            break

                    subset    = numpy.array(subset, dtype=numpy.int32)
                    argmax_bi = numpy.delete(argmax_bi, indices)

                    if full_gpu:
                        b_array = b.asarray()
                        sub_b   = b_array[:n_tm, :]

                    inds_t, inds_temp = subset, numpy.argmax(numpy.take(sub_b, subset, axis=1), 0)

                    if full_gpu:
                        best_amp  = sub_b[inds_temp, inds_t]/n_scalar
                        best_amp2 = b_array[inds_temp + n_tm, inds_t]/n_scalar
                    else:
                        
                        best_amp  = sub_b[inds_temp, inds_t]/n_scalar
                        best_amp2 = b[inds_temp + n_tm, inds_t]/n_scalar

                    mask[inds_temp, inds_t] = 0

                    best_amp_n   = best_amp/numpy.take(norm_templates, inds_temp)
                    best_amp2_n  = best_amp2/numpy.take(norm_templates, inds_temp + n_tm)

                    all_idx      = ((best_amp_n >= amp_limits[inds_temp, 0]) & (best_amp_n <= amp_limits[inds_temp, 1]))
                    to_keep      = numpy.where(all_idx == True)[0]
                    to_reject    = numpy.where(all_idx == False)[0]
                    ts           = numpy.take(local_peaktimes, inds_t[to_keep])
                    good         = (ts >= local_bounds[0]) & (ts < local_bounds[1])

                    # We reduce to only the good times that will be kept
                    #to_keep      = to_keep[good]
                    #ts           = ts[good]
                    
                    if len(ts) > 0:
                        if full_gpu:
                            tmp  = cmt.CUDAMatrix(numpy.ones((len(ts), 1)), copy_on_host=False)
                            tmp3 = cmt.CUDAMatrix(-ts.reshape((len(ts), 1)), copy_on_host=False)
                            tmp  = tmp.dot(tmp_gpu)
                            tmp.add_col_vec(tmp3)
                            condition = cmt.empty(tmp.shape)
                            cmt.abs(tmp, condition).less_than(temp_2_shift + 1)
                            condition = condition.asarray().astype(numpy.bool)
                            tmp       = tmp.asarray().astype(numpy.int32)
                        else:
                            tmp      = numpy.dot(numpy.ones((len(ts), 1), dtype=numpy.int32), local_peaktimes.reshape((1, n_t)))
                            tmp     -= ts.reshape((len(ts), 1))
                            condition = numpy.abs(tmp) <= temp_2_shift

                        for count, keep in enumerate(to_keep):
                            
                            idx_b    = numpy.compress(condition[count, :], all_indices)
                            ytmp     = tmp[count, condition[count, :]] + temp_2_shift
                            
                            indices  = numpy.zeros((S_over, len(ytmp)), dtype=numpy.float32)
                            indices[ytmp, numpy.arange(len(ytmp))] = 1

                            if full_gpu: 
                                indices  = cmt.CUDAMatrix(indices, copy_on_host=False)
                                if patch_gpu:
                                    b_lines  = b.get_col_slice(0, b.shape[0])
                                else:
                                    b_lines  = b.get_col_slice(idx_b[0], idx_b[-1]+1)

                                tmp1 = cmt.sparse_dot(c_overs[inds_temp[keep]], indices, mult=-best_amp[keep])
                                tmp2 = cmt.sparse_dot(c_overs[inds_temp[keep] + n_tm], indices, mult=-best_amp2[keep])
                                b_lines.add(tmp1.add(tmp2))
                                del tmp1, tmp2
                            else:
                                tmp1   = c_overs[inds_temp[keep]].multiply(-best_amp[keep]).dot(indices)
                                tmp2   = c_overs[inds_temp[keep] + n_tm].multiply(-best_amp2[keep]).dot(indices)
                                b[:, idx_b] += tmp1 + tmp2

                            if good[count]:

                                t_spike               = ts[count] + local_offset
                                result['spiketimes'] += [t_spike]
                                result['amplitudes'] += [(best_amp_n[keep], best_amp2_n[keep])]
                                result['templates']  += [inds_temp[keep]]

                    myslice           = numpy.take(inds_t, to_reject)
                    failure[myslice] += 1
                    sub_idx           = (numpy.take(failure, myslice) >= nb_chances)
                    
                    mask[:, numpy.compress(sub_idx, myslice)] = 0


            spikes_to_write     = numpy.array(result['spiketimes'], dtype=numpy.uint32)
            amplitudes_to_write = numpy.array(result['amplitudes'], dtype=numpy.float32)
            templates_to_write  = numpy.array(result['templates'], dtype=numpy.int32)

            spiketimes_file.write(spikes_to_write.tostring())
            amplitudes_file.write(amplitudes_to_write.tostring())
            templates_file.write(templates_to_write.tostring())

            if collect_all:

                for temp, spike in zip(templates_to_write, spikes_to_write - local_offset):
                    c_all_times[neigbors[temp], c_min_times[spike-min_time]:c_max_times[spike-min_time]] = False

                gspikes       = numpy.where(numpy.sum(c_all_times, 0) > 0)[0]
                c_local_chunk = numpy.take(c_local_chunk, gspikes, axis=0)

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
                gspikes_to_write     = numpy.array(gspikes + local_offset, dtype=numpy.uint32)
                gtemplates_to_write  = numpy.array(bestlecs, dtype=numpy.int32)

                garbage_times_file.write(gspikes_to_write.tostring())
                garbage_temp_file.write(gtemplates_to_write.tostring())
            

            if full_gpu:
                del gpu_mask, b, data

        if comm.rank == 0:
            pbar.update(gcount)

    spiketimes_file.close()
    amplitudes_file.close()
    templates_file.close()

    comm.Barrier()


    if comm.rank == 0:
        pbar.finish()

    if comm.rank == 0:
        io.collect_data(comm.size, params, erase=True)