from .shared.utils import *
import circus.shared.algorithms as algo
from .shared import plot
from circus.shared.probes import get_nodes_and_edges
import h5py
from circus.shared.messages import print_and_log, init_logging

def main(params, nb_cpu, nb_gpu, use_gpu):
    # Part 1: Whitening
    numpy.random.seed(420)

    logger         = init_logging(params.logfile)
    logger         = logging.getLogger('circus.whitening')
    #################################################################
    data_file      = params.data_file
    data_file.open()
    N_e            = params.getint('data', 'N_e')
    N_total        = params.nb_channels
    N_t            = params.getint('detection', 'N_t')
    dist_peaks     = params.getint('detection', 'dist_peaks')
    template_shift = params.getint('detection', 'template_shift')
    file_out_suff  = params.get('data', 'file_out_suff')
    file_out       = params.get('data', 'file_out')
    spike_thresh   = params.getfloat('detection', 'spike_thresh')
    matched_filter = params.getboolean('detection', 'matched-filter')
    matched_thresh = params.getfloat('detection', 'matched_thresh')
    sign_peaks     = params.get('detection', 'peaks')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    chunk_size       = params.getint('whitening', 'chunk_size')
    plot_path        = os.path.join(params.get('data', 'data_file_noext'), 'plots')
    nodes, edges     = get_nodes_and_edges(params)
    safety_time      = params.getint('whitening', 'safety_time')
    nb_temp_white    = min(max(20, comm.size), N_e)
    max_silence_1    = int(20*params.rate // comm.size)
    max_silence_2    = 5000
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)
    #################################################################

    if comm.rank == 0:
        print_and_log(["Analyzing data to get whitening matrices and thresholds..."], 'default', logger)

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

    nb_chunks, last_chunk_len = data_file.analyze(chunk_size)

    if nb_chunks < comm.size:

        res        = io.data_stats(params, show=False)
        chunk_size = int(res*params.rate//comm.size)
        if comm.rank == 0:
            print_and_log(["Too much cores, automatically resizing the data chunks"], 'debug', logger)

        nb_chunks, last_chunk_len = data_file.analyze(chunk_size)


    # I guess this is more relevant, to take signals from all over the recordings
    all_chunks     = numpy.random.permutation(numpy.arange(nb_chunks, dtype=numpy.int32))
    all_electrodes = numpy.random.permutation(N_e)

    for gidx in [all_chunks[comm.rank]]:

        #print "Node", comm.rank, "is analyzing chunk", gidx,  "/", nb_chunks, " ..."
        local_chunk, t_offset = data_file.get_data(gidx, chunk_size, nodes=nodes)
        local_shape = len(local_chunk)

        #print "Node", comm.rank, "computes the median absolute deviations in a random chunk"
        thresholds = numpy.zeros(N_e, dtype=numpy.float32)
        for i in xrange(N_e):
            u             = numpy.median(local_chunk[:, i], 0)
            thresholds[i] = numpy.median(numpy.abs(local_chunk[:, i] - u), 0)
        gdata      = gather_array(thresholds, comm)
        if comm.rank == 0:
            gdata      = gdata.reshape((comm.size, N_e))
            thresholds = numpy.mean(gdata, 0)
            bfile      = h5py.File(file_out_suff + '.basis.hdf5', 'w', libver='latest')
            io.write_datasets(bfile, ['thresholds'], {'thresholds' : thresholds})
            bfile.close()
        comm.Barrier()
        thresholds  = io.load_data(params, 'thresholds')

        #print "Extracting the peaks..."
        local_peaktimes = numpy.zeros(0, dtype=numpy.int32)
        for i in xrange(N_e):
            peaktimes       = algo.detect_peaks(numpy.abs(local_chunk[:, i]), thresholds[i], valley=False, mpd=dist_peaks)
            local_peaktimes = numpy.concatenate((local_peaktimes, peaktimes))

        local_peaktimes = numpy.unique(local_peaktimes)

        #print "Removing the useless borders..."
        local_borders   = (template_shift, local_shape - template_shift)
        idx             = (local_peaktimes >= local_borders[0]) & (local_peaktimes < local_borders[1])
        local_peaktimes = numpy.compress(idx, local_peaktimes)

        if len(local_peaktimes) > 0:

            diff_times      = local_peaktimes[-1]-local_peaktimes[0]
            all_times       = numpy.zeros((N_e, diff_times+1), dtype=numpy.bool)
            min_times       = numpy.maximum(local_peaktimes - local_peaktimes[0] - safety_time, 0)
            max_times       = numpy.minimum(local_peaktimes - local_peaktimes[0] + safety_time + 1, diff_times)
            argmax_peak     = numpy.random.permutation(numpy.arange(len(local_peaktimes)))
            all_idx         = numpy.take(local_peaktimes, argmax_peak)

            #print "Selection of the peaks with spatio-temporal masks..."
            for idx, peak in zip(argmax_peak, all_idx):
                elec    = numpy.argmax(numpy.abs(local_chunk[peak]))
                indices = numpy.take(inv_nodes, edges[nodes[elec]])
                all_times[indices, min_times[idx]:max_times[idx]] = True
        else:
            all_times   = numpy.zeros((N_e, len(local_chunk)), dtype=numpy.bool)

    all_times_Ne   = numpy.any(all_times, 0)
    subset         = numpy.where(all_times_Ne == False)[0]
    all_silences   = []

    if do_spatial_whitening:
        local_silences = numpy.take(local_chunk, subset, axis=0)[:max_silence_1]
        all_silences   = gather_array(local_silences, comm, 0, 1)

    local_res      = []

    if do_temporal_whitening:

        for elec in all_electrodes[numpy.arange(comm.rank, nb_temp_white, comm.size)]:
            res            = numpy.zeros((0, N_t), dtype=numpy.float32)
            scount         = 0
            indices        = numpy.take(inv_nodes, edges[nodes[elec]])
            all_times_elec = numpy.any(numpy.take(all_times, indices, axis=0), 0)
            esubset        = numpy.where(all_times_elec == False)[0]
            bound          = len(esubset) - N_t
            while (scount < bound) and (len(res) < max_silence_2):
                myslice = esubset[scount:scount + N_t]
                if numpy.all((myslice - esubset[scount]) == numpy.arange(N_t)):
                    scount += N_t
                    res     = numpy.vstack((res, local_chunk[myslice, elec]))
                else:
                    scount += 1
            if len(res) > 5:
                local_res += [numpy.cov(res.T)]

        nb_elecs  = numpy.array([len(local_res)], dtype=numpy.float32)
        local_res = numpy.array(local_res, dtype=numpy.float32)
        if len(local_res) == 0:
            local_res = numpy.zeros(0, dtype=numpy.float32)
        else:
            local_res = numpy.sum(local_res, 0)
        all_res   = gather_array(local_res.ravel(), comm, 0, 1)
        all_elecs = gather_array(nb_elecs, comm, 0, 1)

    if comm.rank == 0:

        to_write = {}

        if do_temporal_whitening:
            try:
                nb_silences     = numpy.sum(all_elecs > 0)
                all_res         = all_res.reshape((nb_silences, N_t**2))
            except Exception:
                print_and_log(["No silent periods detected: something wrong with the parameters?"], 'error', logger)
            all_res             = numpy.sum(all_res, 0)
            all_res             = all_res.reshape((N_t, N_t))/numpy.sum(all_elecs)
            temporal_whitening  = get_whitening_matrix(all_res.astype(numpy.double), fudge=1e-3)[template_shift].astype(numpy.float32)
            temporal_whitening /= temporal_whitening.sum()
            to_write['temporal'] = temporal_whitening
            have_nans = numpy.sum(numpy.isnan(temporal_whitening))

            if have_nans > 0:
                temporal_whitening = numpy.zeros(N_t, dtype=numpy.float32)
                temporal_whitening[N_t//2] = 1
                to_write['temporal']       = temporal_whitening
                print_and_log(["Disabling temporal whitening because of NaNs found"], 'info', logger)

        if do_spatial_whitening:
            if len(all_silences)/params.rate == 0:
                print_and_log(["No silent periods detected: something wrong with the parameters?"], 'error', logger)
            spatial_whitening = get_whitening_matrix(all_silences.astype(numpy.double)).astype(numpy.float32)
            to_write['spatial'] = spatial_whitening
            print_and_log(["Found %gs without spikes for whitening matrices..." %(len(all_silences)/params.rate)], 'default', logger)

            have_nans = numpy.sum(numpy.isnan(spatial_whitening))

            if have_nans > 0:
                spatial_whitening = numpy.eye(spatial_whitening.shape[0], dtype=numpy.float32)
                to_write['spatial'] = spatial_whitening
                print_and_log(["Disabling spatial whitening because of NaNs found"], 'info', logger)

        bfile = h5py.File(file_out_suff + '.basis.hdf5', 'r+', libver='latest')
        io.write_datasets(bfile, to_write.keys(), to_write)
        bfile.close()

    del all_silences
    comm.Barrier()

    if do_spatial_whitening or do_temporal_whitening:

        if comm.rank == 0:
            print_and_log(["Because of whitening, need to recompute the thresholds..."], 'default', logger)

        if do_spatial_whitening:
            spatial_whitening  = io.load_data(params, 'spatial_whitening')
            if use_gpu:
                spatial_whitening = cmt.CUDAMatrix(spatial_whitening, copy_on_host=False)
        if do_temporal_whitening:
            temporal_whitening = io.load_data(params, 'temporal_whitening')

        for gidx in [all_chunks[comm.rank]]:
            local_chunk, t_offset = data_file.get_data(gidx, chunk_size, nodes=nodes)
            local_shape = len(local_chunk)

            if do_spatial_whitening:
                if use_gpu:
                    local_chunk = cmt.CUDAMatrix(local_chunk, copy_on_host=False)
                    local_chunk = local_chunk.dot(spatial_whitening).asarray()
                else:
                    local_chunk = numpy.dot(local_chunk, spatial_whitening)
            if do_temporal_whitening:
                local_chunk = scipy.ndimage.filters.convolve1d(local_chunk, temporal_whitening, axis=0, mode='constant')

            thresholds = numpy.zeros(N_e, dtype=numpy.float32)
            for i in xrange(N_e):
                u             = numpy.median(local_chunk[:, i], 0)
                thresholds[i] = numpy.median(numpy.abs(local_chunk[:, i] - u), 0)
            gdata      = gather_array(thresholds, comm)
            if comm.rank == 0:
                gdata      = gdata.reshape((comm.size, N_e))
                thresholds = numpy.mean(gdata, 0)
                bfile      = h5py.File(file_out_suff + '.basis.hdf5', 'r+', libver='latest')
                bfile.pop('thresholds')
                io.write_datasets(bfile, ['thresholds'], {'thresholds' : thresholds})
                bfile.close()
            comm.Barrier()

    #if comm.rank == 0:
        #if not os.path.exists(plot_path):
        #    os.makedirs(plot_path)
        #N_elec = min(int(numpy.sqrt(data_file.N_e)), 5)
        #plot.view_fit(filename, t_start=0, t_stop=1, fit_on=False, square=True,
        #              n_elec=N_elec, save=[plot_path, 'electrodes'])

    # Part 2: Basis
    numpy.random.seed(422)

    #################################################################
    file_out       = params.get('data', 'file_out')
    alignment      = params.getboolean('detection', 'alignment')
    spike_thresh   = params.getfloat('detection', 'spike_thresh')
    nodes, edges   = get_nodes_and_edges(params)
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    chunk_size       = params.getint('data', 'chunk_size')
    safety_time      = params.getint('whitening', 'safety_time')
    max_elts_elec    = params.getint('whitening', 'max_elts')
    output_dim       = params.getfloat('whitening', 'output_dim')
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)
    if sign_peaks == 'both':
       max_elts_elec *= 2
    nb_elts          = int(params.getfloat('whitening', 'nb_elts')*N_e*max_elts_elec)
    #################################################################


    if comm.rank == 0:
        print_and_log(["Searching spikes to construct the PCA basis..."], 'default', logger)

    nb_chunks, last_chunk_len = data_file.analyze(chunk_size)

    if nb_chunks < comm.size:

        res        = io.data_stats(params, show=False)
        chunk_size = int(res*params.rate//comm.size)
        if comm.rank == 0:
            print_and_log(["Too much cores, automatically resizing the data chunks"], 'debug', logger)

        nb_chunks, last_chunk_len = data_file.analyze(chunk_size)

    groups    = {}
    for i in xrange(N_e):
        groups[i] = 0

    # I guess this is more relevant, to take signals from all over the recordings
    all_chunks     = numpy.random.permutation(numpy.arange(nb_chunks, dtype=numpy.int32))
    max_elts_elec //= comm.size
    nb_elts       //= comm.size

    elt_count_pos = 0
    elt_count_neg = 0

    if sign_peaks in ['positive', 'both']:
        elts_pos      = numpy.zeros((N_t, nb_elts), dtype=numpy.float32)
    if sign_peaks in ['negative', 'both']:
        elts_neg      = numpy.zeros((N_t, nb_elts), dtype=numpy.float32)

    chunks_to_load = all_chunks[comm.rank::comm.size]

    thresholds = io.load_data(params, 'thresholds')

    if alignment:
        cdata = numpy.linspace(-template_shift, template_shift, 5*N_t)
        xdata = numpy.arange(-2*template_shift, 2*template_shift+1)

    to_explore = xrange(comm.rank, nb_chunks, comm.size)

    if comm.rank == 0:
        to_explore = get_tqdm_progressbar(to_explore)

    for gcount, gidx in enumerate(to_explore):

        gidx = all_chunks[gidx]

        if ((elt_count_pos + elt_count_neg) < nb_elts):
            #print "Node", comm.rank, "is analyzing chunk", gidx, "/", nb_chunks, " ..."
            local_chunk, t_offset = data_file.get_data(gidx, chunk_size, nodes=nodes)
            local_shape = len(local_chunk)

            if do_spatial_whitening:
                if use_gpu:
                    local_chunk = cmt.CUDAMatrix(local_chunk, copy_on_host=False)
                    local_chunk = local_chunk.dot(spatial_whitening).asarray()
                else:
                    local_chunk = numpy.dot(local_chunk, spatial_whitening)
            if do_temporal_whitening:
                local_chunk = scipy.ndimage.filters.convolve1d(local_chunk, temporal_whitening, axis=0, mode='constant')

            #print "Extracting the peaks..."
            all_peaktimes = numpy.zeros(0, dtype=numpy.int32)
            all_extremas  = numpy.zeros(0, dtype=numpy.int32)

            for i in xrange(N_e):

                if sign_peaks == 'negative':
                    peaktimes = algo.detect_peaks(local_chunk[:, i], thresholds[i], valley=True, mpd=dist_peaks)
                elif sign_peaks == 'positive':
                    peaktimes = algo.detect_peaks(local_chunk[:, i], thresholds[i], valley=False, mpd=dist_peaks)
                elif sign_peaks == 'both':
                    peaktimes = algo.detect_peaks(numpy.abs(local_chunk[:, i]), thresholds[i], valley=False, mpd=dist_peaks)
                all_peaktimes = numpy.concatenate((all_peaktimes, peaktimes))
                all_extremas  = numpy.concatenate((all_extremas, i*numpy.ones(len(peaktimes), dtype=numpy.int32)))

            #print "Removing the useless borders..."
            if alignment:
                local_borders = (2*template_shift, local_shape - 2*template_shift)
            else:
                local_borders = (template_shift, local_shape - template_shift)
            idx             = (all_peaktimes >= local_borders[0]) & (all_peaktimes < local_borders[1])
            all_peaktimes   = numpy.compress(idx, all_peaktimes)
            all_extremas    = numpy.compress(idx, all_extremas)

            local_peaktimes = numpy.unique(all_peaktimes)

            if len(local_peaktimes) > 0:

                diff_times      = local_peaktimes[-1]-local_peaktimes[0]
                all_times       = numpy.zeros((N_e, diff_times+1), dtype=numpy.bool)
                min_times       = numpy.maximum(local_peaktimes - local_peaktimes[0] - safety_time, 0)
                max_times       = numpy.minimum(local_peaktimes - local_peaktimes[0] + safety_time + 1, diff_times)

                n_times         = len(local_peaktimes)
                argmax_peak     = numpy.random.permutation(numpy.arange(n_times))
                all_idx         = numpy.take(local_peaktimes, argmax_peak)

                #print "Selection of the peaks with spatio-temporal masks..."
                for midx, peak in zip(argmax_peak, all_idx):
                    if (elt_count_neg + elt_count_pos) == nb_elts:
                        break

                    if sign_peaks == 'negative':
                        elec = numpy.argmin(local_chunk[peak])
                        negative_peak = True
                    elif sign_peaks == 'positive':
                        elec = numpy.argmax(local_chunk[peak])
                        negative_peak = False
                    elif sign_peaks == 'both':
                        if numpy.abs(numpy.max(local_chunk[peak])) > numpy.abs(numpy.min(local_chunk[peak])):
                            elec = numpy.argmax(local_chunk[peak])
                            negative_peak = False
                        else:
                            elec = numpy.argmin(local_chunk[peak])
                            negative_peak = True

                    indices = numpy.take(inv_nodes, edges[nodes[elec]])
                    myslice = all_times[indices, min_times[midx]:max_times[midx]]
                    is_local_extrema = elec in all_extremas[all_peaktimes == peak]
                    if is_local_extrema and not myslice.any():
                        upper_bounds = max_elts_elec

                        if groups[elec] < upper_bounds:

                            if negative_peak:
                                elts_neg[:, elt_count_neg] = local_chunk[peak - template_shift:peak + template_shift + 1, elec]
                            else:
                                elts_pos[:, elt_count_pos] = local_chunk[peak - template_shift:peak + template_shift + 1, elec]
                            if alignment:
                                ydata    = local_chunk[peak-2*template_shift:peak+2*template_shift+1, elec]
                                f        = scipy.interpolate.UnivariateSpline(xdata, ydata, s=0)
                                if negative_peak:
                                    rmin = (numpy.argmin(f(cdata)) - len(cdata)/2.)/5.
                                else:
                                    rmin = (numpy.argmax(f(cdata)) - len(cdata)/2.)/5.
                                ddata    = numpy.linspace(rmin-template_shift, rmin+template_shift, N_t)

                                if negative_peak:
                                    elts_neg[:, elt_count_neg] = f(ddata).astype(numpy.float32)
                                else:
                                    elts_pos[:, elt_count_pos] = f(ddata).astype(numpy.float32)

                            if negative_peak:
                                elt_count_neg += 1
                            else:
                                elt_count_pos += 1

                        groups[elec] += 1
                        all_times[indices, min_times[midx]:max_times[midx]] = True

    print_and_log(["Node %d has collected %d waveforms" %(comm.rank, elt_count_pos + elt_count_neg)], 'debug', logger)

    if sign_peaks in ['negative', 'both']:
        gdata_neg = gather_array(elts_neg[:, :elt_count_neg].T, comm, 0, 1)
    if sign_peaks in ['positive', 'both']:
        gdata_pos = gather_array(elts_pos[:, :elt_count_pos].T, comm, 0, 1)

    if comm.rank == 0:
        #DO PCA on elts and store the basis obtained.

        nb_waveforms = 0
        if sign_peaks in ['negative', 'both']:
            nb_waveforms += gdata_neg.shape[0]
        if sign_peaks in ['positive', 'both']:
            nb_waveforms += gdata_pos.shape[0]

        print_and_log(["Found %d waveforms over %d requested" %(nb_waveforms, int(nb_elts*comm.size))], 'default', logger)
        pca = PCA(output_dim, copy=False)
        res = {}
        if sign_peaks in ['negative', 'both']:
            if len(gdata_neg) > 0:
                res_pca      = pca.fit_transform(gdata_neg.astype(numpy.double)).astype(numpy.float32)
                res['proj']  = pca.components_.T.astype(numpy.float32)
            else:
                res['proj']  = numpy.identity(int(output_dim), dtype=numpy.float32)
            res['rec']       = res['proj'].T
            res['waveform']  = numpy.median(gdata_neg, 0)
            idx              = numpy.random.permutation(numpy.arange(gdata_neg.shape[0]))[:1000]
            res['waveforms'] = gdata_neg[idx, :]
        if sign_peaks in ['positive', 'both']:
            if len(gdata_pos) > 0:
                res_pca         = pca.fit_transform(gdata_pos.astype(numpy.double)).astype(numpy.float32)
                res['proj_pos'] = pca.components_.T.astype(numpy.float32)
            else:
                res['proj_pos'] = numpy.identity(output_dim, dtype=numpy.float32)
            res['rec_pos']       = res['proj_pos'].T
            res['waveform_pos']  = numpy.median(gdata_pos, 0)
            idx                  = numpy.random.permutation(numpy.arange(gdata_pos.shape[0]))[:1000]
            res['waveforms_pos'] = gdata_pos[idx, :]

        bfile    = h5py.File(file_out_suff + '.basis.hdf5', 'r+', libver='latest')
        io.write_datasets(bfile, res.keys(), res)
        if sign_peaks == 'positive':
            print_and_log(["A basis with %s dimensions has been built" %res['proj_pos'].shape[1]], 'info', logger)
        elif sign_peaks == 'negative':
            print_and_log(["A basis with %s dimensions has been built" %res['proj'].shape[1]], 'info', logger)
        elif sign_peaks == 'both':
            print_and_log(["Two basis with %s dimensions has been built" %res['proj'].shape[1]], 'info', logger)

        bfile.close()

    comm.Barrier()

    if matched_filter:

        if comm.rank == 0:
            print_and_log(["Because of matched filters, need to recompute the thresholds..."], 'default', logger)

        if do_spatial_whitening:
            spatial_whitening  = io.load_data(params, 'spatial_whitening')
            if use_gpu:
                spatial_whitening = cmt.CUDAMatrix(spatial_whitening, copy_on_host=False)
        if do_temporal_whitening:
            temporal_whitening = io.load_data(params, 'temporal_whitening')

        if sign_peaks in ['negative', 'both']:
            waveform_neg  = io.load_data(params, 'waveform')
            waveform_neg /= (numpy.abs(numpy.sum(waveform_neg))* len(waveform_neg))
        if sign_peaks in ['positive', 'both']:
            waveform_pos  = io.load_data(params, 'waveform-pos')
            waveform_pos /= (numpy.abs(numpy.sum(waveform_pos))* len(waveform_pos))

        for gidx in [all_chunks[comm.rank]]:
            local_chunk, t_offset = data_file.get_data(gidx, chunk_size, nodes=nodes)
            local_shape = len(local_chunk)

            if do_spatial_whitening:
                if use_gpu:
                    local_chunk = cmt.CUDAMatrix(local_chunk, copy_on_host=False)
                    local_chunk = local_chunk.dot(spatial_whitening).asarray()
                else:
                    local_chunk = numpy.dot(local_chunk, spatial_whitening)
            if do_temporal_whitening:
                local_chunk = scipy.ndimage.filters.convolve1d(local_chunk, temporal_whitening, axis=0, mode='constant')

            if sign_peaks in ['negative', 'both']:
                tmp_chunk = scipy.ndimage.filters.convolve1d(local_chunk, waveform_neg, axis=0, mode='constant')
                thresholds = numpy.zeros(N_e, dtype=numpy.float32)
                for i in xrange(N_e):
                    u             = numpy.median(tmp_chunk[:, i], 0)
                    thresholds[i] = numpy.median(numpy.abs(tmp_chunk[:, i] - u), 0)
                gdata      = gather_array(thresholds, comm)
                if comm.rank == 0:
                    gdata      = gdata.reshape((comm.size, N_e))
                    thresholds = numpy.mean(gdata, 0)
                    bfile      = h5py.File(file_out + '.basis.hdf5', 'r+', libver='latest')
                    io.write_datasets(bfile, ['matched_thresholds'], {'matched_thresholds' : thresholds})
                    bfile.close()
                comm.Barrier()

            if sign_peaks in ['positive', 'both']:
                tmp_chunk = scipy.ndimage.filters.convolve1d(local_chunk, waveform_pos, axis=0, mode='constant')
                thresholds = numpy.zeros(N_e, dtype=numpy.float32)
                for i in xrange(N_e):
                    u             = numpy.median(tmp_chunk[:, i], 0)
                    thresholds[i] = numpy.median(numpy.abs(tmp_chunk[:, i] - u), 0)
                gdata      = gather_array(thresholds, comm)
                if comm.rank == 0:
                    gdata      = gdata.reshape((comm.size, N_e))
                    thresholds = numpy.mean(gdata, 0)
                    bfile      = h5py.File(file_out + '.basis.hdf5', 'r+', libver='latest')
                    io.write_datasets(bfile, ['matched_thresholds_pos'], {'matched_thresholds_pos' : thresholds})
                    bfile.close()
                comm.Barrier()

    data_file.close()
