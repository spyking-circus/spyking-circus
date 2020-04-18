from circus.shared.utils import *
import circus.shared.files as io
from circus.shared.probes import get_nodes_and_edges
from circus.shared.files import get_dead_times
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
from circus.shared.messages import print_and_log, init_logging
from circus.shared.mpi import detect_memory


def main(params, nb_cpu, nb_gpu, use_gpu):
    # Part 1: Whitening
    numpy.random.seed(420)
    # params = detect_memory(params)
    _ = init_logging(params.logfile)
    logger = logging.getLogger('circus.whitening')
    #################################################################
    data_file = params.data_file
    N_e = params.getint('data', 'N_e')
    hdf5_compress = params.getboolean('data', 'hdf5_compress')
    N_total = params.nb_channels
    N_t = params.getint('detection', 'N_t')
    dist_peaks = params.getint('detection', 'dist_peaks')
    template_shift = params.getint('detection', 'template_shift')
    file_out_suff = params.get('data', 'file_out_suff')
    spike_thresh = params.getfloat('detection', 'spike_thresh')
    spike_width = params.getfloat('detection', 'spike_width')
    matched_filter = params.getboolean('detection', 'matched-filter')
    matched_thresh = params.getfloat('detection', 'matched_thresh')
    fudge = params.getfloat('whitening', 'fudge')
    sign_peaks = params.get('detection', 'peaks')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening = params.getboolean('whitening', 'spatial')
    ignore_spikes = params.getboolean('whitening', 'ignore_spikes')
    chunk_size = detect_memory(params, whitening=True)
    plot_path = os.path.join(params.get('data', 'file_out_suff'), 'plots')
    nodes, edges = get_nodes_and_edges(params)
    safety_time = params.getint('whitening', 'safety_time')
    safety_space = params.getboolean('whitening', 'safety_space')
    nb_temp_white = min(max(20, comm.size), N_e)
    max_silence_1 = int(20 * params.rate // comm.size)
    max_silence_2 = 5000
    inv_nodes = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.arange(len(nodes))
    jitter_range = params.getint('detection', 'jitter_range')
    template_shift_2 = template_shift + jitter_range
    use_hanning = params.getboolean('detection', 'hanning')
    rejection_threshold = params.getfloat('detection', 'rejection_threshold')
    noise_window = params.getint('detection', 'noise_time')
    data_file.open()
    #################################################################

    if use_hanning:
        hanning_filter = numpy.hanning(N_t)

    if comm.rank == 0:
        print_and_log(["Analyzing data to get whitening matrices and thresholds..."], 'default', logger)

    nodes_indices = {}
    for elec in numpy.arange(N_e):
        nodes_indices[elec] = inv_nodes[edges[nodes[elec]]]

    if use_gpu:
        import cudamat as cmt
        # # Need to properly handle multi GPU per MPI nodes?
        if nb_gpu > nb_cpu:
            gpu_id = int(comm.rank//nb_cpu)
        else:
            gpu_id = 0
        cmt.cuda_set_device(gpu_id)
        cmt.init()
        cmt.cuda_sync_threads()

    nb_chunks, last_chunk_len = data_file.analyze(chunk_size)

    if nb_chunks < comm.size:

        res = io.data_stats(params, show=False)
        chunk_size = int(res*params.rate//comm.size)
        if comm.rank == 0:
            print_and_log(["Too much cores, automatically resizing the data chunks"], 'debug', logger)

        nb_chunks, last_chunk_len = data_file.analyze(chunk_size)

    # I guess this is more relevant, to take signals from all over the recordings.
    all_chunks = numpy.random.permutation(numpy.arange(nb_chunks, dtype=numpy.int32))
    all_electrodes = numpy.random.permutation(N_e)

    for gidx in [all_chunks[comm.rank]]:

        # print "Node", comm.rank, "is analyzing chunk", gidx,  "/", nb_chunks, " ..."
        local_chunk, t_offset = data_file.get_data(gidx, chunk_size, nodes=nodes)
        local_shape = len(local_chunk)

        # print "Node", comm.rank, "computes the median absolute deviations in a random chunk"
        thresholds = numpy.zeros(N_e, dtype=numpy.float32)
        for i in range(N_e):
            u = numpy.median(local_chunk[:, i], 0)
            thresholds[i] = numpy.median(numpy.abs(local_chunk[:, i] - u), 0)
        gdata = gather_array(thresholds, comm)
        if comm.rank == 0:
            gdata = gdata.reshape((comm.size, N_e))
            thresholds = numpy.mean(gdata, 0)
            bfile = h5py.File(file_out_suff + '.basis.hdf5', 'w', libver='earliest')
            io.write_datasets(bfile, ['thresholds'], {'thresholds': thresholds}, compression=hdf5_compress)
            bfile.close()
        comm.Barrier()
        thresholds = io.load_data(params, 'thresholds')

        local_borders = (template_shift, local_shape - template_shift)
        found_peaktimes = []

        if ignore_spikes:
            # Extracting the peaks.
            local_peaktimes = [np.empty(0, dtype=numpy.uint32)]
            for i in range(N_e):
                peaktimes = scipy.signal.find_peaks(
                    numpy.abs(local_chunk[:, i]), height=thresholds[i], width=spike_width, wlen=N_t
                )[0]
                peaktimes = peaktimes.astype(numpy.uint32)

                # print "Removing the useless borders..."
                idx = (peaktimes >= local_borders[0]) & (peaktimes < local_borders[1])
                peaktimes = numpy.compress(idx, peaktimes)

                found_peaktimes.append(peaktimes)
        else:
            for i in range(N_e):
                found_peaktimes.append(numpy.zeros(0, dtype=numpy.uint32))

        all_peaktimes = numpy.concatenate(found_peaktimes)
        local_peaktimes = numpy.unique(all_peaktimes)

        if len(local_peaktimes) > 0:

            diff_times = local_peaktimes[-1]-local_peaktimes[0]
            all_times = numpy.zeros((N_e, diff_times+1), dtype=numpy.bool)
            padded_peaks = (local_peaktimes - local_peaktimes[0]).astype(numpy.int32)
            min_times = numpy.maximum(padded_peaks - safety_time, 0)
            max_times = numpy.minimum(padded_peaks + safety_time + 1, diff_times + 1)

            test_extremas = numpy.zeros((N_e, diff_times + 1), dtype=numpy.bool)
            for i in range(N_e):
                test_extremas[i, found_peaktimes[i] - local_peaktimes[0]] = True

            argmax_peak = numpy.random.permutation(numpy.arange(len(local_peaktimes)))
            all_idx = numpy.take(local_peaktimes, argmax_peak)

            # print "Selection of the peaks with spatio-temporal masks..."
            for idx, peak in zip(argmax_peak, all_idx):

                all_elecs = numpy.where(test_extremas[:, peak - local_peaktimes[0]])[0]
                data = local_chunk[peak, all_elecs]
                elec = all_elecs[numpy.argmax(numpy.abs(data))]
                indices = nodes_indices[elec]
                if safety_space:
                    all_times[indices, min_times[idx]:max_times[idx]] = True
                else:
                    all_times[elec, min_times[idx]:max_times[idx]] = True
        else:
            all_times = numpy.zeros((N_e, len(local_chunk)), dtype=numpy.bool)

    if do_temporal_whitening:

        local_res_temp = []

        for elec in all_electrodes[numpy.arange(comm.rank, nb_temp_white, comm.size)]:
            res = numpy.zeros((0, N_t), dtype=numpy.float32)
            scount = 0
            indices = nodes_indices[elec]
            all_times_elec = numpy.any(numpy.take(all_times, indices, axis=0), 0)
            esubset = numpy.where(all_times_elec == False)[0]
            bound = len(esubset) - N_t
            while (scount < bound) and (len(res) < max_silence_2):
                myslice = esubset[scount:scount + N_t]
                if numpy.all((myslice - esubset[scount]) == numpy.arange(N_t)):
                    scount += N_t
                    res = numpy.vstack((res, local_chunk[myslice, elec]))
                else:
                    scount += 1
            if len(res) > 5:
                local_res_temp += [numpy.cov(res.T)]

        nb_elecs = numpy.array([len(local_res_temp)], dtype=numpy.float32)
        local_res_temp = numpy.array(local_res_temp, dtype=numpy.float32)
        if len(local_res_temp) == 0:
            local_res_temp = numpy.zeros(0, dtype=numpy.float32)
        else:
            local_res_temp = numpy.sum(local_res_temp, 0)
        all_res_temp = gather_array(local_res_temp.ravel(), comm, 0, 1)
        all_elecs = gather_array(nb_elecs, comm, 0, 1)

    if do_spatial_whitening:

        local_res_spac = numpy.zeros((N_e, N_e), dtype=numpy.float32)
        local_silences = []

        for elec in numpy.arange(comm.rank, N_e, comm.size):
            indices = nodes_indices[elec]
            all_times_elec = numpy.any(numpy.take(all_times, indices, axis=0), 0)
            esubset = numpy.where(all_times_elec == False)[0]
            local_data = local_chunk[esubset][:, indices]
            local_whitening = get_whitening_matrix(local_data, fudge=fudge).astype(numpy.float32)
            pos = numpy.where(elec == indices)[0]
            local_res_spac[elec, indices] = local_whitening[pos]
            local_silences += [len(esubset)]

        all_res_spac = gather_array(local_res_spac.ravel(), comm, 0, 1)
        all_silences = gather_array(numpy.array(local_silences, dtype=numpy.int32), comm, 0, 1, 'uint32')

    if comm.rank == 0:

        to_write = {}

        if do_temporal_whitening:
            try:
                nb_silences = numpy.sum(all_elecs > 0)
                all_res_temp = all_res_temp.reshape((nb_silences, N_t**2))
            except Exception:
                print_and_log(["No silent periods detected: something wrong with the parameters?"], 'error', logger)
            all_res_temp = numpy.sum(all_res_temp, 0)
            all_res_temp = all_res_temp.reshape((N_t, N_t))/numpy.sum(all_elecs)
            temporal_whitening = get_whitening_matrix(all_res_temp.astype(numpy.double), fudge=1e-3)[template_shift].astype(numpy.float32)
            temporal_whitening /= temporal_whitening.sum()
            to_write['temporal'] = temporal_whitening
            have_nans = numpy.sum(numpy.isnan(temporal_whitening))

            if have_nans > 0:
                temporal_whitening = numpy.zeros(N_t, dtype=numpy.float32)
                temporal_whitening[N_t // 2] = 1
                to_write['temporal'] = temporal_whitening
                print_and_log(["Disabling temporal whitening because of NaNs found"], 'info', logger)

        if do_spatial_whitening:
            all_res_spac = all_res_spac.reshape(comm.size, N_e, N_e)
            spatial_whitening = numpy.sum(all_res_spac, 0)
            to_write['spatial'] = spatial_whitening

            print_and_log(["Found %gs without spikes for whitening matrices..." % (numpy.mean(all_silences) / params.rate)], 'default', logger)

            have_nans = numpy.sum(numpy.isnan(spatial_whitening))

            if have_nans > 0:
                spatial_whitening = numpy.eye(spatial_whitening.shape[0], dtype=numpy.float32)
                to_write['spatial'] = spatial_whitening
                print_and_log(["Disabling spatial whitening because of NaNs found"], 'info', logger)

        bfile = h5py.File(file_out_suff + '.basis.hdf5', 'r+', libver='earliest')
        io.write_datasets(bfile, to_write.keys(), to_write, compression=hdf5_compress)
        bfile.close()

    comm.Barrier()

    if do_spatial_whitening or do_temporal_whitening:

        if comm.rank == 0:
            print_and_log(["Because of whitening, need to recompute the thresholds..."], 'default', logger)

        if do_spatial_whitening:
            spatial_whitening = io.load_data(params, 'spatial_whitening')
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
            for i in range(N_e):
                u = numpy.median(local_chunk[:, i], 0)
                thresholds[i] = numpy.median(numpy.abs(local_chunk[:, i] - u), 0)
            gdata = gather_array(thresholds, comm)
            if comm.rank == 0:
                gdata = gdata.reshape((comm.size, N_e))
                thresholds = numpy.mean(gdata, 0)
                bfile = h5py.File(file_out_suff + '.basis.hdf5', 'r+', libver='earliest')
                bfile.pop('thresholds')
                io.write_datasets(bfile, ['thresholds'], {'thresholds': thresholds}, compression=hdf5_compress)
                bfile.close()
            comm.Barrier()

    # if comm.rank == 0:
    #     if not os.path.exists(plot_path):
    #         os.makedirs(plot_path)
    #     N_elec = min(int(numpy.sqrt(data_file.N_e)), 5)
    #     plot.view_fit(filename, t_start=0, t_stop=1, fit_on=False, square=True,
    #                   n_elec=N_elec, save=[plot_path, 'electrodes'])

    # Part 2: Basis
    numpy.random.seed(422)

    #################################################################
    file_out = params.get('data', 'file_out')
    alignment = params.getboolean('detection', 'alignment')
    over_factor = params.getint('detection', 'oversampling_factor')
    nb_jitter = params.getint('detection', 'nb_jitter')
    spike_thresh = params.getfloat('detection', 'spike_thresh')
    nodes, edges = get_nodes_and_edges(params)
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening = params.getboolean('whitening', 'spatial')
    if matched_filter:
        chunk_size = detect_memory(params, whitening=True)
    else:
        chunk_size = detect_memory(params)
    safety_time = params.getint('whitening', 'safety_time')
    max_elts_elec = params.getint('whitening', 'max_elts')
    output_dim = params.getfloat('whitening', 'output_dim')
    inv_nodes = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.arange(len(nodes))
    smoothing_factor = params.getfloat('detection', 'smoothing_factor')
    if sign_peaks == 'both':
        max_elts_elec *= 2
    nb_elts = int(params.getfloat('whitening', 'nb_elts') * N_e * max_elts_elec)

    ignore_dead_times = params.getboolean('triggers', 'ignore_times')
    if ignore_dead_times:
        all_dead_times = get_dead_times(params)
    data_file.open()
    #################################################################

    if comm.rank == 0:
        print_and_log(["Searching spikes to construct the PCA basis..."], 'default', logger)

    nb_chunks, last_chunk_len = data_file.analyze(chunk_size)

    if nb_chunks < comm.size:

        res = io.data_stats(params, show=False)
        chunk_size = int(res*params.rate//comm.size)
        if comm.rank == 0:
            print_and_log(["Too much cores, automatically resizing the data chunks"], 'debug', logger)

        nb_chunks, last_chunk_len = data_file.analyze(chunk_size)

    groups = {}
    for i in range(N_e):
        groups[i] = 0

    # I guess this is more relevant, to take signals from all over the recordings
    all_chunks = numpy.random.permutation(numpy.arange(nb_chunks, dtype=numpy.int32))
    max_elts_elec //= comm.size
    nb_elts //= comm.size

    elt_count_pos = 0
    elt_count_neg = 0

    if sign_peaks in ['positive', 'both']:
        elts_pos = numpy.zeros((N_t, nb_elts), dtype=numpy.float32)
    if sign_peaks in ['negative', 'both']:
        elts_neg = numpy.zeros((N_t, nb_elts), dtype=numpy.float32)

    chunks_to_load = all_chunks[comm.rank::comm.size]

    thresholds = io.load_data(params, 'thresholds')
    mads = io.load_data(params, 'mads')
    stds = io.load_data(params, 'stds')

    if alignment:
        cdata = numpy.linspace(-jitter_range, +jitter_range, nb_jitter)
        xdata = numpy.arange(-template_shift_2, template_shift_2 + 1)
        xoff = len(cdata) / 2.0
        snippet_duration = template_shift_2
        m_size = 2 * template_shift_2 + 1
        align_factor = m_size
        local_factors = align_factor*((smoothing_factor*mads)**2)
    else:
        snippet_duration = template_shift
        xdata = numpy.arange(-template_shift, template_shift+1)

    if rejection_threshold > 0:
        reject_noise = True
        noise_levels = stds * (2 * noise_window + 1)
    else:
        reject_noise = False

    to_explore = range(comm.rank, nb_chunks, comm.size)

    upper_bounds = max_elts_elec

    if comm.rank == 0:
        to_explore = get_tqdm_progressbar(params, to_explore)

    for gcount, gidx in enumerate(to_explore):

        gidx = all_chunks[gidx]

        if (elt_count_pos + elt_count_neg) < nb_elts:
            # print "Node", comm.rank, "is analyzing chunk", gidx, "/", nb_chunks, " ..."
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

            local_borders = (snippet_duration, local_shape - snippet_duration)

            if ignore_dead_times:
                dead_indices = numpy.searchsorted(all_dead_times, [t_offset, t_offset + local_shape])

            # Extracting the peaks.
            all_peaktimes = [numpy.empty(0, dtype=numpy.uint32)]
            found_peaktimes = []
            for i in range(N_e):
                height = thresholds[i]
                if sign_peaks == 'negative':
                    peaktimes = scipy.signal.find_peaks(-local_chunk[:, i], height=height, distance=dist_peaks)[0]
                elif sign_peaks == 'positive':
                    peaktimes = scipy.signal.find_peaks(local_chunk[:, i], height=height, distance=dist_peaks)[0]
                elif sign_peaks == 'both':
                    peaktimes = scipy.signal.find_peaks(numpy.abs(local_chunk[:, i]), height=height, distance=dist_peaks)[0]
                else:
                    peaktimes = numpy.empty(0, dtype=numpy.uint32)

                idx = (peaktimes >= local_borders[0]) & (peaktimes < local_borders[1])
                peaktimes = peaktimes[idx]

                if ignore_dead_times:
                    if dead_indices[0] != dead_indices[1]:
                        is_included = numpy.in1d(
                            peaktimes + t_offset,
                            all_dead_times[dead_indices[0]:dead_indices[1]]
                        )
                        peaktimes = peaktimes[~is_included]

                peaktimes = peaktimes.astype(numpy.uint32)
                found_peaktimes.append(peaktimes)

            all_peaktimes = numpy.concatenate(found_peaktimes) # i.e. concatenate once for efficiency
            local_peaktimes, local_indices = numpy.unique(all_peaktimes, return_inverse=True)

            if len(local_peaktimes) > 0:

                diff_times = (local_peaktimes[-1] - local_peaktimes[0]) + 1
                all_times = numpy.zeros((N_e, diff_times), dtype=numpy.bool)

                padded_peaks = (local_peaktimes - local_peaktimes[0]).astype(numpy.int32)
                min_times = numpy.maximum(padded_peaks - safety_time, 0)
                max_times = numpy.minimum(padded_peaks + safety_time + 1, diff_times + 1)
                test_extremas = numpy.zeros((N_e, diff_times + 1), dtype=numpy.bool)
                for i in range(N_e):
                    test_extremas[i, found_peaktimes[i] - local_peaktimes[0]] = True

                n_times = len(all_peaktimes)
                shuffling = numpy.random.permutation(numpy.arange(n_times))
                all_idx = numpy.take(all_peaktimes, shuffling)
                argmax_peak = local_indices[shuffling]

                # print "Selection of the peaks with spatio-temporal masks..."
                for midx, peak in zip(argmax_peak, all_idx):
                    if (elt_count_neg + elt_count_pos) == nb_elts:
                        break

                    all_elecs = numpy.where(test_extremas[:, peak - local_peaktimes[0]])[0]
                    data = local_chunk[peak, all_elecs]

                    #target_area = test_extremas[:, min_times[midx]:max_times[midx]].sum(1)
                    #all_elecs = numpy.where(target_area)[0]
                    #data = local_chunk[peak, all_elecs]

                    if sign_peaks == 'negative':
                        elec = numpy.argmin(data)
                        negative_peak = True
                    elif sign_peaks == 'positive':
                        elec = numpy.argmax(data)
                        negative_peak = False
                    elif sign_peaks == 'both':
                        if N_e == 1:
                            if data < 0:
                                negative_peak = True
                            elif data > 0:
                                negative_peak = False
                            elec = 0
                        else:
                            if numpy.abs(numpy.max(data)) > numpy.abs(numpy.min(data)):
                                elec = numpy.argmax(data)
                                negative_peak = False
                            else:
                                elec = numpy.argmin(data)
                                negative_peak = True

                    elec = all_elecs[elec]

                    if groups[elec] < upper_bounds:

                        indices = nodes_indices[elec]
                        myslice = all_times[indices, min_times[midx]:max_times[midx]]

                        if not myslice.any():

                            sub_mat = local_chunk[peak - snippet_duration:peak + snippet_duration + 1, elec]

                            if reject_noise:
                                slice_window = sub_mat[snippet_duration - noise_window: snippet_duration + noise_window + 1]
                                value = numpy.linalg.norm(slice_window)/noise_levels[elec]
                                is_noise = value < rejection_threshold
                            else:
                                is_noise = False
                            
                            if not is_noise:
                                if alignment:
                                    smoothed = True
                                    try:
                                        f = scipy.interpolate.UnivariateSpline(xdata, sub_mat, s=local_factors[elec], k=3)
                                    except Exception:
                                        smoothed = False
                                        f = scipy.interpolate.UnivariateSpline(xdata, sub_mat, k=3, s=0)
                                
                                    if negative_peak:
                                        rmin = (numpy.argmin(f(cdata)) - xoff)/over_factor
                                    else:
                                        rmin = (numpy.argmax(f(cdata)) - xoff)/over_factor
                                    ddata = numpy.linspace(rmin-template_shift, rmin+template_shift, N_t)

                                    if smoothed:
                                        f = scipy.interpolate.UnivariateSpline(xdata, sub_mat, s=0, k=3)

                                    sub_mat = f(ddata).astype(numpy.float32)

                                if negative_peak:
                                    elts_neg[:, elt_count_neg] = sub_mat
                                else:
                                    elts_pos[:, elt_count_pos] = sub_mat

                                if negative_peak:
                                    elt_count_neg += 1
                                else:
                                    elt_count_pos += 1

                                groups[elec] += 1
                                all_times[indices, min_times[midx]:max_times[midx]] = True
                                test_extremas[elec, peak - local_peaktimes[0]] = False

    sys.stderr.flush()

    print_and_log(["Node %d has collected %d waveforms" % (comm.rank, elt_count_pos + elt_count_neg)], 'debug', logger)

    if sign_peaks in ['negative', 'both']:
        gdata_neg = gather_array(elts_neg[:, :elt_count_neg].T, comm, 0, 1)
    if sign_peaks in ['positive', 'both']:
        gdata_pos = gather_array(elts_pos[:, :elt_count_pos].T, comm, 0, 1)

    nb_waveforms = 0

    if comm.rank == 0:
        # DO PCA on elts and store the basis obtained.

        if sign_peaks in ['negative', 'both']:
            nb_waveforms += gdata_neg.shape[0]
        if sign_peaks in ['positive', 'both']:
            nb_waveforms += gdata_pos.shape[0]

    nb_waveforms = all_gather_array(numpy.array([nb_waveforms], dtype=numpy.float32), comm, 0)[0]

    if comm.rank == 0:
        print_and_log(["Found %d waveforms over %d requested" % (nb_waveforms, int(nb_elts * comm.size))], 'default', logger)

        if nb_waveforms == 0:
            print_and_log(['No waveforms found! Are the data properly loaded??'], 'error', logger)
    
    if nb_waveforms == 0:
        sys.exit(0)

    if comm.rank == 0:
        res = {}
        pca = None
        pca_pos = None
        pca_neg = None
        warning_n_t = False
        if sign_peaks in ['negative', 'both']:
            if len(gdata_neg) > 0:
                pca = PCA(output_dim)
                if use_hanning:
                    pca.fit(gdata_neg * hanning_filter)
                else:
                    pca.fit(gdata_neg)
                res['proj'] = pca.components_.T.astype(numpy.float32)
                pca_neg = numpy.sum(pca.explained_variance_ratio_)
            else:
                res['proj'] = numpy.identity(int(output_dim), dtype=numpy.float32)
            res['rec'] = res['proj'].T
            res['waveform'] = numpy.median(gdata_neg, 0)
            # dispersion = numpy.std(gdata_neg, 0) / numpy.median(stds)
            # ratio = numpy.sum(dispersion > 1.1) / float(len(dispersion))
            # if ratio < 0.25:
            #     print_and_log(["Time window N_t in [detection] seems too large!"], 'info', logger)
            #     warning_n_t = True
            # elif ratio == 1:
            #     print_and_log(["Time window N_t in [detection] seems too small!"], 'info', logger)
            #     warning_n_t = True
            idx = numpy.random.permutation(numpy.arange(gdata_neg.shape[0]))[:2500]
            res['waveforms'] = gdata_neg[idx, :]
        if sign_peaks in ['positive', 'both']:
            if len(gdata_pos) > 0:
                pca = PCA(output_dim)
                if use_hanning:
                    pca.fit(gdata_pos*hanning_filter)
                else:
                    pca.fit(gdata_pos)
                res['proj_pos'] = pca.components_.T.astype(numpy.float32)
                pca_pos = numpy.sum(pca.explained_variance_ratio_)
            else:
                res['proj_pos'] = numpy.identity(int(output_dim), dtype=numpy.float32)
            res['rec_pos'] = res['proj_pos'].T
            res['waveform_pos'] = numpy.median(gdata_pos, 0)
            # dispersion = numpy.std(gdata_pos, 0) / numpy.median(stds)
            # ratio = numpy.sum(dispersion > 1.1) / float(len(dispersion))
            # if ratio < 0.25 and not warning_n_t:
            #     print_and_log(["Time window N_t in [detection] seems too large!"], 'info', logger)
            # elif ratio == 1 and not warning_n_t:
            #     print_and_log(["Time window N_t in [detection] seems too small!"], 'info', logger)
            idx = numpy.random.permutation(numpy.arange(gdata_pos.shape[0]))[:2500]
            res['waveforms_pos'] = gdata_pos[idx, :]

        bfile = h5py.File(file_out_suff + '.basis.hdf5', 'r+', libver='earliest')
        io.write_datasets(bfile, res.keys(), res, compression=hdf5_compress)
        if sign_peaks == 'positive':
            print_and_log(["A basis with %s dimensions has been built" % res['proj_pos'].shape[1]], 'info', logger)
        elif sign_peaks == 'negative':
            print_and_log(["A basis with %s dimensions has been built" % res['proj'].shape[1]], 'info', logger)
        elif sign_peaks == 'both':
            print_and_log(["Two basis with %s dimensions has been built" % res['proj'].shape[1]], 'debug', logger)
        if pca_pos is not None:
            print_and_log(["The percentage of variance explained is %s for positive spikes" % pca_pos], 'debug', logger)
        if pca_neg is not None:
            print_and_log(["The percentage of variance explained is %s for negative spikes" % pca_neg], 'debug', logger)

        bfile.close()

    comm.Barrier()

    if matched_filter:

        if comm.rank == 0:
            print_and_log(["Because of matched filters, need to recompute the thresholds..."], 'default', logger)

        if do_spatial_whitening:
            spatial_whitening = io.load_data(params, 'spatial_whitening')
            if use_gpu:
                spatial_whitening = cmt.CUDAMatrix(spatial_whitening, copy_on_host=False)
        if do_temporal_whitening:
            temporal_whitening = io.load_data(params, 'temporal_whitening')

        if sign_peaks in ['negative', 'both']:
            waveform_neg = io.load_data(params, 'waveform')[::-1]
            waveform_neg /= (numpy.abs(numpy.sum(waveform_neg)) * len(waveform_neg))
        if sign_peaks in ['positive', 'both']:
            waveform_pos = io.load_data(params, 'waveform-pos')[::-1]
            waveform_pos /= (numpy.abs(numpy.sum(waveform_pos)) * len(waveform_pos))

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

            local_chunk /= thresholds

            if sign_peaks in ['negative', 'both']:
                tmp_chunk = scipy.ndimage.filters.convolve1d(local_chunk, waveform_neg, axis=0, mode='constant')
                thresholds = numpy.zeros(N_e, dtype=numpy.float32)
                for i in range(N_e):
                    u = numpy.median(tmp_chunk[:, i], 0)
                    thresholds[i] = numpy.median(numpy.abs(tmp_chunk[:, i] - u), 0)
                gdata = gather_array(thresholds, comm)
                if comm.rank == 0:
                    gdata = gdata.reshape((comm.size, N_e))
                    thresholds = numpy.mean(gdata, 0)
                    bfile = h5py.File(file_out_suff + '.basis.hdf5', 'r+', libver='earliest')
                    io.write_datasets(bfile, ['matched_thresholds'], {'matched_thresholds': thresholds}, compression=hdf5_compress)
                    bfile.close()
                comm.Barrier()

            if sign_peaks in ['positive', 'both']:
                tmp_chunk = scipy.ndimage.filters.convolve1d(local_chunk, waveform_pos, axis=0, mode='constant')
                thresholds = numpy.zeros(N_e, dtype=numpy.float32)
                for i in range(N_e):
                    u = numpy.median(tmp_chunk[:, i], 0)
                    thresholds[i] = numpy.median(numpy.abs(tmp_chunk[:, i] - u), 0)
                gdata = gather_array(thresholds, comm)
                if comm.rank == 0:
                    gdata = gdata.reshape((comm.size, N_e))
                    thresholds = numpy.mean(gdata, 0)
                    bfile = h5py.File(file_out_suff + '.basis.hdf5', 'r+', libver='earliest')
                    io.write_datasets(bfile, ['matched_thresholds_pos'], {'matched_thresholds_pos': thresholds}, compression=hdf5_compress)
                    bfile.close()
                comm.Barrier()

    data_file.close()
