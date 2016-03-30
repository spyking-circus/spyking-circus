from .shared.utils import *
import circus.shared.algorithms as algo
from .shared import plot

def main(filename, params, nb_cpu, nb_gpu, use_gpu):
    # Part 1: Whitening
    numpy.random.seed(420)
    import h5py

    #################################################################
    sampling_rate  = params.getint('data', 'sampling_rate')
    N_e            = params.getint('data', 'N_e')
    N_t            = params.getint('data', 'N_t')
    N_total        = params.getint('data', 'N_total')
    skip_artefact  = params.getboolean('data', 'skip_artefact')
    spike_thresh   = params.getfloat('data', 'spike_thresh')
    dist_peaks     = params.getint('data', 'dist_peaks')
    template_shift = params.getint('data', 'template_shift')
    file_out       = params.get('data', 'file_out')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    chunk_size       = params.getint('whitening', 'chunk_size')
    plot_path        = os.path.join(params.get('data', 'data_file_noext'), 'plots')
    nodes, edges     = io.get_nodes_and_edges(params)
    safety_time      = int(params.getfloat('whitening', 'safety_time')*sampling_rate*1e-3)
    nb_temp_white    = min(max(20, comm.size), N_e)
    max_silence_1    = int(20*sampling_rate // comm.size)
    max_silence_2    = 5000
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)
    #################################################################

    if comm.rank == 0:
        io.print_and_log(["Analyzing data to get whitening matrices and thresholds..."], 'default', params)

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

    borders, nb_chunks, chunk_len, last_chunk_len = io.analyze_data(params, chunk_size)

    if nb_chunks < comm.size:

        res        = io.data_stats(params, show=False)
        chunk_size = res*sampling_rate//comm.size
        if comm.rank == 0:
            io.print_and_log(["Too much cores, automatically resizing the data chunks"], 'debug', params)

        borders, nb_chunks, chunk_len, last_chunk_len = io.analyze_data(params, chunk_size)

    # I guess this is more relevant, to take signals from all over the recordings
    all_chunks     = numpy.random.permutation(numpy.arange(nb_chunks))
    all_electrodes = numpy.random.permutation(N_e)

    for gidx in [all_chunks[comm.rank]]:

        #print "Node", comm.rank, "is analyzing chunk", gidx,  "/", nb_chunks, " ..."
        local_chunk, local_shape = io.load_chunk(params, gidx, chunk_len, chunk_size, nodes=nodes)
        #print "Node", comm.rank, "computes the median absolute deviations in a random chunk"
        thresholds = numpy.zeros(N_e, dtype=numpy.float32)
        for i in xrange(N_e):
            u             = numpy.median(local_chunk[:, i], 0)
            thresholds[i] = numpy.median(numpy.abs(local_chunk[:, i] - u), 0)
        gdata      = gather_array(thresholds, comm)
        if comm.rank == 0:
            gdata      = gdata.reshape((comm.size, N_e))
            thresholds = numpy.mean(gdata, 0)
            bfile      = h5py.File(file_out + '.basis.hdf5', 'w', libver='latest')
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
            all_times_elec = numpy.any(numpy.take(all_times, indices), 0)
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
                io.print_and_log(["No silent periods detected: something wrong with the parameters?"], 'error', params)
            all_res             = numpy.sum(all_res, 0)
            all_res             = all_res.reshape((N_t, N_t))/numpy.sum(all_elecs)
            temporal_whitening  = get_whitening_matrix(all_res.astype(numpy.double), fudge=1e-3)[template_shift].astype(numpy.float32)
            temporal_whitening /= temporal_whitening.sum()
            to_write['temporal'] = temporal_whitening

        if do_spatial_whitening:
            if len(all_silences)/sampling_rate == 0:
                io.print_and_log(["No silent periods detected: something wrong with the parameters?"], 'error', params)
            spatial_whitening = get_whitening_matrix(all_silences.astype(numpy.double)).astype(numpy.float32)
            to_write['spatial'] = spatial_whitening
            io.print_and_log(["Found %gs without spikes for whitening matrices..." %(len(all_silences)/sampling_rate)], 'default', params)
        
        bfile = h5py.File(file_out + '.basis.hdf5', 'r+', libver='latest')
        io.write_datasets(bfile, to_write.keys(), to_write)
        bfile.close()

    del all_silences
    comm.Barrier()

    if do_spatial_whitening or do_temporal_whitening:

        if comm.rank == 0:
            io.print_and_log(["Because of whitening, need to recompute the thresholds..."], 'default', params)

        if do_spatial_whitening:
            spatial_whitening  = io.load_data(params, 'spatial_whitening')
            if use_gpu:
                spatial_whitening = cmt.CUDAMatrix(spatial_whitening)
        if do_temporal_whitening:
            temporal_whitening = io.load_data(params, 'temporal_whitening')

        for gidx in [all_chunks[comm.rank]]:
            local_chunk, local_shape = io.load_chunk(params, gidx, chunk_len, chunk_size, nodes=nodes)
            if do_spatial_whitening:
                if use_gpu:
                    local_chunk = cmt.CUDAMatrix(local_chunk)
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
                bfile      = h5py.File(file_out + '.basis.hdf5', 'r+', libver='latest')
                bfile.pop('thresholds')
                io.write_datasets(bfile, ['thresholds'], {'thresholds' : thresholds})
                bfile.close()
            comm.Barrier()

    # Part 2: Basis
    numpy.random.seed(422)

    #################################################################
    sampling_rate  = params.getint('data', 'sampling_rate')
    N_e            = params.getint('data', 'N_e')
    N_t            = params.getint('data', 'N_t')
    N_total        = params.getint('data', 'N_total')
    skip_artefact  = params.getboolean('data', 'skip_artefact')
    dist_peaks     = params.getint('data', 'dist_peaks')
    template_shift = params.getint('data', 'template_shift')
    alignment      = params.getboolean('data', 'alignment')
    file_out       = params.get('data', 'file_out')
    spike_thresh   = params.getfloat('data', 'spike_thresh')
    stationary     = params.getboolean('data', 'stationary')
    nodes, edges   = io.get_nodes_and_edges(params)
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    chunk_size       = params.getint('data', 'chunk_size')
    safety_time      = int(params.getfloat('whitening', 'safety_time')*sampling_rate*1e-3)
    max_elts_elec    = params.getint('whitening', 'max_elts')
    nb_elts          = int(params.getfloat('whitening', 'nb_elts')*N_e*max_elts_elec)
    output_dim       = params.getfloat('whitening', 'output_dim')
    elt_count        = 0
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)
    take_all         = False
    #################################################################


    if comm.rank == 0:
        io.print_and_log(["Searching spikes to construct the PCA basis..."], 'default', params)
        
    borders, nb_chunks, chunk_len, last_chunk_len = io.analyze_data(params, chunk_size)

    if nb_chunks < comm.size:

        res        = io.data_stats(params, show=False)
        chunk_size = res*sampling_rate//comm.size
        if comm.rank == 0:
            io.print_and_log(["Too much cores, automatically resizing the data chunks"], 'debug', params)

        borders, nb_chunks, chunk_len, last_chunk_len = io.analyze_data(params, chunk_size)

    groups    = {}
    for i in xrange(N_e):
        groups[i] = 0

    # I guess this is more relevant, to take signals from all over the recordings
    all_chunks     = numpy.random.permutation(numpy.arange(nb_chunks))
    max_elts_elec //= comm.size
    nb_elts       //= comm.size
    elts           = numpy.zeros((N_t, nb_elts), dtype=numpy.float32)
    chunks_to_load = all_chunks[comm.rank::comm.size]

    thresholds = io.load_data(params, 'thresholds')

    if comm.rank == 0:
        pbar = get_progressbar(nb_elts)

    if alignment:
        cdata = numpy.linspace(-template_shift, template_shift, 5*N_t)
        xdata = numpy.arange(-2*template_shift, 2*template_shift+1)

    for gcount, gidx in enumerate(chunks_to_load):

        if (elt_count < nb_elts):
            #print "Node", comm.rank, "is analyzing chunk", gidx, "/", nb_chunks, " ..."
            local_chunk, local_shape = io.load_chunk(params, gidx, chunk_len, chunk_size, nodes=nodes)
            if do_spatial_whitening:
                if use_gpu:
                    local_chunk = cmt.CUDAMatrix(local_chunk)
                    local_chunk = local_chunk.dot(spatial_whitening).asarray()
                else:
                    local_chunk = numpy.dot(local_chunk, spatial_whitening)
            if do_temporal_whitening:
                local_chunk = scipy.ndimage.filters.convolve1d(local_chunk, temporal_whitening, axis=0, mode='constant')

            #print "Extracting the peaks..."
            all_peaktimes = numpy.zeros(0, dtype=numpy.int32)
            all_minimas   = numpy.zeros(0, dtype=numpy.int32)
            if not stationary:
                for i in xrange(N_e):
                    u             = numpy.median(local_chunk[:, i], 0)
                    thresholds[i] = numpy.median(numpy.abs(local_chunk[:, i] - u), 0)
                thresholds *= spike_thresh

            for i in xrange(N_e):
                peaktimes     = algo.detect_peaks(local_chunk[:, i], thresholds[i], valley=True, mpd=dist_peaks)
                if skip_artefact:
                    real_peaktimes = numpy.zeros(0, dtype=numpy.int32)
                    indices        = numpy.take(inv_nodes, edges[nodes[i]])
                    for idx in xrange(len(peaktimes)):
                        values      = numpy.take(local_chunk[idx], indices)
                        is_artefact = numpy.any(values < -20*numpy.take(thresholds, indices))
                        if not is_artefact:
                            real_peaktimes = numpy.concatenate((real_peaktimes, [idx]))
                    peaktimes = numpy.take(peaktimes, real_peaktimes)
                all_peaktimes = numpy.concatenate((all_peaktimes, peaktimes))
                all_minimas   = numpy.concatenate((all_minimas, i*numpy.ones(len(peaktimes), dtype=numpy.int32)))

            #print "Removing the useless borders..."
            if alignment:
                local_borders = (2*template_shift, local_shape - 2*template_shift)
            else:
                local_borders = (template_shift, local_shape - template_shift)
            idx             = (all_peaktimes >= local_borders[0]) & (all_peaktimes < local_borders[1])
            all_peaktimes   = numpy.compress(idx, all_peaktimes)
            all_minimas     = numpy.compress(idx, all_minimas)

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
                    if elt_count == nb_elts:
                        break
                    elec    = numpy.argmin(local_chunk[peak])
                    indices = numpy.take(inv_nodes, edges[nodes[elec]])
                    myslice = all_times[indices, min_times[midx]:max_times[midx]]
                    is_local_min = elec in all_minimas[all_peaktimes == peak]
                    if is_local_min and not myslice.any():
                        upper_bounds = max_elts_elec
                        if take_all:
                            upper_bounds //= len(indices)

                        if groups[elec] < upper_bounds:

                            elts[:, elt_count]  = local_chunk[peak - template_shift:peak + template_shift + 1, elec]
                            if alignment:
                                ydata    = local_chunk[peak-2*template_shift:peak+2*template_shift+1, elec]
                                f        = scipy.interpolate.UnivariateSpline(xdata, ydata, s=0)
                                smoothed = smooth(f(cdata), template_shift)
                                rmin     = (numpy.argmin(smoothed) - len(cdata)/2.)/5.
                                ddata    = numpy.linspace(rmin-template_shift, rmin+template_shift, N_t)
                                elts[:, elt_count] = f(ddata).astype(numpy.float32)
                                
                            elt_count         += 1

                        groups[elec] += 1
                        all_times[indices, min_times[midx]:max_times[midx]] = True

            if comm.rank == 0:
                pbar.update(elt_count)

        if comm.rank == 0:
            if (elt_count < (gcount+1)*max_elts_elec//len(chunks_to_load)):
               pbar.update((gcount+1)*max_elts_elec//len(chunks_to_load))

    if comm.rank == 0:
        pbar.finish()

    io.print_and_log(["Node %d has collected %d waveforms" %(comm.rank, elt_count)], 'debug', params)
    gdata = gather_array(elts[:, :elt_count].T, comm, 0, 1)

    if comm.rank == 0:
        #DO PCA on elts and store the basis obtained.
        io.print_and_log(["Found %d waveforms over %d requested" %(gdata.shape[0], int(nb_elts*comm.size))], 'default', params)
        pca = PCA(output_dim, copy=False)
        res = {}     
        if len(gdata) > 0:
            res_pca     = pca.fit_transform(gdata.astype(numpy.double)).astype(numpy.float32)
            res['proj'] = pca.components_.T.astype(numpy.float32)
        else:
            res['proj'] = numpy.identity(N_t, dtype=numpy.float32)
        res['rec']  = res['proj'].T
        res['waveforms'] = gdata[:, :100]
        bfile    = h5py.File(file_out + '.basis.hdf5', 'r+', libver='latest')
        io.write_datasets(bfile, res.keys(), res)
        io.print_and_log(["A basis with %s dimensions has been built" %res['proj'].shape[1]], 'info', params)
        bfile.close()