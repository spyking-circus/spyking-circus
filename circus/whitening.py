from .shared.utils import *
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
    max_silence_1    = int(20*sampling_rate / comm.size)
    max_silence_2    = 5000
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)
    #################################################################

    if comm.rank == 0:
        print "Analyzing data to get whitening matrices and thresholds..."

    borders, nb_chunks, chunk_len, last_chunk_len = io.analyze_data(params, chunk_size)

    if nb_chunks < comm.size:
        if comm.rank == 0:
            io.print_error(["More nodes than 1 min chunks to load: decrease n_cpu or provide more data"])
        sys.exit(0)

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
            bfile      = h5py.File(file_out + '.basis.hdf5', 'w')
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
        local_peaktimes = local_peaktimes[idx]

        if len(local_peaktimes) > 0:

            diff_times      = local_peaktimes[-1]-local_peaktimes[0]
            all_times       = numpy.zeros((N_e, diff_times+1), dtype=numpy.bool)
            min_times       = numpy.maximum(local_peaktimes - local_peaktimes[0] - safety_time, 0)
            max_times       = numpy.minimum(local_peaktimes - local_peaktimes[0] + safety_time + 1, diff_times)
            argmax_peak     = numpy.random.permutation(numpy.arange(len(local_peaktimes)))
            all_idx         = local_peaktimes[argmax_peak]

            #print "Selection of the peaks with spatio-temporal masks..."
            for idx, sidx in zip(argmax_peak, all_idx):
                elec    = numpy.argmax(numpy.abs(local_chunk[sidx]))
                indices = inv_nodes[edges[nodes[elec]]]
                myslice = all_times[indices, min_times[idx]:max_times[idx]]
                peak    = local_peaktimes[idx]
                all_times[indices, min_times[idx]:max_times[idx]] = True
        else:
            all_times   = numpy.zeros((N_e, len(local_chunk)), dtype=numpy.bool)
    
    all_times_Ne   = numpy.any(all_times, 0)
    subset         = numpy.where(all_times_Ne == False)[0]
    local_silences = local_chunk[subset, :][:max_silence_1]
    all_silences   = gather_array(local_silences, comm, 0, 1)
    local_res      = []

    for elec in all_electrodes[numpy.arange(comm.rank, nb_temp_white, comm.size)]:
        res            = numpy.zeros((0, N_t), dtype=numpy.float32)
        scount         = 0
        indices        = inv_nodes[edges[nodes[elec]]]
        all_times_elec = numpy.any(all_times[indices], 0)
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
    all_res   = gather_array(local_res.flatten(), comm, 0, 1)
    all_elecs = gather_array(nb_elecs, comm, 0, 1)

    if comm.rank == 0 and (do_spatial_whitening or do_temporal_whitening):
        try:
            nb_silences     = numpy.sum(all_elecs > 0)
            all_res         = all_res.reshape((nb_silences, N_t**2))
        except Exception:
            print io.print_info(["No silent periods detected: something wrong with the parameters?"])
        all_res             = numpy.sum(all_res, 0)
        all_res             = all_res.reshape((N_t, N_t))/numpy.sum(all_elecs)
        temporal_whitening  = get_whitening_matrix(all_res.astype(numpy.double), fudge=1e-3)[template_shift].astype(numpy.float32)
        temporal_whitening /= temporal_whitening.sum()

        print "We found", len(all_silences), "times without spikes for whitening matrices..."
        spatial_whitening = get_whitening_matrix(all_silences.astype(numpy.double)).astype(numpy.float32)
        bfile = h5py.File(file_out + '.basis.hdf5', 'r+')
        io.write_datasets(bfile, ['spatial', 'temporal'], {'spatial' : spatial_whitening, 'temporal' : temporal_whitening})
        bfile.close()
        print "Because of whitening, we need to recompute the thresholds..."

    del all_res, all_silences
    comm.Barrier()

    if do_spatial_whitening or do_temporal_whitening:

        spatial_whitening  = io.load_data(params, 'spatial_whitening')
        temporal_whitening = io.load_data(params, 'temporal_whitening')

        for gidx in [all_chunks[comm.rank]]:
            local_chunk, local_shape = io.load_chunk(params, gidx, chunk_len, nodes=nodes)
            if do_spatial_whitening:
                local_chunk = numpy.dot(local_chunk, spatial_whitening)
            if do_temporal_whitening:
                for i in xrange(N_e):
                    local_chunk[:, i] = numpy.convolve(local_chunk[:, i], temporal_whitening, 'same')

            thresholds = numpy.zeros(N_e, dtype=numpy.float32)
            for i in xrange(N_e):
                u             = numpy.median(local_chunk[:, i], 0)
                thresholds[i] = numpy.median(numpy.abs(local_chunk[:, i] - u), 0)
            gdata      = gather_array(thresholds, comm)
            if comm.rank == 0:
                gdata      = gdata.reshape((comm.size, N_e))
                thresholds = numpy.mean(gdata, 0)
                bfile      = h5py.File(file_out + '.basis.hdf5', 'r+')
                bfile.pop('thresholds')
                io.write_datasets(bfile, ['thresholds'], {'thresholds' : thresholds})
                bfile.close()
            comm.Barrier()

    if comm.rank == 0:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        n_elec = min(int(numpy.sqrt(N_e)), 5)
        plot.view_fit(filename, t_start=0, t_stop=1, fit_on=False, square=True, n_elec=n_elec, save=[plot_path, 'electrodes'])

    # Part 2: Basis
    numpy.random.seed(422)

    #################################################################
    sampling_rate  = params.getint('data', 'sampling_rate')
    N_e            = params.getint('data', 'N_e')
    N_t            = params.getint('data', 'N_t')
    N_total        = params.getint('data', 'N_total')
    dist_peaks     = params.getint('data', 'dist_peaks')
    template_shift = params.getint('data', 'template_shift')
    file_out       = params.get('data', 'file_out')
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
    #################################################################


    if comm.rank == 0:
        print "Searching spikes to construct the PCA basis..."

    if do_spatial_whitening or do_temporal_whitening:
        spatial_whitening  = io.load_data(params, 'spatial_whitening')
        temporal_whitening = io.load_data(params, 'temporal_whitening')

    borders, nb_chunks, chunk_len, last_chunk_len = io.analyze_data(params, chunk_size)

    groups    = {}
    for i in xrange(N_e):
        groups[i] = 0

    # I guess this is more relevant, to take signals from all over the recordings
    all_chunks     = numpy.random.permutation(numpy.arange(nb_chunks))
    max_elts_elec /= comm.size
    nb_elts       /= comm.size
    elts           = numpy.zeros((N_t, nb_elts), dtype=numpy.float32)
    chunks_to_load = all_chunks[numpy.arange(comm.rank, nb_chunks, comm.size)]

    thresholds = io.load_data(params, 'thresholds')

    if comm.rank == 0:
        pbar = get_progressbar(nb_elts)



    for gcount, gidx in enumerate(chunks_to_load):

        if (elt_count < nb_elts):
            #print "Node", comm.rank, "is analyzing chunk", gidx, "/", nb_chunks, " ..."
            local_chunk, local_shape = io.load_chunk(params, gidx, chunk_len, chunk_size, nodes=nodes)
            if do_spatial_whitening:
                local_chunk = numpy.dot(local_chunk, spatial_whitening)
            if do_temporal_whitening:
                for i in xrange(N_e):
                    local_chunk[:, i] = numpy.convolve(local_chunk[:, i], temporal_whitening, 'same')

            #print "Extracting the peaks..."
            all_peaktimes = numpy.zeros(0, dtype=numpy.int32)
            all_minimas   = numpy.zeros(0, dtype=numpy.int32)
            for i in xrange(N_e):
                peaktimes     = algo.detect_peaks(local_chunk[:, i], thresholds[i], valley=True, mpd=dist_peaks)
                all_peaktimes = numpy.concatenate((all_peaktimes, peaktimes))
                all_minimas   = numpy.concatenate((all_minimas, i*numpy.ones(len(peaktimes), dtype=numpy.int32)))

            #print "Removing the useless borders..."
            local_borders   = (template_shift, local_shape - template_shift)
            idx             = (all_peaktimes >= local_borders[0]) & (all_peaktimes < local_borders[1])
            all_peaktimes   = all_peaktimes[idx]
            all_minimas     = all_minimas[idx]

            local_peaktimes = numpy.unique(all_peaktimes)

            if len(local_peaktimes) > 0:

                diff_times      = local_peaktimes[-1]-local_peaktimes[0]
                all_times       = numpy.zeros((N_e, diff_times+1), dtype=numpy.bool)
                min_times       = numpy.maximum(local_peaktimes - local_peaktimes[0] - safety_time, 0)
                max_times       = numpy.minimum(local_peaktimes - local_peaktimes[0] + safety_time + 1, diff_times)

                n_times         = len(local_peaktimes)
                argmax_peak     = numpy.random.permutation(numpy.arange(n_times))
                all_idx         = local_peaktimes[argmax_peak]

                #print "Selection of the peaks with spatio-temporal masks..."
                for idx, sidx in zip(argmax_peak, all_idx):
                    if elt_count == nb_elts:
                        break
                    elec    = numpy.argmin(local_chunk[sidx])
                    indices = inv_nodes[edges[nodes[elec]]]
                    myslice = all_times[indices, min_times[idx]:max_times[idx]]
                    peak    = local_peaktimes[idx]
                    is_local_min = elec in all_minimas[all_peaktimes == peak]
                    if is_local_min and not myslice.any():
                        if groups[elec] < max_elts_elec:
                            elts[:, elt_count]  = local_chunk[peak - template_shift:peak + template_shift + 1, elec]
                            groups[elec]       += 1
                            elt_count          += 1
                        all_times[indices, min_times[idx]:max_times[idx]] = True

            if comm.rank == 0:
                pbar.update(elt_count)

        if comm.rank == 0:
            if (elt_count < (gcount+1)*max_elts_elec/len(chunks_to_load)):
               pbar.update((gcount+1)*max_elts_elec/len(chunks_to_load))

    if comm.rank == 0:
        pbar.finish()

    gdata = gather_array(elts[:, :elt_count], comm, 0)

    if comm.rank == 0:
        #DO PCA on elts and store the basis obtained.
        print "We found", gdata.shape[1], "spikes over", int(nb_elts*comm.size), "requested"
        pca      = mdp.nodes.PCANode(output_dim=output_dim)
        res_pca  = pca(elts.astype(numpy.double).T)
        bfile    = h5py.File(file_out + '.basis.hdf5', 'r+')
        io.write_datasets(bfile, ['proj', 'rec'], {'proj' : pca.get_projmatrix().astype(numpy.float32), 
                                                    'rec' : pca.get_recmatrix().astype(numpy.float32)})
        io.print_info(["A basis with %s dimensions has been built" %pca.get_projmatrix().shape[1]])
        bfile.close()