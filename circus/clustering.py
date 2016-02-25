from .shared.utils import *
import circus.shared.algorithms as algo
from .shared import plot



def main(filename, params, nb_cpu, nb_gpu, use_gpu):

    import h5py
    parallel_hdf5 = h5py.get_config().mpi

    #################################################################
    sampling_rate  = params.getint('data', 'sampling_rate')
    N_e            = params.getint('data', 'N_e')
    N_t            = params.getint('data', 'N_t')
    N_total        = params.getint('data', 'N_total')
    dist_peaks     = params.getint('data', 'dist_peaks')
    skip_artefact  = params.getboolean('data', 'skip_artefact')
    template_shift = params.getint('data', 'template_shift')
    alignment      = params.getboolean('data', 'alignment')
    file_out       = params.get('data', 'file_out')
    file_out_suff  = params.get('data', 'file_out_suff')
    spike_thresh   = params.getfloat('data', 'spike_thresh')
    stationary     = params.getboolean('data', 'stationary')
    if params.get('data', 'global_tmp'):
        tmp_path_loc = os.path.join(os.path.abspath(params.get('data', 'data_file_noext')), 'tmp')
    else:
        tmp_path_loc = tempfile.gettempdir()
    plot_path      = os.path.join(params.get('data', 'data_file_noext'), 'plots')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    safety_time    = int(params.getfloat('clustering', 'safety_time')*sampling_rate*1e-3)
    safety_space   = params.getboolean('clustering', 'safety_space')
    nodes, edges   = io.get_nodes_and_edges(params)
    chunk_size     = params.getint('data', 'chunk_size')
    max_elts_elec  = params.getint('clustering', 'max_elts')
    nb_elts        = int(params.getfloat('clustering', 'nb_elts')*N_e*max_elts_elec)
    nb_repeats     = params.getint('clustering', 'nb_repeats')
    nclus_min      = params.getfloat('clustering', 'nclus_min')
    max_clusters   = params.getint('clustering', 'max_clusters')
    m_ratio        = 0.01  
    make_plots     = params.get('clustering', 'make_plots')
    sim_same_elec  = params.getfloat('clustering', 'sim_same_elec')
    noise_thr      = params.getfloat('clustering', 'noise_thr')
    remove_mixture = params.getboolean('clustering', 'remove_mixture')
    extraction     = params.get('clustering', 'extraction')
    smart_search   = numpy.ones(N_e, dtype=numpy.float32)*params.getfloat('clustering', 'smart_search')
    test_clusters  = params.getboolean('clustering', 'test_clusters')
    tmp_limits     = params.get('fitting', 'amp_limits').replace('(', '').replace(')', '').split(',')
    amp_limits     = map(float, tmp_limits)
    elt_count      = 0
    sub_output_dim = 5   
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)
    to_write         = ['data_', 'clusters_', 'debug_', 'w_', 'pca_', 'times_']
    #################################################################

    basis_proj, basis_rec = io.load_data(params, 'basis')
    thresholds = io.load_data(params, 'thresholds')
    if do_spatial_whitening:
        spatial_whitening  = io.load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = io.load_data(params, 'temporal_whitening')

    result   = {}

    if test_clusters:
        injected_spikes = io.load_data(params, 'injected_spikes')

    if comm.rank == 0:
        if not os.path.exists(tmp_path_loc):
            os.makedirs(tmp_path_loc)

    if alignment:
        cdata = numpy.linspace(-template_shift, template_shift, 5*N_t)
        xdata = numpy.arange(-2*template_shift, 2*template_shift+1)

    comm.Barrier()

    for i in xrange(N_e):
        result['loc_times_' + str(i)] = numpy.zeros(0, dtype=numpy.int32)
        result['times_' + str(i)]     = numpy.zeros(0, dtype=numpy.int32)
        result['pca_' + str(i)]       = None


    max_elts_elec //= comm.size
    nb_elts       //= comm.size
    few_elts       = False
    borders, nb_chunks, chunk_len, last_chunk_len = io.analyze_data(params, chunk_size)

    if numpy.all(smart_search == 0):
        gpass = 1
    else:
        gpass = 0

    ## We will perform several passes to enhance the quality of the clustering
    while gpass < (nb_repeats + 1):

        comm.Barrier()

        if gpass == 1:
            sdata = all_gather_array(smart_search[numpy.arange(comm.rank, N_e, comm.size)], comm, 0)

        if comm.rank == 0:
            if gpass == 0:
                io.print_and_log(["Searching random spikes to estimate distances..."], 'default', params)
            elif gpass == 1:
                if not numpy.all(sdata > 0):
                    lines = ["Smart Search disabled on %d electrodes" %(numpy.sum(sdata == 0))]
                    io.print_and_log(lines, 'info', params)
                if numpy.any(sdata > 0):
                    io.print_and_log(["Smart Search of good spikes for the clustering (%d/%d)..." %(gpass, nb_repeats)], 'default', params)
                else:
                    io.print_and_log(["Searching random spikes for the clustering (%d/%d) (no smart search)..." %(gpass, nb_repeats)], 'default', params)
            else:
                io.print_and_log(["Searching random spikes to refine the clustering (%d/%d)..." %(gpass, nb_repeats)], 'default', params)

        for i in xrange(N_e):
            n_neighb                     = len(edges[nodes[i]])
            result['tmp_'  + str(i)]     = numpy.zeros((0, basis_proj.shape[1] * n_neighb), dtype=numpy.float32)
            # If not the first pass, we sync all the detected times among nodes and give all nodes the w/pca
            result['all_times_' + str(i)] = all_gather_array(result['loc_times_' + str(i)], comm, dtype='int32')
            if gpass == 1:
                result['pca_'  + str(i)] = comm.bcast(result['pca_' + str(i)], root=numpy.mod(i, comm.size))
                result['data_' + str(i)] = numpy.zeros((0, basis_proj.shape[1] * n_neighb), dtype=numpy.float32)
                if numpy.any(smart_search > 0):
                    result['sub_' + str(i)] = numpy.zeros((0, result['pca_' + str(i)].shape[1]), dtype=numpy.float32)

        # I guess this is more relevant, to take signals from all over the recordings
        numpy.random.seed(gpass)
        all_chunks = numpy.random.permutation(numpy.arange(nb_chunks))
        rejected   = 0
        elt_count  = 0

        ## This is not easy to read, but during the smart search pass, we need to loop over all chunks, and every nodes should
        ## search spikes for a subset of electrodes, to avoid too many communications.
        if gpass <= 1:
            chunks_to_load     = all_chunks
            nb_elecs           = numpy.sum(comm.rank == numpy.mod(numpy.arange(N_e), comm.size))
            loop_max_elts_elec = params.getint('clustering', 'max_elts')
            loop_nb_elts       = int(params.getfloat('clustering', 'nb_elts') * nb_elecs * loop_max_elts_elec)
        else:
            chunks_to_load     = all_chunks[numpy.arange(comm.rank, nb_chunks, comm.size)]
            loop_max_elts_elec = max_elts_elec
            loop_nb_elts       = nb_elts

        if comm.rank == 0:
            pbar = get_progressbar(loop_nb_elts)

        comm.Barrier()
        ## Random selection of spikes

        for gcount, gidx in enumerate(chunks_to_load):

            if (elt_count < loop_nb_elts):
                #print "Node", comm.rank, "is analyzing chunk", gidx, "/", nb_chunks, " ..."
                local_chunk, local_shape = io.load_chunk(params, gidx, chunk_len, chunk_size, nodes=nodes)
                if do_spatial_whitening:
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
                
                if gpass > 1:
                    search_from = numpy.arange(N_e)
                else:
                    search_from = numpy.arange(comm.rank, N_e, comm.size)    

                for i in search_from:
                    peaktimes     = algo.detect_peaks(local_chunk[:, i], thresholds[i], valley=True, mpd=dist_peaks)
                    if skip_artefact:
                        values    = local_chunk[peaktimes, i]
                        idx       = numpy.where(values >= -10*thresholds[i])[0]
                        peaktimes = peaktimes[idx]
                    all_peaktimes = numpy.concatenate((all_peaktimes, peaktimes))
                    all_minimas   = numpy.concatenate((all_minimas, i*numpy.ones(len(peaktimes), dtype=numpy.int32)))

                #print "Removing the useless borders..."
                if alignment:
                    local_borders = (2*template_shift, local_shape - 2*template_shift)
                else:
                    local_borders = (template_shift, local_shape - template_shift)
                idx             = (all_peaktimes >= local_borders[0]) & (all_peaktimes < local_borders[1])
                all_peaktimes   = all_peaktimes[idx]
                all_minimas     = all_minimas[idx]

                local_peaktimes = numpy.unique(all_peaktimes)
                local_offset    = gidx*chunk_size

                if len(local_peaktimes) > 0:

                    diff_times      = local_peaktimes[-1]-local_peaktimes[0]
                    all_times       = numpy.zeros((N_e, diff_times+1), dtype=numpy.bool)
                    min_times       = numpy.maximum(local_peaktimes - local_peaktimes[0] - safety_time, 0)
                    max_times       = numpy.minimum(local_peaktimes - local_peaktimes[0] + safety_time + 1, diff_times)

                    n_times         = len(local_peaktimes)
                    argmax_peak     = numpy.random.permutation(numpy.arange(n_times))
                    all_idx         = local_peaktimes[argmax_peak]

                    if gpass > 1:
                        for elec in xrange(N_e):
                            subset  = result['all_times_' + str(elec)] - local_offset
                            peaks   = subset[numpy.where((subset >= 0) & (subset < (local_shape)))[0]]
                            inter   = numpy.in1d(local_peaktimes, peaks)
                            indices = inv_nodes[edges[nodes[elec]]]
                            remove  = numpy.where(inter == True)[0]
                            for t in remove:
                                if safety_space:
                                    all_times[indices, min_times[t]:max_times[t]] = True
                                else:
                                    all_times[elec, min_times[t]:max_times[t]] = True

                    #print "Selection of the peaks with spatio-temporal masks..."
                    for midx, peak in zip(argmax_peak, all_idx):

                        if elt_count == loop_nb_elts:
                            break

                        elec = numpy.argmin(local_chunk[peak])
                        
                        if ((gpass > 1) or (numpy.mod(elec, comm.size) == comm.rank)):

                            indices = inv_nodes[edges[nodes[elec]]]

                            if safety_space:
                                myslice = all_times[indices, min_times[midx]:max_times[midx]]
                            else:
                                myslice = all_times[elec, min_times[midx]:max_times[midx]]

                            is_local_min = elec in all_minimas[all_peaktimes == peak]

                            if is_local_min and not myslice.any():

                                to_accept  = False

                                if gpass == 1:
                                    to_update = result['data_' + str(elec)]
                                else:
                                    to_update = result['tmp_' + str(elec)]

                                if len(to_update) < loop_max_elts_elec:
                                    if alignment:
                                        idx   = numpy.where(indices == elec)[0]
                                        zdata = local_chunk[peak-2*template_shift:peak+2*template_shift+1, indices]
                                        ydata = numpy.arange(len(indices))
                                        f     = scipy.interpolate.RectBivariateSpline(xdata, ydata, zdata, s=0)
                                        smoothed = smooth(f(cdata, idx)[:, 0], template_shift)
                                        rmin     = (numpy.argmin(smoothed) - len(cdata)//2.)/5.
                                        ddata    = numpy.linspace(rmin-template_shift, rmin+template_shift, N_t)
                                        sub_mat  = f(ddata, ydata).astype(numpy.float32)
                                    else:
                                        sub_mat = local_chunk[peak-template_shift:peak+template_shift+1, indices]

                                    sub_mat    = numpy.dot(basis_rec, sub_mat)
                                    nx, ny     = sub_mat.shape
                                    sub_mat    = sub_mat.reshape((1, nx * ny))

                                    if gpass == 0:
                                        to_accept  = True
                                        result['tmp_' + str(elec)] = numpy.vstack((result['tmp_' + str(elec)], sub_mat))
                                    elif gpass == 1:
                                        if smart_search[elec] > 0:
                                            sub_sub_mat = numpy.dot(sub_mat, result['pca_' + str(elec)])
                                            if len(result['data_' + str(elec)]) == 0:
                                                to_accept = True
                                            else:
                                                dist = algo.distancematrix(sub_sub_mat, result['w_' + str(elec)], result['sub_' + str(elec)])
                                                if numpy.min(dist) >= smart_search[elec]:
                                                    to_accept = True
                                                else:
                                                    rejected += 1
                                        else:
                                            to_accept = True
                                        if to_accept:

                                            result['data_' + str(elec)] = numpy.vstack((result['data_' + str(elec)], sub_mat))
                                            if smart_search[elec] > 0:
                                                result['sub_' + str(elec)] = numpy.vstack((result['sub_' + str(elec)], sub_sub_mat))
                                    else:
                                        to_accept  = True
                                        result['tmp_' + str(elec)] = numpy.vstack((result['tmp_' + str(elec)], sub_mat))

                                if to_accept:
                                    elt_count += 1
                                    if gpass > 0:
                                        to_add = numpy.array([peak + local_offset], dtype=numpy.int32)
                                        result['loc_times_' + str(elec)] = numpy.concatenate((result['loc_times_' + str(elec)], to_add))
                                    if safety_space:
                                        all_times[indices, min_times[midx]:max_times[midx]] = True
                                    else:
                                        all_times[elec, min_times[midx]:max_times[midx]] = True

                if comm.rank == 0:
                    pbar.update(elt_count)

            if comm.rank == 0:
                if (elt_count < (gcount+1)*loop_nb_elts//len(chunks_to_load)):
                    pbar.update((gcount+1)*loop_nb_elts//len(chunks_to_load))

        if comm.rank == 0:
            pbar.finish()

        comm.Barrier()

        io.print_and_log(['Node %d has collected %d spikes and rejected %d spikes' % (comm.rank, elt_count, rejected)], 'debug', params)
        gdata       = all_gather_array(numpy.array([elt_count], dtype=numpy.float32), comm, 0)
        gdata2      = gather_array(numpy.array([rejected], dtype=numpy.float32), comm, 0)
        nb_elements = int(numpy.sum(gdata))
        nb_rejected = int(numpy.sum(gdata2))
        nb_total    = int(nb_elts*comm.size)

        if nb_elements == 0:
            gpass = nb_repeats

        if comm.rank == 0:
            if gpass != 1:
                io.print_and_log(["We found %d spikes over %d requested" %(nb_elements, nb_total)], 'default', params)
                if nb_elements == 0:
                    io.print_and_log(["No more isolated spikes in the recording, stop searching"], 'info', params)
            else:
                io.print_and_log(["We found %d spikes over %d requested (%d rejected)" %(nb_elements, nb_total, nb_rejected)], 'default', params)
                if nb_elements < 0.2*nb_total:
                    few_elts = True

        #CLUSTERING: once we have been through enough chunks (we don't need all of them), we run a clustering for each electrode.
        #print "Clustering the data..."
        local_nb_clusters = 0
        local_hits        = 0
        local_mergings    = 0
        cluster_results   = {}

        if gpass > 1:
            for ielec in xrange(N_e):
                result['tmp_' + str(ielec)] = gather_array(result['tmp_' + str(ielec)], comm, numpy.mod(ielec, comm.size), 1)
        elif gpass == 1:
            for ielec in xrange(comm.rank, N_e, comm.size):
                result['times_' + str(ielec)] = numpy.copy(result['loc_times_' + str(ielec)])

        if comm.rank == 0:
            if gpass == 0:
                io.print_and_log(["Estimating the distances..."], 'default', params)
            elif gpass == 1:
                io.print_and_log(["Computing density estimations..."], 'default', params)
            else:
                io.print_and_log(["Refining density estimations..."], 'default', params)
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)

        if gpass == 1:
            dist_file = tempfile.NamedTemporaryFile()
            tmp_file  = os.path.join(tmp_path_loc, os.path.basename(dist_file.name)) + '.hdf5'
            dist_file.close()
            result['dist_file'] = tmp_file
            tmp_h5py  = h5py.File(result['dist_file'], 'w', libver='latest')
            io.print_and_log(["Node %d will use temp file %s" %(comm.rank, tmp_file)], 'debug', params)
        elif gpass > 1:
            tmp_h5py  = h5py.File(result['dist_file'], 'r', libver='latest')

        for ielec in xrange(comm.rank, N_e, comm.size):
            cluster_results[ielec] = {}

            if gpass == 0:
                if len(result['tmp_' + str(ielec)]) > 1:
                    pca                          = PCA(sub_output_dim)
                    result['tmp_' + str(ielec)]  = pca.fit_transform(result['tmp_' + str(ielec)].astype(numpy.double)).astype(numpy.float32)
                    result['w_' + str(ielec)]    = pca.explained_variance_/pca.explained_variance_.sum()
                    result['pca_' + str(ielec)]  = pca.components_.T.astype(numpy.float32)
                    rho, dist = algo.rho_estimation(result['tmp_' + str(ielec)], weight=result['w_' + str(ielec)], compute_rho=False)
                    target                       = params.getfloat('clustering', 'smart_search')
                    bounds                       = [0.75*target, 1.25*target]
                    smart_search[ielec]          = algo.autoselect_dc(dist, bounds=bounds)
                else:
                    n_neighb                     = len(edges[nodes[ielec]])
                    dimension                    = basis_proj.shape[1] * n_neighb
                    result['w_' + str(ielec)]    = numpy.ones(dimension, dtype=numpy.float64)/dimension
                    result['pca_' + str(ielec)]  = numpy.identity(dimension, dtype=numpy.float32)
                    smart_search[ielec]          = 0

                smart_search[ielec] *= int(len(result['tmp_' + str(ielec)]) >= params.getfloat('clustering', 'nb_elts')*loop_max_elts_elec)
                if smart_search[ielec] > 0:
                    io.print_and_log(['Smart search is actived on channel %d' % ielec], 'debug', params)

            elif gpass == 1:
                if len(result['data_' + str(ielec)]) > 1:

                    if result['pca_' + str(ielec)] is None:
                        pca                          = PCA(sub_output_dim)
                        data                         = pca.fit_transform(result['data_' + str(ielec)].astype(numpy.double)).astype(numpy.float32)
                        result['w_' + str(ielec)]    = pca.explained_variance_/pca.explained_variance_.sum()
                        result['pca_' + str(ielec)]  = pca.components_.T.astype(numpy.float32)
                        
                    if smart_search[ielec] == 0:
                        result['sub_' + str(ielec)] = numpy.dot(result['data_' + str(ielec)], result['pca_' + str(ielec)])

                    rho, dist = algo.rho_estimation(result['sub_' + str(ielec)], weight=result['w_' + str(ielec)], compute_rho=True, mratio=m_ratio)
                    result['norm_' + str(ielec)] = int(m_ratio*(len(result['data_' + str(ielec)]) - 1))
                    result['rho_'  + str(ielec)] = rho
                    tmp_h5py.create_dataset('dist_' + str(ielec), data=dist, chunks=True)
                    del dist
                else:
                    if result['pca_' + str(ielec)] is None:
                        n_neighb                    = len(edges[nodes[ielec]])
                        dimension                   = basis_proj.shape[1] * n_neighb
                        result['w_' + str(ielec)]   = numpy.ones(dimension, dtype=numpy.float64)/dimension
                        result['pca_' + str(ielec)] = numpy.identity(dimension, dtype=numpy.float32)
                    result['rho_'  + str(ielec)] = numpy.zeros(len(result['data_' + str(ielec)]), dtype=numpy.float64)
                    result['norm_' + str(ielec)] = 1
                    dist                         = numpy.zeros(0, dtype=numpy.float64)

                    if smart_search[ielec] == 0:
                        result['sub_' + str(ielec)] = numpy.dot(result['data_' + str(ielec)], result['pca_' + str(ielec)])
            else:
                if len(result['tmp_' + str(ielec)]) > 1:
                    data  = numpy.dot(result['tmp_' + str(ielec)], result['pca_' + str(ielec)])
                    rho, dist = algo.rho_estimation(result['sub_' + str(ielec)], weight=result['w_' + str(ielec)], update=data, mratio=m_ratio)
                    result['rho_'  + str(ielec)] += rho
                    result['norm_' + str(ielec)] += int(m_ratio*len(result['tmp_' + str(ielec)]))

            if gpass == nb_repeats:
                result.pop('tmp_' + str(ielec))
                n_data  = len(result['data_' + str(ielec)])
                n_min   = numpy.maximum(50, int(nclus_min*n_data))
                if (n_data > 1):
                    #tmp_file = os.path.join(tmp_path_loc, os.path.basename(result['dist_' + str(ielec)].name))
                    dist     = tmp_h5py.get('dist_' + str(ielec))[:]
                    result['rho_' + str(ielec)]  = -result['rho_' + str(ielec)] + result['rho_' + str(ielec)].max() 
                    result['rho_' + str(ielec)] /= max(1, result['norm_' + str(ielec)])

                    cluster_results[ielec]['groups'], r, d, c = algo.clustering(result['rho_' + str(ielec)], dist,
                                                                              m_ratio,
                                                                              n_min=n_min,
                                                                              max_clusters=max_clusters)

                    # Now we perform a merging step, for clusters that look too similar
                    data = result['sub_' + str(ielec)]
                    cluster_results[ielec]['groups'], merged = algo.merging(cluster_results[ielec]['groups'],
                                                                    sim_same_elec,
                                                                    data)


                    if make_plots not in ['None', '']:
                        save     = [plot_path, '%d.%s' %(ielec, make_plots)]
                        injected = None
                        if test_clusters:
                            injected = numpy.zeros(len(result['data_' + str(ielec)]), dtype=numpy.bool)
                            key = 'spikes_' + str(ielec)
                            thresh = 2
                            if injected_spikes.has_key(key):
                                for icount, spike in enumerate(result['loc_times_' + str(ielec)]):
                                    idx = numpy.where(numpy.abs(spike - injected_spikes['spikes_' + str(ielec)]) < thresh)[0]
                                    if len(idx) > 0:
                                        if icount < (len(injected) - 1):
                                            injected[icount] = True

                        mask = numpy.where(cluster_results[ielec]['groups'] > -1)[0]
                        sel  = numpy.unique(cluster_results[ielec]['groups'][mask])
                        data = numpy.dot(result['data_' + str(ielec)], result['pca_' + str(ielec)])
                        plot.view_clusters(data, r, d, c[:max_clusters],
                                           cluster_results[ielec]['groups'], injected=injected,
                                           save=save)

                    result.pop('loc_times_' + str(ielec))
                    result.pop('all_times_' + str(ielec))
                    result.pop('rho_' + str(ielec))
                    result.pop('norm_' + str(ielec))
                    result['debug_' + str(ielec)]       = numpy.array([r, d], dtype=numpy.float32)
                    mask                                = numpy.where(cluster_results[ielec]['groups'] > -1)[0]
                    cluster_results[ielec]['n_clus']    = len(numpy.unique(cluster_results[ielec]['groups'][mask]))
                    n_clusters                          = []
                    result['clusters_' + str(ielec)]    = cluster_results[ielec]['groups']
                    for i in numpy.unique(cluster_results[ielec]['groups'][mask]):
                        n_clusters += [numpy.sum(cluster_results[ielec]['groups'][mask] == i)]
    
                    line = ["Node %d: %d-%d templates on channel %d from %d spikes: %s" %(comm.rank, merged[0], merged[1], ielec, n_data, str(n_clusters))]
                    io.print_and_log(line, 'default', params)
                    if (merged[0]-merged[1]) == max_clusters:
                        local_hits += 1
                    local_mergings += merged[1]
                else:
                    cluster_results[ielec]['groups'] = numpy.zeros(0, dtype=numpy.int32)
                    cluster_results[ielec]['n_clus'] = 0
                    result['clusters_' + str(ielec)] = numpy.zeros(0, dtype=numpy.int32)
                    result['debug_'    + str(ielec)] = numpy.zeros((2,0), dtype=numpy.float32)
                    line = ["Node %d: not enough spikes on channel %d" %(comm.rank, ielec)]
                    io.print_and_log(line, 'default', params)

                local_nb_clusters += cluster_results[ielec]['n_clus']

        if gpass >= 1:
            tmp_h5py.close()
        gpass += 1
        
    #os.remove(result['dist_file'])

    comm.Barrier()

    gdata      = gather_array(numpy.array([local_hits], dtype=numpy.float32), comm, 0)
    gdata2     = gather_array(numpy.array([local_mergings], dtype=numpy.float32), comm, 0)
    gdata3     = gather_array(numpy.array([local_nb_clusters], dtype=numpy.float32), comm, 0)

    if comm.rank == 0:
        total_hits        = int(numpy.sum(gdata))
        total_mergings    = int(numpy.sum(gdata2))
        total_nb_clusters = int(numpy.sum(gdata3))
        lines = ["Number of clusters found : %d" %total_nb_clusters,
                 "Number of local merges   : %d" %total_mergings]
        if few_elts:
            lines += ["Not enough spikes gathered: -decrease smart_search?"]
            lines += ["                            -put safety_space=False?"]
        if total_hits > 0:
            lines += ["%d electrodes has %d clusters: -increase max_clusters?" %(total_hits, max_clusters)]
            lines += ["                              -increase sim_same_elec?"]
        io.print_and_log(lines, 'info', params)

        io.print_and_log(["Estimating the templates with the %s procedure ..." %extraction], 'default', params)

    if extraction == 'quadratic':

        total_nb_clusters = int(comm.bcast(numpy.array([int(numpy.sum(gdata3))], dtype=numpy.float32), root=0)[0])
        offsets    = numpy.zeros(comm.size, dtype=numpy.int32)
        for i in xrange(comm.size-1):
            offsets[i+1] = comm.bcast(numpy.array([local_nb_clusters], dtype=numpy.float32), root=i)
        node_pad   = numpy.sum(offsets[:comm.rank+1])        

        if parallel_hdf5:
            hfile      = h5py.File(file_out_suff + '.templates.hdf5', 'w', driver='mpio', comm=comm, libver='latest')
            norms      = hfile.create_dataset('norms', shape=(2*total_nb_clusters, ), dtype=numpy.float32, chunks=True)
            electrodes = hfile.create_dataset('electrodes', shape=(total_nb_clusters, ), dtype=numpy.int32, chunks=True)
            amps_lims  = hfile.create_dataset('limits', shape=(total_nb_clusters, 2), dtype=numpy.float32, chunks=True)
            g_count    = node_pad
            g_offset   = total_nb_clusters
        else:
            hfile      = h5py.File(file_out_suff + '.templates-%d.hdf5' %comm.rank, 'w', libver='latest')
            electrodes = hfile.create_dataset('electrodes', shape=(local_nb_clusters, ), dtype=numpy.int32, chunks=True)
            norms      = hfile.create_dataset('norms', shape=(2*local_nb_clusters, ), dtype=numpy.float32, chunks=True)
            amps_lims  = hfile.create_dataset('limits', shape=(local_nb_clusters, 2), dtype=numpy.float32, chunks=True)
            g_count    = 0
            g_offset   = local_nb_clusters
    
        temp_x     = numpy.zeros(0, dtype=numpy.int32)
        temp_y     = numpy.zeros(0, dtype=numpy.int32)
        temp_data  = numpy.zeros(0, dtype=numpy.float32)
        
        comm.Barrier()
        cfile           = h5py.File(file_out_suff + '.clusters-%d.hdf5' %comm.rank, 'w', libver='latest')
        count_templates = node_pad

        for ielec in range(comm.rank, N_e, comm.size):
            io.write_datasets(cfile, to_write, result, ielec)
        cfile.close()
        comm.Barrier()
        
        if comm.rank == 0:
            rs         = [h5py.File(file_out_suff + '.clusters-%d.hdf5' %i, 'r', libver='latest') for i in xrange(comm.size)]
            cfile      = h5py.File(file_out_suff + '.clusters.hdf5', 'w', libver='latest')
            for i in xrange(comm.size):
                for j in range(i, N_e, comm.size):
                    io.write_datasets(cfile, to_write, rs[i], j)
                rs[i].close()
                os.remove(file_out_suff + '.clusters-%d.hdf5' %i)
            cfile.close()
        
        comm.Barrier()

        callfile   = h5py.File(file_out_suff + '.clusters.hdf5', 'r', libver='latest')

        def cross_corr(spike_1, spike_2):
            x1, x2 = spike_1.min(), spike_2.min()
            y1, y2 = spike_1.max(), spike_2.max()
            x_cc   = numpy.zeros(N_t, dtype=numpy.int32)
            if ((max(y1, y2) - min(x1, x2)) <= (y1 - x1) + (y2 - x2) + N_t):
                for d in xrange(N_t):
                    x_cc[d] += len(numpy.intersect1d(spike_1, spike_2 + d - template_shift, assume_unique=True))
            return x_cc

        if comm.rank == 0:
            pbar = get_progressbar(len(numpy.arange(comm.rank, N_e, comm.size)))

        x, y       = numpy.mgrid[0:N_t, 0:N_t]
        x          = x.flatten()
        y          = y.flatten()
        cdic       = {}
        all_labels = {}
        all_times  = {}
        all_stas   = {}

        #import time

        for count, ielec in enumerate(range(comm.rank, N_e, comm.size)):

            n_neighb = inv_nodes[edges[nodes[ielec]]]
            elecs    = numpy.zeros(0, dtype=numpy.int32)
            labels   = numpy.zeros(0, dtype=numpy.int32)
            stas     = numpy.zeros((0, N_t), dtype=numpy.float32)

            start = time.time()
            for i in n_neighb:
                if not all_labels.has_key(i):
                    all_labels[i] = callfile.get('clusters_%d' %i)[:]
                    all_times[i]  = callfile.get('times_%d' %i)[:]
                mask      = numpy.where(all_labels[i] > -1)[0]
                labels_i  = all_labels[i][mask]
                unique_i  = numpy.unique(labels_i)
                if len(unique_i) > 0:
                    times_i = all_times[i][mask]
                    elecs   = numpy.concatenate((elecs, i*numpy.ones(len(unique_i))))
                    labels  = numpy.concatenate((labels, unique_i))
                    indices = inv_nodes[edges[nodes[i]]]
                    myfilt  = numpy.mod(indices, comm.size) == comm.rank
                    locind  = indices[myfilt]
                    if not all_stas.has_key(i):
                        all_stas[i] = io.get_stas(params, times_i, labels_i, i, indices, nodes=nodes, all_labels=True)
                        all_stas[i] = all_stas[i][:, myfilt, :]

                    subidx  = numpy.where(locind == ielec)[0]                    
                    a, b, c = all_stas[i][:, subidx, :].shape
                    data    = all_stas[i][:, subidx, :].reshape(a, c)
                    stas    = numpy.vstack((stas, data))
            #print "1", time.time() - start

            data = numpy.zeros(0, dtype=numpy.float32)
            row  = numpy.zeros(0, dtype=numpy.int32)
            col  = numpy.zeros(0, dtype=numpy.int32)

            start = time.time()
            for ci in xrange(len(elecs)):
                i        = elecs[ci]
                li       = labels[ci]
                mask_i   = all_labels[i] == li
                spikes_i = all_times[i][mask_i]

                for cj in xrange(ci, len(elecs)):
                    j        = elecs[cj]
                    lj       = labels[cj]
                    mask_j   = all_labels[j] == lj
                    spikes_j = all_times[j][mask_j]

                    if cdic.has_key((i,j,li,lj)):
                        data_i   = cdic[i, j, li, lj][0]
                        data_j   = cdic[i, j, li, lj][1]
                    elif cdic.has_key((j,i,lj,li)):
                        data_i   = cdic[j, i, lj, li][1]
                        data_j   = cdic[j, i, lj, li][0]
                    else:
                        data_i   = cross_corr(spikes_i-N_t//2, spikes_j)
                        data_j   = cross_corr(spikes_i, spikes_j-N_t//2)[::-1]
                        cdic[i,j,li,lj] = [data_i, data_j]
                    
                    if (numpy.any(data_i != 0) or numpy.any(data_j != 0)):
                        new_data = scipy.linalg.toeplitz(data_j, data_i).flatten()
                        idx      = new_data.nonzero()
                        data     = numpy.concatenate((data, new_data[idx]))
                        row      = numpy.concatenate((row, ci*N_t + x[idx]))
                        col      = numpy.concatenate((col, cj*N_t + y[idx]))

            autocorr = scipy.sparse.bsr_matrix((data, (row, col)), shape=(len(elecs)*N_t, len(elecs)*N_t), blocksize=(N_t, N_t), dtype=numpy.float32)
            autocorr = autocorr + autocorr.T - scipy.sparse.diags(autocorr.diagonal(), 0)
            stas     = stas.flatten()
            #print "2", time.time() - start
            
            #print "Optimization for electrode", ielec
            #from sklearn.linear_model import Lasso
            #optimizer        = Lasso(alpha=0.01)
            #local_waveforms  = optimizer.fit(autocorr, stas.astype(numpy.double)).coef_.astype(numpy.float32)
            local_waveforms = scipy.sparse.linalg.minres(autocorr, stas)[0]
            local_waveforms = local_waveforms.reshape(len(elecs), N_t)

            tmp_file = os.path.join(tmp_path_loc, 'tmp_%d.hdf5' %ielec)
            tmpdata  = h5py.File(tmp_file, 'w', libver='latest')
            output   = tmpdata.create_dataset('waveforms', data=local_waveforms)
            limits   = tmpdata.create_dataset('limits', data=elecs)
            tmpdata.close()

            if comm.rank == 0:
                pbar.update(count)

        if comm.rank == 0:
            pbar.finish()

        comm.Barrier()
        callfile.close()

        if comm.rank == 0:
            pbar = get_progressbar(local_nb_clusters)

        for ielec in range(comm.rank, N_e, comm.size):
            
            n_data   = len(result['data_' + str(ielec)])
            n_neighb = len(edges[nodes[ielec]])
            data     = result['data_' + str(ielec)].reshape(n_data, basis_proj.shape[1], n_neighb)
            mask     = numpy.where(cluster_results[ielec]['groups'] > -1)[0]
            indices  = inv_nodes[edges[nodes[ielec]]]
            loc_pad  = g_count
            for xcount, group in enumerate(numpy.unique(cluster_results[ielec]['groups'][mask])):
            
                electrodes[g_count] = ielec
                tmp_templates = numpy.zeros((len(indices), N_t), dtype=numpy.float32)
                myslice       = numpy.where(cluster_results[ielec]['groups'] == group)[0]
                for count, i in enumerate(indices):
                    pfile = h5py.File(os.path.join(tmp_path_loc, 'tmp_%d.hdf5' %i), 'r', libver='latest')
                    lmask = pfile.get('limits')[:] == ielec 
                    tmp_templates[count] = pfile.get('waveforms')[lmask, :][xcount]
                    pfile.close()

                tmpidx           = divmod(tmp_templates.argmin(), tmp_templates.shape[1])
                shift            = template_shift - tmpidx[1]
                templates        = numpy.zeros((N_e, N_t), dtype=numpy.float32)
                if shift > 0:
                    templates[indices, shift:] = tmp_templates[:, :-shift]
                elif shift < 0:
                    templates[indices, :shift] = tmp_templates[:, -shift:]
                else:
                    templates[indices, :] = tmp_templates

                slice_temp = templates[indices]
                templates  = templates.flatten()
                dx         = templates.nonzero()[0].astype(numpy.int32)

                temp_x     = numpy.concatenate((temp_x, dx))
                temp_y     = numpy.concatenate((temp_y, count_templates*numpy.ones(len(dx), dtype=numpy.int32)))
                temp_data  = numpy.concatenate((temp_data, templates[dx]))

                norms[g_count] = numpy.sqrt(numpy.sum(templates.flatten()**2)/(N_e*N_t))

                amplitudes, ortho = io.get_amplitudes(params, result['times_' + str(ielec)][myslice], ielec, indices, slice_temp, nodes)
                variations         = 5*numpy.median(numpy.abs(amplitudes - numpy.median(amplitudes)))
                physical_limit     = noise_thr*(-thresholds[indices[tmpidx[0]]])/tmp_templates.min()
                amp_min            = min(0.8, max([physical_limit, numpy.median(amplitudes) - variations, amp_limits[0]]))
                amp_max            = max(1.2, min([amp_limits[1], numpy.median(amplitudes) + variations, amp_limits[1]]))
                amps_lims[g_count] = [amp_min, amp_max]

                offset        = total_nb_clusters + count_templates
                sub_templates = numpy.zeros((N_e, N_t), dtype=numpy.float32)
                sub_templates[indices, :] = ortho
                sub_templates = sub_templates.flatten()
                dx            = sub_templates.nonzero()[0].astype(numpy.int32)

                temp_x     = numpy.concatenate((temp_x, dx))
                temp_y     = numpy.concatenate((temp_y, offset*numpy.ones(len(dx), dtype=numpy.int32)))
                temp_data  = numpy.concatenate((temp_data, sub_templates[dx]))

                norms[g_count + g_offset] = numpy.sqrt(numpy.sum(sub_templates.flatten()**2)/(N_e*N_t))

                count_templates += 1
                g_count         += 1

            if make_plots not in ['None', '']:
                if n_data > 1:
                    save     = [plot_path, '%d.%s' %(ielec, make_plots)]
                    idx      = numpy.where(indices == ielec)[0][0]
                    sub_data = data[:,:,idx]
                    nb_temp  = cluster_results[ielec]['n_clus']
                    vidx     = numpy.where((temp_y >= loc_pad) & (temp_y < loc_pad+nb_temp))[0] 
                    sub_tmp  = scipy.sparse.csr_matrix((temp_data[vidx], (temp_x[vidx], temp_y[vidx]-loc_pad)), shape=(N_e*N_t, nb_temp))
                    sub_tmp  = sub_tmp[numpy.arange(ielec*N_t, (ielec+1)*N_t), :]
                    sub_tmp  = sub_tmp.toarray().reshape(N_t, nb_temp)
                    plot.view_waveforms_clusters(numpy.dot(sub_data, basis_rec), cluster_results[ielec]['groups'],
                        thresholds[ielec], sub_tmp,
                        amps_lims[loc_pad:loc_pad+nb_temp], save=save)

            if comm.rank == 0:
                pbar.update(count_templates)

        if comm.rank == 0:
            pbar.finish()

        #We need to gather the sparse arrays
        temp_x    = gather_array(temp_x, comm, dtype='int32')        
        temp_y    = gather_array(temp_y, comm, dtype='int32')
        temp_data = gather_array(temp_data, comm)

        if parallel_hdf5:
            if comm.rank == 0:
                cfile = h5py.File(file_out_suff + '.clusters.hdf5', 'r+', libver='latest')
                io.write_datasets(cfile, ['electrodes'], {'electrodes' : electrodes[:]}) 
                cfile.close()
            hfile.close()
        else:
            hfile.close()
            comm.Barrier()
            if comm.rank == 0:
                ts         = [h5py.File(file_out_suff + '.templates-%d.hdf5' %i, 'r', libver='latest') for i in xrange(comm.size)]
                hfile      = h5py.File(file_out_suff + '.templates.hdf5', 'w', libver='latest')
                cfile      = h5py.File(file_out_suff + '.clusters.hdf5', 'r+', libver='latest')
                electrodes = hfile.create_dataset('electrodes', shape=(total_nb_clusters, ), dtype=numpy.int32, chunks=True)
                norms      = hfile.create_dataset('norms', shape=(2*total_nb_clusters, ), dtype=numpy.float32, chunks=True)
                amplitudes = hfile.create_dataset('limits', shape=(total_nb_clusters, 2), dtype=numpy.float32, chunks=True)
                count      = 0
                for i in xrange(comm.size):
                    loc_norms   = ts[i].get('norms')
                    middle      = len(loc_norms)//2
                    norms[count:count+middle]                                     = loc_norms[:middle]
                    norms[total_nb_clusters+count:total_nb_clusters+count+middle] = loc_norms[middle:]
                    electrodes[count:count+middle] = ts[i].get('electrodes')
                    amplitudes[count:count+middle] = ts[i].get('limits')
                    count      += middle
                    os.remove(file_out_suff + '.templates-%d.hdf5' %i)
                io.write_datasets(cfile, ['electrodes'], {'electrodes' : electrodes[:]})
                hfile.close()
                cfile.close()

        if comm.rank == 0:
            hfile      = h5py.File(file_out_suff + '.templates.hdf5', 'r+', libver='latest')
            hfile.create_dataset('temp_x', data=temp_x)
            hfile.create_dataset('temp_y', data=temp_y)
            hfile.create_dataset('temp_data', data=temp_data)
            hfile.create_dataset('temp_shape', data=numpy.array([N_e * N_t, 2*total_nb_clusters], dtype=numpy.int32))
            hfile.close()

        comm.Barrier()
        if comm.rank == 0:
            for i in xrange(N_e):
                os.remove(os.path.join(tmp_path_loc, 'tmp_%d.hdf5' %i))

    elif extraction in ['median-raw', 'median-pca', 'mean-raw', 'mean-pca']:

        total_nb_clusters = int(comm.bcast(numpy.array([int(numpy.sum(gdata3))], dtype=numpy.float32), root=0)[0])
        offsets    = numpy.zeros(comm.size, dtype=numpy.int32)
        for i in xrange(comm.size-1):
            offsets[i+1] = comm.bcast(numpy.array([local_nb_clusters], dtype=numpy.float32), root=i)
        node_pad   = numpy.sum(offsets[:comm.rank+1])        

        if parallel_hdf5:
            hfile      = h5py.File(file_out_suff + '.templates.hdf5', 'w', driver='mpio', comm=comm, libver='latest')
            norms      = hfile.create_dataset('norms', shape=(2*total_nb_clusters, ), dtype=numpy.float32, chunks=True)
            electrodes = hfile.create_dataset('electrodes', shape=(total_nb_clusters, ), dtype=numpy.int32, chunks=True)
            amps_lims  = hfile.create_dataset('limits', shape=(total_nb_clusters, 2), dtype=numpy.float32, chunks=True)
            g_count    = node_pad
            g_offset   = total_nb_clusters
        else:
            hfile      = h5py.File(file_out_suff + '.templates-%d.hdf5' %comm.rank, 'w', libver='latest')
            electrodes = hfile.create_dataset('electrodes', shape=(local_nb_clusters, ), dtype=numpy.int32, chunks=True)
            norms      = hfile.create_dataset('norms', shape=(2*local_nb_clusters, ), dtype=numpy.float32, chunks=True)
            amps_lims  = hfile.create_dataset('limits', shape=(local_nb_clusters, 2), dtype=numpy.float32, chunks=True)
            g_count    = 0
            g_offset   = local_nb_clusters
    
        temp_x     = numpy.zeros(0, dtype=numpy.int32)
        temp_y     = numpy.zeros(0, dtype=numpy.int32)
        temp_data  = numpy.zeros(0, dtype=numpy.float32)

        comm.Barrier()
        cfile           = h5py.File(file_out_suff + '.clusters-%d.hdf5' %comm.rank, 'w', libver='latest')
        count_templates = node_pad


        if comm.rank == 0:
            pbar = get_progressbar(local_nb_clusters)

        for ielec in range(comm.rank, N_e, comm.size):
            #print "Dealing with cluster", ielec
            n_data   = len(result['data_' + str(ielec)])
            n_neighb = len(edges[nodes[ielec]])
            data     = result['data_' + str(ielec)].reshape(n_data, basis_proj.shape[1], n_neighb)
            mask     = numpy.where(cluster_results[ielec]['groups'] > -1)[0]
            loc_pad  = g_count
            indices  = inv_nodes[edges[nodes[ielec]]]
                    
            for group in numpy.unique(cluster_results[ielec]['groups'][mask]):
                electrodes[g_count] = ielec
                myslice          = numpy.where(cluster_results[ielec]['groups'] == group)[0]
                if extraction == 'median-pca':
                    sub_data         = data[myslice]
                    first_component  = numpy.median(sub_data, axis=0)
                    tmp_templates    = numpy.dot(first_component.T, basis_rec)
                elif extraction == 'mean-pca':
                    sub_data         = data[myslice]
                    first_component  = numpy.mean(sub_data, axis=0)
                    tmp_templates    = numpy.dot(first_component.T, basis_rec)
                elif extraction == 'median-raw':                
                    labels_i         = numpy.random.permutation(myslice)[:min(len(myslice), 1000)]
                    times_i          = result['times_' + str(ielec)][labels_i]
                    sub_data         = io.get_stas(params, times_i, labels_i, ielec, neighs=indices, nodes=nodes)
                    first_component  = numpy.median(sub_data, 0)
                    tmp_templates    = first_component
                elif extraction == 'mean-raw':                
                    labels_i         = numpy.random.permutation(myslice)[:min(len(myslice), 1000)]
                    times_i          = result['times_' + str(ielec)][labels_i]
                    sub_data         = io.get_stas(params, times_i, labels_i, ielec, neighs=indices, nodes=nodes)
                    first_component  = numpy.mean(sub_data, 0)
                    tmp_templates    = first_component

                tmpidx           = divmod(tmp_templates.argmin(), tmp_templates.shape[1])
                shift            = template_shift - tmpidx[1]
                templates        = numpy.zeros((N_e, N_t), dtype=numpy.float32)
                if shift > 0:
                    templates[indices, shift:] = tmp_templates[:, :-shift]
                elif shift < 0:
                    templates[indices, :shift] = tmp_templates[:, -shift:]
                else:
                    templates[indices, :] = tmp_templates

                templates  = templates.flatten()
                dx         = templates.nonzero()[0].astype(numpy.int32)

                temp_x     = numpy.concatenate((temp_x, dx))
                temp_y     = numpy.concatenate((temp_y, count_templates*numpy.ones(len(dx), dtype=numpy.int32)))
                temp_data  = numpy.concatenate((temp_data, templates[dx]))

                norms[g_count] = numpy.sqrt(numpy.sum(templates.flatten()**2)/(N_e*N_t))

                x, y, z          = sub_data.shape
                sub_data_flat    = sub_data.reshape(x, y*z)
                first_flat       = first_component.reshape(y*z, 1)
                amplitudes       = numpy.dot(sub_data_flat, first_flat)
                amplitudes      /= numpy.sum(first_flat**2)

                variations       = 5*numpy.median(numpy.abs(amplitudes - numpy.median(amplitudes)))
                physical_limit   = noise_thr*(-thresholds[indices[tmpidx[0]]])/tmp_templates.min()
                amp_min          = min(0.8, max([physical_limit, numpy.median(amplitudes) - variations, amp_limits[0]]))
                amp_max          = max(1.2, min([amp_limits[1], numpy.median(amplitudes) + variations, amp_limits[1]]))
                amps_lims[g_count] = [amp_min, amp_max]

                for i in xrange(x):
                    sub_data_flat[i, :] -= amplitudes[i]*first_flat[:, 0]

                if len(sub_data_flat) > 1:
                    pca              = PCA(1)
                    res_pca          = pca.fit_transform(sub_data_flat.astype(numpy.double)).astype(numpy.float32)
                    second_component = pca.components_.T.astype(numpy.float32).reshape(y, z)
                else:
                    second_component = sub_data_flat.reshape(y, z)/numpy.sum(sub_data_flat**2)

                if extraction in ['median-pca', 'mean-pca']:
                    tmp_templates = numpy.dot(second_component.T, basis_rec)
                elif extraction in ['median-raw', 'mean-raw']:
                    tmp_templates = second_component
                
                offset        = total_nb_clusters + count_templates
                sub_templates = numpy.zeros((N_e, N_t), dtype=numpy.float32)
                if shift > 0:
                    sub_templates[indices, shift:] = tmp_templates[:, :-shift]
                elif shift < 0:
                    sub_templates[indices, :shift] = tmp_templates[:, -shift:]
                else:
                    sub_templates[indices, :] = tmp_templates

                sub_templates = sub_templates.flatten()
                dx            = sub_templates.nonzero()[0].astype(numpy.int32)

                temp_x     = numpy.concatenate((temp_x, dx))
                temp_y     = numpy.concatenate((temp_y, offset*numpy.ones(len(dx), dtype=numpy.int32)))
                temp_data  = numpy.concatenate((temp_data, sub_templates[dx]))

                norms[g_count + g_offset] = numpy.sqrt(numpy.sum(sub_templates.flatten()**2)/(N_e*N_t))

                count_templates += 1
                g_count         += 1

            if make_plots not in ['None', '']:
                if n_data > 1:
                    save     = [plot_path, '%d.%s' %(ielec, make_plots)]
                    idx      = numpy.where(indices == ielec)[0][0]
                    sub_data = data[:,:,idx]
                    nb_temp  = cluster_results[ielec]['n_clus']
                    vidx     = numpy.where((temp_y >= loc_pad) & (temp_y < loc_pad+nb_temp))[0] 
                    sub_tmp  = scipy.sparse.csr_matrix((temp_data[vidx], (temp_x[vidx], temp_y[vidx]-loc_pad)), shape=(N_e*N_t, nb_temp))
                    sub_tmp  = sub_tmp[numpy.arange(ielec*N_t, (ielec+1)*N_t), :]
                    sub_tmp  = sub_tmp.toarray().reshape(N_t, nb_temp)
                    plot.view_waveforms_clusters(numpy.dot(sub_data, basis_rec), cluster_results[ielec]['groups'],
                        thresholds[ielec], sub_tmp,
                        amps_lims[loc_pad:loc_pad+nb_temp], save=save)

            io.write_datasets(cfile, to_write, result, ielec)

            if comm.rank == 0:
                pbar.update(count_templates)

        if comm.rank == 0:
            pbar.finish()

        #At the end we should have a templates variable to store.
        cfile.close()
        del result, amps_lims
        
        comm.Barrier()

        #We need to gather the sparse arrays
        temp_x    = gather_array(temp_x, comm, dtype='int32')        
        temp_y    = gather_array(temp_y, comm, dtype='int32')
        temp_data = gather_array(temp_data, comm)

        if parallel_hdf5:
            if comm.rank == 0:
                rs         = [h5py.File(file_out_suff + '.clusters-%d.hdf5' %i, 'r', libver='latest') for i in xrange(comm.size)]
                cfile      = h5py.File(file_out_suff + '.clusters.hdf5', 'w', libver='latest')
                io.write_datasets(cfile, ['electrodes'], {'electrodes' : electrodes[:]})
                for i in xrange(comm.size):
                    for j in range(i, N_e, comm.size):
                        io.write_datasets(cfile, to_write, rs[i], j)
                    rs[i].close()
                    os.remove(file_out_suff + '.clusters-%d.hdf5' %i)
                cfile.close()
            hfile.close()
        else:
            hfile.close()
            comm.Barrier()
            if comm.rank == 0:
                ts         = [h5py.File(file_out_suff + '.templates-%d.hdf5' %i, 'r', libver='latest') for i in xrange(comm.size)]
                rs         = [h5py.File(file_out_suff + '.clusters-%d.hdf5' %i, 'r', libver='latest') for i in xrange(comm.size)]
                result     = {}
                hfile      = h5py.File(file_out_suff + '.templates.hdf5', 'w', libver='latest')
                cfile      = h5py.File(file_out_suff + '.clusters.hdf5', 'w', libver='latest')
                electrodes = hfile.create_dataset('electrodes', shape=(total_nb_clusters, ), dtype=numpy.int32, chunks=True)
                norms      = hfile.create_dataset('norms', shape=(2*total_nb_clusters, ), dtype=numpy.float32, chunks=True)
                amplitudes = hfile.create_dataset('limits', shape=(total_nb_clusters, 2), dtype=numpy.float32, chunks=True)
                count      = 0
                for i in xrange(comm.size):
                    loc_norms   = ts[i].get('norms')
                    middle      = len(loc_norms)//2
                    norms[count:count+middle]                                     = loc_norms[:middle]
                    norms[total_nb_clusters+count:total_nb_clusters+count+middle] = loc_norms[middle:]
                    electrodes[count:count+middle] = ts[i].get('electrodes')
                    amplitudes[count:count+middle] = ts[i].get('limits')
                    count      += middle
                    for j in range(i, N_e, comm.size):
                        io.write_datasets(cfile, to_write, rs[i], j)
                    ts[i].close()
                    rs[i].close()
                    os.remove(file_out_suff + '.templates-%d.hdf5' %i)
                    os.remove(file_out_suff + '.clusters-%d.hdf5' %i)
                io.write_datasets(cfile, ['electrodes'], {'electrodes' : electrodes[:]})
                hfile.close()
                cfile.close()

        if comm.rank == 0:
            hfile      = h5py.File(file_out_suff + '.templates.hdf5', 'r+', libver='latest')
            hfile.create_dataset('temp_x', data=temp_x)
            hfile.create_dataset('temp_y', data=temp_y)
            hfile.create_dataset('temp_data', data=temp_data)
            hfile.create_dataset('temp_shape', data=numpy.array([N_e, N_t, 2*total_nb_clusters], dtype=numpy.int32))
            hfile.close()
    
    comm.Barrier()

    if comm.rank == 0:
        io.print_and_log(["Merging similar templates..."], 'default', params)
    
    
    merged1 = algo.merging_cc(comm, params, parallel_hdf5)
    
    comm.Barrier()

    if remove_mixture:
        if comm.rank == 0:
            io.print_and_log(["Removing mixtures..."], 'default', params)
        merged2 = algo.delete_mixtures(comm, params, parallel_hdf5)
    else:
        merged2 = [0, 0]

    if comm.rank == 0:

        io.print_and_log(["Number of global merges    : %d" %merged1[1], 
                          "Number of mixtures removed : %d" %merged2[1]], 'info', params)    
    
    
    comm.Barrier()
    io.get_overlaps(comm, params, erase=True, parallel_hdf5=parallel_hdf5)
