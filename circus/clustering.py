from .shared.utils import *
from .shared import plot


def main(filename, params, nb_cpu, nb_gpu, use_gpu):

    #################################################################
    sampling_rate  = params.getint('data', 'sampling_rate')
    N_e            = params.getint('data', 'N_e')
    N_t            = params.getint('data', 'N_t')
    N_total        = params.getint('data', 'N_total')
    dist_peaks     = params.getint('data', 'dist_peaks')
    template_shift = params.getint('data', 'template_shift')
    file_out       = params.get('data', 'file_out')
    file_out_suff  = params.get('data', 'file_out_suff')
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
    make_plots     = params.getboolean('clustering', 'make_plots')
    sim_same_elec  = params.getfloat('clustering', 'sim_same_elec')
    cc_merge       = params.getfloat('clustering', 'cc_merge')
    noise_thr      = params.getfloat('clustering', 'noise_thr')
    smart_search   = numpy.ones(N_e, dtype=numpy.float32)*params.getfloat('clustering', 'smart_search')
    test_clusters  = params.getboolean('clustering', 'test_clusters')
    tmp_limits     = params.get('fitting', 'amp_limits').replace('(', '').replace(')', '').split(',')
    amp_limits     = map(float, tmp_limits)
    elt_count      = 0
    sub_output_dim = 0.95
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)
    #################################################################

    basis_proj, basis_rec = io.load_data(params, 'basis')
    thresholds = io.load_data(params, 'thresholds')
    if do_spatial_whitening or do_temporal_whitening:
        spatial_whitening  = io.load_data(params, 'spatial_whitening')
        temporal_whitening = io.load_data(params, 'temporal_whitening')

    result   = {}

    if test_clusters:
        injected_spikes = io.load_data(params, 'injected_spikes')

    if comm.rank == 0:
        if not os.path.exists(tmp_path_loc):
            os.makedirs(tmp_path_loc)

    comm.Barrier()

    for i in xrange(N_e):
        result['loc_times_' + str(i)] = numpy.zeros(0, dtype=numpy.int32)
        result['times_' + str(i)]     = numpy.zeros(0, dtype=numpy.int32)
        result['dc_' + str(i)]        = None
        result['pca_' + str(i)]       = None


    max_elts_elec /= comm.size
    nb_elts       /= comm.size
    few_elts       = False
    borders, nb_chunks, chunk_len, last_chunk_len = io.analyze_data(params)

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
                print "Searching random spikes to estimate distances..."
            elif gpass == 1:
                if not numpy.all(sdata > 0):
                    lines = ["Smart Search disabled on %d electrodes" %(numpy.sum(sdata == 0))]
                    io.print_info(lines)
                if numpy.any(sdata > 0):
                    print "Smart Search of good spikes for the clustering (%d/%d)..." %(gpass, nb_repeats)
                else:
                    print "Searching random spikes for the clustering (%d/%d) (no smart search)..." %(gpass, nb_repeats)
            else:
                print "Searching random spikes to refine the clustering (%d/%d)..." %(gpass, nb_repeats)

        for i in xrange(N_e):
            n_neighb                     = len(edges[nodes[i]])
            result['tmp_'  + str(i)]     = numpy.zeros((0, basis_proj.shape[1] * n_neighb), dtype=numpy.float32)
            # If not the first pass, we sync all the detected times among nodes and give all nodes the w/pca
            result['all_times_' + str(i)] = all_gather_array(result['loc_times_' + str(i)], comm, dtype='int32')
            if gpass == 1:
                result['dc_'   + str(i)] = comm.bcast(result['dc_' + str(i)], root=numpy.mod(i, comm.size))
                result['pca_'  + str(i)] = comm.bcast(result['pca_' + str(i)], root=numpy.mod(i, comm.size))
                result['data_' + str(i)]  = numpy.zeros((0, basis_proj.shape[1] * n_neighb), dtype=numpy.float32)
                if numpy.any(smart_search > 0):
                    result['sub_' + str(i)] = numpy.zeros((0, result['pca_' + str(i)].shape[1]), dtype=numpy.float32)

        # I guess this is more relevant, to take signals from all over the recordings
        numpy.random.seed(gpass)
        all_chunks = numpy.random.permutation(numpy.arange(nb_chunks))
        rejected   = 0
        elt_count  = 0

        ## This is not easy to read, but during the smart search pass, we need to loop over all chunks, and every nodes should
        ## search spikes for a subset of electrodes, to avoid too many communications.
        if gpass == 1:
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
                local_chunk, local_shape = io.load_chunk(params, gidx, chunk_len, nodes=nodes)
                if do_spatial_whitening:
                    local_chunk = numpy.dot(local_chunk, spatial_whitening)
                if do_temporal_whitening:
                    for i in xrange(N_e):
                        local_chunk[:, i] = numpy.convolve(local_chunk[:, i], temporal_whitening, 'same')

                #print "Extracting the peaks..."
                all_peaktimes = []
                all_minimas   = []
                for i in xrange(N_e):
                    peaktimes      = algo.detect_peaks(local_chunk[:, i], thresholds[i], valley=True, mpd=dist_peaks).tolist()
                    all_peaktimes += peaktimes
                    all_minimas   += [i]*len(peaktimes)

                all_peaktimes      = numpy.array(all_peaktimes, dtype=numpy.int32)
                all_minimas        = numpy.array(all_minimas, dtype=numpy.int32)

                #print "Removing the useless borders..."
                local_borders   = (template_shift, local_shape - template_shift)
                idx             = (all_peaktimes >= local_borders[0]) & (all_peaktimes < local_borders[1])
                all_peaktimes   = all_peaktimes[idx]
                all_minimas     = all_minimas[idx]

                local_peaktimes = numpy.lib.arraysetops.unique(all_peaktimes)
                local_offset    = gidx*chunk_size

                if len(local_peaktimes) > 0:

                    diff_times      = local_peaktimes[-1]-local_peaktimes[0]
                    all_times       = numpy.zeros((N_e, diff_times+1), dtype=numpy.bool)
                    min_times       = numpy.maximum(local_peaktimes - local_peaktimes[0] - safety_time, 0)
                    max_times       = numpy.minimum(local_peaktimes - local_peaktimes[0] + safety_time + 1, diff_times)

                    n_times         = len(local_peaktimes)
                    argmax_peak     = numpy.random.permutation(numpy.arange(n_times))
                    best_electrode  = numpy.argmin(local_chunk[local_peaktimes[argmax_peak]], 1)

                    if gpass == 1:
                        myslice        = numpy.mod(best_electrode, comm.size) == comm.rank
                        argmax_peak    = argmax_peak[myslice]
                        best_electrode = best_electrode[myslice]
                    elif gpass > 1:
                        for elec in xrange(N_e):
                            subset  = result['all_times_' + str(elec)] - local_offset
                            peaks   = subset[numpy.where((subset >= 0) & (subset < (local_shape)))[0]]
                            inter   = numpy.lib.arraysetops.in1d(local_peaktimes, peaks)
                            indices = inv_nodes[edges[nodes[elec]]]
                            remove  = numpy.where(inter == True)[0]
                            for t in remove:
                                if safety_space:
                                    all_times[indices, min_times[t]:max_times[t]] = True
                                else:
                                    all_times[elec, min_times[t]:max_times[t]] = True

                    #print "Selection of the peaks with spatio-temporal masks..."
                    for idx, elec in zip(argmax_peak, best_electrode):

                        if elt_count == loop_nb_elts:
                            break

                        indices = inv_nodes[edges[nodes[elec]]]

                        if safety_space:
                            myslice = all_times[indices, min_times[idx]:max_times[idx]]
                        else:
                            myslice = all_times[elec, min_times[idx]:max_times[idx]]

                        peak         = local_peaktimes[idx]
                        is_local_min = elec in all_minimas[all_peaktimes == peak]

                        if is_local_min and not myslice.any():

                            to_accept  = False

                            if gpass == 1:
                                to_update = result['data_' + str(elec)]
                            else:
                                to_update = result['tmp_' + str(elec)]

                            if len(to_update) < loop_max_elts_elec:
                                sub_mat    = local_chunk[peak-template_shift:peak+template_shift+1, indices]
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
                                            dist = numpy.mean((sub_sub_mat - result['sub_' + str(elec)])**2, 1)
                                            if numpy.min(dist) >= smart_search[elec]*result['dc_' + str(elec)]:
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
                                    all_times[indices, min_times[idx]:max_times[idx]] = True
                                else:
                                    all_times[elec, min_times[idx]:max_times[idx]] = True

                if comm.rank == 0:
                    pbar.update(elt_count)

            if comm.rank == 0:
                if (elt_count < (gcount+1)*loop_nb_elts/len(chunks_to_load)):
                    pbar.update((gcount+1)*loop_nb_elts/len(chunks_to_load))

        if comm.rank == 0:
            pbar.finish()

        comm.Barrier()

        gdata       = all_gather_array(numpy.array([elt_count], dtype=numpy.float32), comm, 0)
        gdata2      = all_gather_array(numpy.array([rejected], dtype=numpy.float32), comm, 0)
        nb_elements = int(numpy.sum(gdata))
        nb_rejected = int(numpy.sum(gdata2))
        nb_total    = int(nb_elts*comm.size)

        if nb_elements == 0:
            gpass = nb_repeats

        if comm.rank == 0:
            if gpass != 1:
                print "We found", nb_elements, "spikes over", nb_total, "requested"
                if nb_elements == 0:
                    io.print_info(["No more isolated spikes in the recording, stop searching"])
            else:
                print "We found", nb_elements, "spikes over", nb_total, "requested (%d rejected)" %nb_rejected
                if nb_elements < 0.2*nb_total:
                    few_elts = True

        #CLUSTERING: once we have been through enough chunks (we don't need all of them), we run a clustering for each electrode.
        #print "Clustering the data..."
        local_nb_clusters = 0
        local_hits        = 0
        local_mergings    = 0
        cluster_results   = {}

        if gpass != 1:
            for ielec in xrange(N_e):
                result['tmp_' + str(ielec)] = gather_array(numpy.array(result['tmp_' + str(ielec)], dtype=numpy.float32), comm, numpy.mod(ielec, comm.size), 1)
        elif gpass == 1:
            for ielec in xrange(comm.rank, N_e, comm.size):
                result['times_' + str(ielec)] = numpy.copy(result['loc_times_' + str(ielec)])
                if numpy.any(smart_search > 0):
                    result.pop('sub_' + str(ielec))

        if comm.rank == 0:
            if gpass == 0:
                print "Estimating the distances..."
            elif gpass == 1:
                print "Computing density estimations..."
            else:
                print "Refining density estimations..."
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)

        for ielec in xrange(comm.rank, N_e, comm.size):
            cluster_results[ielec] = {}

            if gpass == 0:
                if len(result['tmp_' + str(ielec)]) > 1:
                    pca  = mdp.nodes.PCANode(output_dim=sub_output_dim)
                    data = pca(result['tmp_' + str(ielec)].astype(numpy.double)).astype(numpy.float32)
                    result['w_' + str(ielec)]    = pca.d/pca.d.sum()
                    result['pca_' + str(ielec)]  = pca.get_projmatrix().astype(numpy.float32)
                    result['tmp_' + str(ielec)]  = data
                    rho, dist, dc = algo.rho_estimation(result['tmp_' + str(ielec)], weight=None, compute_rho=False)
                    result['dc_' + str(ielec)]   = dc
                else:
                    n_neighb                     = len(edges[nodes[ielec]])
                    dimension                    = basis_proj.shape[1] * n_neighb
                    result['w_' + str(ielec)]    = numpy.ones(dimension, dtype=numpy.float32)/dimension
                    result['pca_' + str(ielec)]  = numpy.identity(dimension, dtype=numpy.float32)
                smart_search[ielec] *= int(len(result['tmp_' + str(ielec)]) >= 0.9*max_elts_elec*comm.size)
            elif gpass == 1:
                if len(result['data_' + str(ielec)]) > 1:
                    pca  = mdp.nodes.PCANode(output_dim=sub_output_dim)
                    data = pca(result['data_' + str(ielec)].astype(numpy.double)).astype(numpy.float32)
                    result['w_' + str(ielec)]    = pca.d/pca.d.sum()
                    result['pca_' + str(ielec)]  = pca.get_projmatrix().astype(numpy.float32)
                    rho, dist, dc = algo.rho_estimation(data, weight=result['w_' + str(ielec)], compute_rho=True)
                    dist_file = tempfile.NamedTemporaryFile()
                    tmp_file  = os.path.join(tmp_path_loc, os.path.basename(dist_file.name))
                    numpy.save(tmp_file, dist)
                    result['dist_' + str(ielec)] = dist_file
                    result['norm_' + str(ielec)] = len(result['data_' + str(ielec)])
                    result['rho_'  + str(ielec)] = rho
                    result['dc_' + str(ielec)]   = dc
                    del dist
                else:
                    n_neighb                     = len(edges[nodes[ielec]])
                    dimension                    = basis_proj.shape[1] * n_neighb
                    result['w_' + str(ielec)]    = numpy.ones(dimension, dtype=numpy.float32)/dimension
                    result['pca_' + str(ielec)]  = numpy.identity(dimension, dtype=numpy.float32)
                    result['rho_'  + str(ielec)] = numpy.zeros(len(result['data_' + str(ielec)]), dtype=numpy.float32)
                    result['norm_' + str(ielec)] = 0
            else:
                if len(result['tmp_' + str(ielec)]) > 1:
                    data  = numpy.dot(result['tmp_' + str(ielec)], result['pca_' + str(ielec)])
                    sdata = numpy.dot(result['data_' + str(ielec)], result['pca_' + str(ielec)])
                    rho, dist, dc = algo.rho_estimation(sdata, dc=result['dc_' + str(ielec)], weight=result['w_' + str(ielec)], update=data)
                    result['rho_'  + str(ielec)] += rho
                    result['norm_' + str(ielec)] += len(result['tmp_' + str(ielec)])

            if gpass == nb_repeats:
                result.pop('tmp_' + str(ielec))
                n_data  = len(result['data_' + str(ielec)])
                n_min   = numpy.maximum(5, int(nclus_min*n_data))
                if (n_data > 1):
                    tmp_file = os.path.join(tmp_path_loc, os.path.basename(result['dist_' + str(ielec)].name))
                    dist     = numpy.load(tmp_file +'.npy')
                    os.remove(tmp_file + '.npy')
                    result['dist_' + str(ielec)].close()
                    result['rho_' + str(ielec)] /= result['norm_' + str(ielec)]

                    cluster_results[ielec]['groups'], r, d, c = algo.clustering(result['rho_' + str(ielec)], dist,
                                                                              result['dc_' + str(ielec)],
                                                                              smart_search=smart_search[ielec],
                                                                              n_min=n_min,
                                                                              max_clusters=max_clusters)

                    # Now we perform a merging step, for clusters that look too similar
                    data = numpy.dot(result['data_' + str(ielec)], result['pca_' + str(ielec)])
                    cluster_results[ielec]['groups'], merged = algo.merging(cluster_results[ielec]['groups'],
                                                                    sim_same_elec,
                                                                    data)

                    if make_plots:
                        save     = [plot_path, '%d' %ielec]
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
                        plot.view_clusters(numpy.sqrt(result['w_' + str(ielec)])*data, r, d, c[:max_clusters],
                                           cluster_results[ielec]['groups'], dc=result['dc_' + str(ielec)], injected=injected,
                                           save=save)

                    result.pop('loc_times_' + str(ielec))
                    result.pop('all_times_' + str(ielec))
                    result.pop('rho_' + str(ielec))
                    result.pop('norm_' + str(ielec))
                    result.pop('dc_' + str(ielec))
                    result.pop('dist_' + str(ielec))
                    result['debug_' + str(ielec)]       = numpy.array([r, d], dtype=numpy.float32)
                    mask                                = numpy.where(cluster_results[ielec]['groups'] > -1)[0]
                    cluster_results[ielec]['n_clus']    = len(numpy.unique(cluster_results[ielec]['groups'][mask]))
                    n_clusters                          = []
                    result['clusters_' + str(ielec)]    = cluster_results[ielec]['groups']
                    for i in numpy.unique(cluster_results[ielec]['groups'][mask]):
                        n_clusters += [numpy.sum(cluster_results[ielec]['groups'][mask] == i)]
                    print "Node %d:" %comm.rank, '%d-%d' %(merged[0], merged[1]), "templates on electrode", ielec, "with", n_data, "spikes:", n_clusters
                    if (merged[0]-merged[1]) == max_clusters:
                        local_hits += 1
                    local_mergings += merged[1]
                else:
                    cluster_results[ielec]['groups'] = numpy.array([], dtype=numpy.int32)
                    cluster_results[ielec]['n_clus'] = 0
                    result['clusters_' + str(ielec)] = []
                    result['debug_'    + str(ielec)] = numpy.zeros((2,0), dtype=numpy.float32)
                    print "Node %d:" %comm.rank, "not enough spikes on electrode", ielec

                local_nb_clusters += cluster_results[ielec]['n_clus']

        gpass += 1

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
        io.print_info(lines)

        print "Extracting the templates..."

    templates       = numpy.zeros((N_e, N_t, 2*local_nb_clusters), dtype=numpy.float32)
    count_templates = 0
    all_clusters    = numpy.array([cluster_results[ielec]['n_clus'] for ielec in range(comm.rank, N_e, comm.size)], dtype=numpy.int32)
    electrodes      = []
    groups          = []
    amps_lims       = []
    similars        = []
    n_datas         = []

    for ielec in range(comm.rank, N_e, comm.size):
        #print "Dealing with cluster", ielec
        n_data   = len(result['data_' + str(ielec)])
        n_neighb = len(edges[nodes[ielec]])
        data     = result['data_' + str(ielec)].reshape(n_data, basis_proj.shape[1], n_neighb)
        mask     = numpy.where(cluster_results[ielec]['groups'] > -1)[0]
        loc_pad  = count_templates
        indices  = inv_nodes[edges[nodes[ielec]]]
        for group in numpy.unique(cluster_results[ielec]['groups'][mask]):
            electrodes      += [ielec]
            groups          += [group]
            myslice          = numpy.where(cluster_results[ielec]['groups'] == group)[0]
            sub_data         = data[myslice]
            first_component  = numpy.median(sub_data, axis=0)

            tmp_templates    = numpy.dot(first_component.T, basis_rec)
            tmpidx           = numpy.where(tmp_templates == tmp_templates.min())
            temporal_shift   = template_shift - tmpidx[1][0]
            if temporal_shift > 0:
                templates[indices, temporal_shift:, count_templates] = tmp_templates[:, :-temporal_shift]
            elif temporal_shift < 0:
                templates[indices, :temporal_shift, count_templates] = tmp_templates[:, -temporal_shift:]
            else:
                templates[indices, :, count_templates] = tmp_templates

            x, y, z          = sub_data.shape
            sub_data_flat    = sub_data.reshape(x, y*z)
            first_flat       = first_component.reshape(y*z, 1)
            amplitudes       = numpy.dot(sub_data_flat, first_flat)
            amplitudes      /= numpy.sum(first_flat**2)

            variations       = 6*numpy.median(numpy.abs(amplitudes - numpy.median(amplitudes)))
            physical_limit   = noise_thr*(-thresholds[tmpidx[0][0]])/tmp_templates.min()
            amp_min          = max(physical_limit, numpy.median(amplitudes) - variations)
            amp_max          = min(amp_limits[1], numpy.median(amplitudes) + variations)
            amps_lims       += [[amp_min, amp_max]]

            for i in xrange(x):
                sub_data_flat[i, :] -= amplitudes[i]*first_flat[:, 0]

            if len(sub_data_flat) > 1:
                pca              = mdp.nodes.PCANode(output_dim=1)
                res_pca          = pca(sub_data_flat.astype(numpy.double))
                second_component = pca.get_projmatrix().reshape(y, z)
            else:
                second_component = sub_data_flat.reshape(y, z)/numpy.sum(sub_data_flat**2)

            tmp_templates        = numpy.dot(second_component.T, basis_rec)
            if temporal_shift > 0:
                templates[indices, temporal_shift:, local_nb_clusters + count_templates] = tmp_templates[:, :-temporal_shift]
            elif temporal_shift < 0:
                templates[indices, :temporal_shift, local_nb_clusters + count_templates] = tmp_templates[:, -temporal_shift:]
            else:
                templates[indices, :, local_nb_clusters + count_templates] = tmp_templates

            count_templates += 1

        if make_plots:
            if n_data > 1:
                save     = [plot_path, '%d' %ielec]
                idx      = numpy.where(indices == ielec)[0][0]
                sub_data = data[:,:,idx]
                nb_temp  = cluster_results[ielec]['n_clus']
                plot.view_waveforms_clusters(numpy.dot(sub_data, basis_rec), cluster_results[ielec]['groups'],
                    thresholds[ielec], templates[indices[idx], :, loc_pad:loc_pad+nb_temp],
                    amps_lims[loc_pad:loc_pad+nb_temp], save=save)


    result['electrodes'] = numpy.array(electrodes, dtype=numpy.int32)
    result['groups']     = numpy.array(groups, dtype=numpy.int32)
    result['amplitudes'] = numpy.array(amps_lims, dtype=numpy.float32)


    #At the end we should have a templates variable to store.
    numpy.save(file_out_suff + '.templates-%d' %comm.rank, templates)
    numpy.save(file_out_suff + '.electrodes-%d' %comm.rank, result['electrodes'])
    numpy.save(file_out_suff + '.amplitudes-%d' %comm.rank, result['amplitudes'])
    cPickle.dump(result, file(file_out_suff + '.data-%d.pic' %comm.rank, 'w'))
    comm.Barrier()

    if comm.rank == 0:

        ts         = [numpy.load(file_out_suff + '.templates-%d.npy' %i) for i in xrange(comm.size)]
        cs         = [numpy.load(file_out_suff + '.electrodes-%d.npy' %i).tolist() for i in xrange(comm.size)]
        bs         = [numpy.load(file_out_suff + '.amplitudes-%d.npy' %i).tolist() for i in xrange(comm.size)]
        rs         = [cPickle.load(file(file_out_suff + '.data-%d.pic' %i, 'r')) for i in xrange(comm.size)]
        result     = {}
        n_clusters = numpy.sum([ts[i].shape[2] for i in xrange(comm.size)])/2
        templates  = numpy.zeros((N_e, N_t, 2*n_clusters), dtype=numpy.float32)
        electrodes = []
        amplitudes = []
        count      = 0
        for i in xrange(comm.size):
            middle      = ts[i].shape[2]/2
            templates[:,:,count:count+middle] = ts[i][:,:,:middle]
            templates[:,:,n_clusters+count:n_clusters+count+middle] = ts[i][:,:,middle:]
            count      += middle
            electrodes += cs[i]
            amplitudes += bs[i]
            os.remove(file_out_suff + '.templates-%d.npy' %i)
            os.remove(file_out_suff + '.electrodes-%d.npy' %i)
            os.remove(file_out_suff + '.amplitudes-%d.npy' %i)
            os.remove(file_out_suff + '.data-%d.pic' %i)
            for j in range(i, N_e, comm.size):
                result['data_' + str(j)]     = rs[i]['data_' + str(j)]
                result['clusters_' + str(j)] = rs[i]['clusters_' + str(j)]
                result['debug_' + str(j)]    = rs[i]['debug_' + str(j)]
                result['w_' + str(j)]        = rs[i]['w_' + str(j)]
                result['pca_' + str(j)]      = rs[i]['pca_' + str(j)]
                result['times_' + str(j)]    = rs[i]['times_' + str(j)]

        result['electrodes']   = numpy.array(electrodes)
        amplitudes             = numpy.array(amplitudes)

        templates, amplitudes, result, merged1 = algo.merging_cc(templates, amplitudes, result, cc_merge)
        templates, amplitudes, result, merged2 = algo.delete_mixtures(templates, amplitudes, result, cc_merge)

        io.print_info(["Number of global merges    : %d" %merged1[1], 
                       "Number of mixtures removed : %d" %merged2[1]])

        if os.path.exists(file_out_suff + '.templates.mat'):
            os.remove(file_out_suff + '.templates.mat')
        hdf5storage.savemat(file_out_suff + '.templates', {'templates' : templates})

        if os.path.exists(file_out_suff + '.clusters.mat'):
            os.remove(file_out_suff + '.clusters.mat')

        hdf5storage.savemat(file_out_suff + '.clusters',   result)
        hdf5storage.savemat(file_out_suff + '.limits', {'limits' : amplitudes})
        del result, templates, amplitudes, electrodes

        io.get_overlaps(params, erase=True)
