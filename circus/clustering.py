from .shared.utils import *
import circus.shared.algorithms as algo
from .shared import plot
import h5py
from circus.shared.probes import get_nodes_and_edges
from circus.shared.messages import print_and_log, init_logging


def main(params, nb_cpu, nb_gpu, use_gpu):

    parallel_hdf5 = h5py.get_config().mpi
    logger         = init_logging(params.logfile)
    logger         = logging.getLogger('circus.clustering')
    #################################################################
    data_file      = params.data_file
    data_file.open()
    N_e            = params.getint('data', 'N_e')
    N_total        = params.nb_channels
    N_t            = params.getint('detection', 'N_t')
    dist_peaks     = params.getint('detection', 'dist_peaks')
    template_shift = params.getint('detection', 'template_shift')
    file_out       = params.get('data', 'file_out')
    file_out_suff  = params.get('data', 'file_out_suff')
    sign_peaks     = params.get('detection', 'peaks')
    alignment      = params.getboolean('detection', 'alignment')
    matched_filter = params.getboolean('detection', 'matched-filter')
    spike_thresh   = params.getfloat('detection', 'spike_thresh')
    if params.get('data', 'global_tmp'):
        tmp_path_loc = os.path.join(os.path.abspath(params.get('data', 'data_file_noext')), 'tmp')
    else:
        tmp_path_loc = tempfile.gettempdir()
    plot_path      = os.path.join(params.get('data', 'data_file_noext'), 'plots')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    safety_time    = params.getint('clustering', 'safety_time')
    safety_space   = params.getboolean('clustering', 'safety_space')
    comp_templates = params.getboolean('clustering', 'compress')
    dispersion     = params.get('clustering', 'dispersion').replace('(', '').replace(')', '').split(',')
    dispersion     = map(float, dispersion)
    nodes, edges   = get_nodes_and_edges(params)
    chunk_size     = params.getint('data', 'chunk_size')
    max_elts_elec  = params.getint('clustering', 'max_elts')
    if sign_peaks == 'both':
       max_elts_elec *= 2
    nb_elts        = int(params.getfloat('clustering', 'nb_elts')*N_e*max_elts_elec)
    nb_repeats     = params.getint('clustering', 'nb_repeats')
    nclus_min      = params.getfloat('clustering', 'nclus_min')
    max_clusters   = params.getint('clustering', 'max_clusters')
    make_plots     = params.get('clustering', 'make_plots')
    sim_same_elec  = params.getfloat('clustering', 'sim_same_elec')
    noise_thr      = params.getfloat('clustering', 'noise_thr')
    remove_mixture = params.getboolean('clustering', 'remove_mixture')
    extraction     = params.get('clustering', 'extraction')
    smart_search   = params.getboolean('clustering', 'smart_search')
    smart_select   = params.getboolean('clustering', 'smart_select')
    if smart_select:
        m_ratio    = nclus_min
    else:
        m_ratio    = params.getfloat('clustering', 'm_ratio')
    test_clusters  = params.getboolean('clustering', 'test_clusters')
    tmp_limits     = params.get('fitting', 'amp_limits').replace('(', '').replace(')', '').split(',')
    amp_limits     = map(float, tmp_limits)
    elt_count      = 0
    sub_output_dim = params.getint('clustering', 'sub_dim')   
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)
    to_write         = ['clusters_', 'times_', 'data_', 'peaks_']
    #################################################################

    if sign_peaks == 'negative':
        search_peaks = ['neg']
    elif sign_peaks == 'positive':
        search_peaks = ['pos']
    elif sign_peaks == 'both':
        search_peaks = ['neg', 'pos']

    smart_searches = {}
    for p in search_peaks:
        smart_searches[p] = numpy.ones(N_e, dtype=numpy.float32)*int(smart_search)

    basis = {}

    if sign_peaks in ['negative', 'both']:
        basis['proj_neg'], basis['rec_neg'] = io.load_data(params, 'basis')
    if sign_peaks in ['positive', 'both']:
        basis['proj_pos'], basis['rec_pos'] = io.load_data(params, 'basis-pos')

    thresholds = io.load_data(params, 'thresholds')
    if do_spatial_whitening:
        spatial_whitening  = io.load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = io.load_data(params, 'temporal_whitening')

    if matched_filter:
        if sign_peaks in ['negative', 'both']:
            waveform_neg  = io.load_data(params, 'waveform')
            waveform_neg /= (numpy.abs(numpy.sum(waveform_neg))* len(waveform_neg))
            matched_tresholds_neg = io.load_data(params, 'matched-thresholds')
        if sign_peaks in ['positive', 'both']:
            waveform_pos  = io.load_data(params, 'waveform-pos')
            waveform_pos /= (numpy.abs(numpy.sum(waveform_pos))* len(waveform_pos))
            matched_tresholds_pos = io.load_data(params, 'matched-thresholds-pos')

    result   = {}

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

    if test_clusters:
        injected_spikes = io.load_data(params, 'injected_spikes')

    if comm.rank == 0:
        if not os.path.exists(tmp_path_loc):
            os.makedirs(tmp_path_loc)

    if alignment:
        cdata = numpy.linspace(-template_shift, template_shift, 5*N_t)
        xdata = numpy.arange(-2*template_shift, 2*template_shift+1)

    comm.Barrier()

    if use_gpu and do_spatial_whitening:
        spatial_whitening = cmt.CUDAMatrix(spatial_whitening, copy_on_host=False)

    elec_positions = {}

    for i in xrange(N_e):
        result['loc_times_' + str(i)] = numpy.zeros(0, dtype=numpy.int32)
        result['all_times_' + str(i)] = numpy.zeros(0, dtype=numpy.int32)
        result['times_' + str(i)]     = numpy.zeros(0, dtype=numpy.int32)
        result['clusters_' + str(i)]  = numpy.zeros(0, dtype=numpy.int32)
        result['peaks_' + str(i)]     = numpy.zeros(0, dtype=numpy.int32)
        for p in search_peaks:
            result['pca_%s_' %p + str(i)] = None
            result['norm_%s_' %p + str(i)] = 0
        indices = numpy.take(inv_nodes, edges[nodes[i]])
        elec_positions[i] = numpy.where(indices == i)[0]

    max_elts_elec //= comm.size
    nb_elts       //= comm.size
    few_elts        = False
    nb_chunks, _    = data_file.analyze(chunk_size)

    if nb_chunks < comm.size:

        res        = io.data_stats(params, show=False)
        chunk_size = int(res*params.rate//comm.size)
        if comm.rank == 0:
            print_and_log(["Too much cores, automatically resizing the data chunks"], 'debug', logger)

        nb_chunks, last_chunk_len = data_file.analyze(chunk_size)
    
    if smart_search is False:
        gpass = 1
    else:
        gpass = 0

    ## We will perform several passes to enhance the quality of the clustering
    while gpass < (nb_repeats + 1):

        comm.Barrier()

        if gpass == 1:
            sdata = all_gather_array(smart_searches[search_peaks[0]][comm.rank::comm.size], comm, 0)

        if comm.rank == 0:
            if gpass == 0:
                print_and_log(["Searching random spikes to sample amplitudes..."], 'default', logger)
            elif gpass == 1:
                if not numpy.all(sdata > 0):
                    lines = ["Smart Search disabled on %d electrodes" %(numpy.sum(sdata == 0))]
                    print_and_log(lines, 'info', logger)
                if numpy.any(sdata > 0):
                    print_and_log(["Smart Search of good spikes for the clustering (%d/%d)..." %(gpass, nb_repeats)], 'default', logger)
                else:
                    print_and_log(["Searching random spikes for the clustering (%d/%d) (no smart search)..." %(gpass, nb_repeats)], 'default', logger)
            else:
                print_and_log(["Searching random spikes to refine the clustering (%d/%d)..." %(gpass, nb_repeats)], 'default', logger)

        for i in xrange(N_e):
            if gpass == 0:
                for p in search_peaks:
                    result['tmp_%s_' %p + str(i)] = numpy.zeros(0, dtype=numpy.float32)
                    result['nb_chunks_%s_' %p + str(i)] = 1
            else:
                n_neighb = len(edges[nodes[i]])
                for p in search_peaks:
                    result['tmp_%s_' %p + str(i)] = numpy.zeros((0, basis['proj_%s' %p].shape[1] * n_neighb), dtype=numpy.float32)

            # If not the first pass, we sync all the detected times among nodes and give all nodes the w/pca
            result['all_times_' + str(i)] = numpy.concatenate((result['all_times_' + str(i)], all_gather_array(result['loc_times_' + str(i)], comm, dtype='int32')))
            result['loc_times_' + str(i)] = numpy.zeros(0, dtype=numpy.int32)
            
            if gpass == 1:
                for p in search_peaks:
                    result['pca_%s_' %p  + str(i)] = comm.bcast(result['pca_%s_' %p + str(i)], root=numpy.mod(i, comm.size))
                    result['data_%s_' %p + str(i)] = numpy.zeros((0, basis['proj_%s' %p].shape[1] * n_neighb), dtype=numpy.float32)
                    result['data_'  + str(i)]      = numpy.zeros((0, basis['proj_%s' %p].shape[1] * n_neighb), dtype=numpy.float32)
        # I guess this is more relevant, to take signals from all over the recordings
        numpy.random.seed(gpass)
        all_chunks = numpy.random.permutation(numpy.arange(nb_chunks, dtype=numpy.int64))
        rejected   = 0
        elt_count  = 0

        ## This is not easy to read, but during the smart search pass, we need to loop over all chunks, and every nodes should
        ## search spikes for a subset of electrodes, to avoid too many communications.
        if gpass <= 1:
            nb_elecs           = numpy.sum(comm.rank == numpy.mod(numpy.arange(N_e), comm.size))
            loop_max_elts_elec = params.getint('clustering', 'max_elts')
            if sign_peaks == 'both':
                loop_max_elts_elec *= 2
            loop_nb_elts       = numpy.int64(params.getfloat('clustering', 'nb_elts') * nb_elecs * loop_max_elts_elec)
            to_explore         = xrange(nb_chunks)

        else:
            loop_max_elts_elec = max_elts_elec
            loop_nb_elts       = nb_elts
            to_explore         = xrange(comm.rank, nb_chunks, comm.size)

        if comm.rank == 0:
            to_explore = get_tqdm_progressbar(to_explore)

        comm.Barrier()
        ## Random selection of spikes

        for gcount, gidx in enumerate(to_explore):

            gidx = all_chunks[gidx]

            if (elt_count < loop_nb_elts):
                #print "Node", comm.rank, "is analyzing chunk", gidx, "/", nb_chunks, " ..."
                local_chunk, t_offset = data_file.get_data(gidx, chunk_size, nodes=nodes)
                local_shape           = len(local_chunk)
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

                if matched_filter:

                    if sign_peaks in ['positive', 'both']:
                        filter_chunk = scipy.ndimage.filters.convolve1d(local_chunk, waveform_pos, axis=0, mode='constant')
                        for i in xrange(N_e):
                            peaktimes = algo.detect_peaks(filter_chunk[:, i], matched_tresholds_pos[i], mpd=dist_peaks)
                            all_peaktimes   = numpy.concatenate((all_peaktimes, peaktimes))
                            all_extremas    = numpy.concatenate((all_extremas, i*numpy.ones(len(peaktimes), dtype=numpy.int32)))

                    if sign_peaks in ['negative', 'both']:
                        filter_chunk = scipy.ndimage.filters.convolve1d(local_chunk, waveform_neg, axis=0, mode='constant')
                        for i in xrange(N_e):
                            peaktimes = algo.detect_peaks(filter_chunk[:, i], matched_tresholds_neg[i], mpd=dist_peaks)
                            all_peaktimes   = numpy.concatenate((all_peaktimes, peaktimes))
                            all_extremas    = numpy.concatenate((all_extremas, i*numpy.ones(len(peaktimes), dtype=numpy.int32)))

                else:
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
                local_offset    = t_offset

                if len(local_peaktimes) > 0:

                    diff_times      = local_peaktimes[-1]-local_peaktimes[0]
                    all_times       = numpy.zeros((N_e, diff_times+1), dtype=numpy.bool)
                    min_times       = numpy.maximum(local_peaktimes - local_peaktimes[0] - safety_time, 0)
                    max_times       = numpy.minimum(local_peaktimes - local_peaktimes[0] + safety_time + 1, diff_times)

                    n_times         = len(local_peaktimes)
                    argmax_peak     = numpy.random.permutation(numpy.arange(n_times))
                    all_idx         = numpy.take(local_peaktimes, argmax_peak)

                    if gpass > 1:
                        for elec in xrange(N_e):
                            subset  = result['all_times_' + str(elec)] - local_offset
                            peaks   = numpy.compress((subset >= 0) & (subset < (local_shape)), subset)
                            inter   = numpy.in1d(local_peaktimes, peaks)
                            indices = numpy.take(inv_nodes, edges[nodes[elec]])
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

                        if sign_peaks == 'negative':
                            elec = numpy.argmin(local_chunk[peak])
                            negative_peak = True
                            loc_peak      = 'neg'
                        elif sign_peaks == 'positive':
                            elec = numpy.argmax(local_chunk[peak])
                            negative_peak = False
                            loc_peak      = 'pos'
                        elif sign_peaks == 'both':
                            if numpy.abs(numpy.max(local_chunk[peak])) > numpy.abs(numpy.min(local_chunk[peak])):
                                elec = numpy.argmax(local_chunk[peak])
                                negative_peak = False
                                loc_peak      = 'pos'
                            else:
                                elec = numpy.argmin(local_chunk[peak])
                                negative_peak = True
                                loc_peak      = 'neg'
                        
                        if ((gpass > 1) or (numpy.mod(elec, comm.size) == comm.rank)):

                            indices = numpy.take(inv_nodes, edges[nodes[elec]])

                            if safety_space:
                                myslice = all_times[indices, min_times[midx]:max_times[midx]]
                            else:
                                myslice = all_times[elec, min_times[midx]:max_times[midx]]

                            is_local_extrema = elec in all_extremas[all_peaktimes == peak]

                            if is_local_extrema and not myslice.any():

                                to_accept  = False

                                if gpass == 1:
                                    to_update = result['data_%s_' %loc_peak + str(elec)]
                                else:
                                    to_update = result['tmp_%s_' %loc_peak + str(elec)]
                                
                                if len(to_update) < loop_max_elts_elec:
                                    
                                    if alignment:
                                        idx   = elec_positions[elec]
                                        zdata = numpy.take(local_chunk[peak-2*template_shift:peak+2*template_shift+1], indices, axis=1)
                                        ydata = numpy.arange(len(indices))
                                        if len(ydata) == 1:
                                            f        = scipy.interpolate.UnivariateSpline(xdata, zdata, s=0)
                                            if negative_peak:
                                                rmin = (numpy.argmin(f(cdata)) - len(cdata)/2.)/5.
                                            else:
                                                rmin = (numpy.argmax(f(cdata)) - len(cdata)/2.)/5.
                                            ddata    = numpy.linspace(rmin-template_shift, rmin+template_shift, N_t)
                                            sub_mat  = f(ddata).astype(numpy.float32).reshape(N_t, 1)
                                        else:
                                            f        = scipy.interpolate.RectBivariateSpline(xdata, ydata, zdata, s=0, ky=min(len(ydata)-1, 3))
                                            if negative_peak:
                                                rmin = (numpy.argmin(f(cdata, idx)[:, 0]) - len(cdata)/2.)/5.
                                            else:
                                                rmin = (numpy.argmax(f(cdata, idx)[:, 0]) - len(cdata)/2.)/5.
                                            ddata    = numpy.linspace(rmin-template_shift, rmin+template_shift, N_t)
                                            sub_mat  = f(ddata, ydata).astype(numpy.float32)
                                    else:
                                        sub_mat = numpy.take(local_chunk[peak-template_shift:peak+template_shift+1], indices, axis=1)

                                    if gpass == 0:
                                        to_accept  = True
                                        idx        = elec_positions[elec]
                                        ext_amp    = sub_mat[template_shift, idx]
                                        result['tmp_%s_' %loc_peak + str(elec)] = numpy.concatenate((result['tmp_%s_' %loc_peak + str(elec)], ext_amp))
                                    elif gpass == 1:

                                        if smart_searches[loc_peak][elec] > 0:
                                            
                                            idx     = elec_positions[elec]
                                            ext_amp = sub_mat[template_shift, idx]
                                            idx     = numpy.searchsorted(result['bounds_%s_' %loc_peak + str(elec)], ext_amp, 'right') - 1
                                            to_keep = result['hist_%s_' %loc_peak + str(elec)][idx] < numpy.random.rand() 

                                            if to_keep:
                                                to_accept = True
                                            else:
                                                rejected += 1
                                            
                                        else:
                                            to_accept = True

                                        if to_accept:
                                            sub_mat    = numpy.dot(basis['rec_%s' %loc_peak], sub_mat)
                                            nx, ny     = sub_mat.shape
                                            sub_mat    = sub_mat.reshape((1, nx * ny))
                                            result['data_%s_' %loc_peak + str(elec)] = numpy.vstack((result['data_%s_' %loc_peak + str(elec)], sub_mat))
                                                
                                    else:

                                        sub_mat    = numpy.dot(basis['rec_%s' %loc_peak], sub_mat)
                                        nx, ny     = sub_mat.shape
                                        sub_mat    = sub_mat.reshape((1, nx * ny))

                                        to_accept  = True
                                        result['tmp_%s_' %loc_peak + str(elec)] = numpy.vstack((result['tmp_%s_' %loc_peak + str(elec)], sub_mat))
                                        
                                if to_accept:
                                    elt_count += 1
                                    if gpass >= 1:
                                        to_add = numpy.array([peak + local_offset], dtype=numpy.int32)
                                        result['loc_times_' + str(elec)] = numpy.concatenate((result['loc_times_' + str(elec)], to_add))
                                    if gpass == 1:
                                        result['peaks_' + str(elec)] = numpy.concatenate((result['peaks_' + str(elec)], [int(negative_peak)]))
                                    if safety_space:
                                        all_times[indices, min_times[midx]:max_times[midx]] = True
                                    else:
                                        all_times[elec, min_times[midx]:max_times[midx]] = True

                if gpass == 0:
                    for i in xrange(comm.rank, N_e, comm.size):
                        for p in search_peaks:
                            if len(result['tmp_%s_' %p + str(i)]) < loop_max_elts_elec:
                                result['nb_chunks_%s_' %p + str(i)] += 1

        comm.Barrier()

        print_and_log(['Node %d has collected %d spikes and rejected %d spikes' % (comm.rank, elt_count, rejected)], 'debug', logger)
        gdata       = all_gather_array(numpy.array([elt_count], dtype=numpy.float32), comm, 0)
        gdata2      = gather_array(numpy.array([rejected], dtype=numpy.float32), comm, 0)
        nb_elements = numpy.int64(numpy.sum(gdata))
        nb_rejected = numpy.int64(numpy.sum(gdata2))
        nb_total    = numpy.int64(nb_elts*comm.size)

        if ((smart_search and (gpass == 0)) or (not smart_search and (gpass == 1))) and nb_elements == 0:
            if comm.rank == 0:
                print_and_log(['No waveforms found! Are the data properly loaded??'], 'error', logger)
            sys.exit(1)

        if nb_elements == 0:
            gpass = nb_repeats

        if comm.rank == 0:
            if gpass != 1:
                print_and_log(["We found %d spikes over %d requested" %(nb_elements, nb_total)], 'default', logger)
                if nb_elements == 0:
                    print_and_log(["No more isolated spikes in the recording, stop searching"], 'info', logger)
            else:
                print_and_log(["We found %d spikes over %d requested (%d rejected)" %(nb_elements, nb_total, nb_rejected)], 'default', logger)
                if nb_elements < 0.2*nb_total:
                    few_elts = True

        #CLUSTERING: once we have been through enough chunks (we don't need all of them), we run a clustering for each electrode.
        #print "Clustering the data..."
        local_nb_clusters = 0
        local_hits        = 0
        local_mergings    = 0
        cluster_results   = {}
        for p in search_peaks:
            cluster_results[p] = {}

        if gpass > 1:
            for ielec in xrange(N_e):
                for p in search_peaks:
                    result['tmp_%s_' %p + str(ielec)] = gather_array(result['tmp_%s_' %p + str(ielec)], comm, numpy.mod(ielec, comm.size), 1)
        elif gpass == 1:
            for ielec in xrange(comm.rank, N_e, comm.size):
                result['times_' + str(ielec)] = numpy.copy(result['loc_times_' + str(ielec)])

        if comm.rank == 0:
            if gpass == 0:
                print_and_log(["Estimating amplitudes distributions..."], 'default', logger)
            elif gpass == 1:
                print_and_log(["Computing density estimations..."], 'default', logger)
            else:
                print_and_log(["Refining density estimations..."], 'default', logger)
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)

        if gpass == 1:
            dist_file = tempfile.NamedTemporaryFile()
            tmp_file  = os.path.join(tmp_path_loc, os.path.basename(dist_file.name)) + '.hdf5'
            dist_file.close()
            result['dist_file'] = tmp_file
            tmp_h5py  = h5py.File(result['dist_file'], 'w', libver='latest')
            print_and_log(["Node %d will use temp file %s" %(comm.rank, tmp_file)], 'debug', logger)
        elif gpass > 1:
            tmp_h5py  = h5py.File(result['dist_file'], 'r', libver='latest')

        for ielec in xrange(comm.rank, N_e, comm.size):
            
            for p in search_peaks:
                cluster_results[p][ielec] = {}
                
                if gpass == 0:
                    if len(result['tmp_%s_' %p + str(ielec)]) > 1:

                        # Need to estimate the number of spikes
                        ratio = nb_chunks / float(result['nb_chunks_%s_' %p + str(ielec)])
                        ampmin, ampmax = numpy.min(result['tmp_%s_' %p + str(ielec)]), numpy.max(result['tmp_%s_' %p + str(ielec)])
                        if p == 'pos':
                            if matched_filter:
                                bound = matched_tresholds_pos[ielec]
                            else:
                                bound = thresholds[ielec]
                            bins =  [-numpy.inf] + numpy.linspace(bound, ampmax, 50).tolist() + [numpy.inf]

                        elif p == 'neg':
                            if matched_filter:
                                bound = -matched_tresholds_neg[ielec]
                            else:
                                bound = -thresholds[ielec]
                            bins  = [-numpy.inf] + numpy.linspace(ampmin, bound, 50).tolist() + [numpy.inf]
                        a, b  = numpy.histogram(result['tmp_%s_' %p + str(ielec)], bins)
                        a     = a/float(numpy.sum(a))
                        
                        z      = a[a > 0]
                        c      = 1./numpy.min(z)
                        d      = (1./(c*a))
                        d      = numpy.minimum(1, d)
                        target = numpy.sum(d)/ratio
                        twist  = numpy.sum(a*d)/target
                        factor = twist*c

                        result['hist_%s_'%p + str(ielec) ]   = factor*a
                        result['bounds_%s_' %p + str(ielec)] = b
                    else:
                        smart_searches[p][ielec] = 0

                    if smart_searches[p][ielec] > 0:
                        print_and_log(['Smart search is actived on channel %d' % ielec], 'debug', logger)

                elif gpass == 1:
                    if len(result['data_%s_' %p + str(ielec)]) > 1:

                        if result['pca_%s_' %p + str(ielec)] is None:
                            pca                               = PCA(sub_output_dim)
                            data                              = pca.fit_transform(result['data_%s_' %p + str(ielec)].astype(numpy.double)).astype(numpy.float32)
                            result['pca_%s_' %p + str(ielec)] = pca.components_.T.astype(numpy.float32)
                                
                        result['sub_%s_' %p + str(ielec)] = numpy.dot(result['data_%s_' %p + str(ielec)], result['pca_%s_' %p + str(ielec)])

                        rho, dist, sdist, nb_selec = algo.rho_estimation(result['sub_%s_' %p + str(ielec)], compute_rho=True, mratio=m_ratio)

                        result['rho_%s_' %p  + str(ielec)]  = rho
                        result['sdist_%s_' %p + str(ielec)] = sdist
                        result['norm_%s_' %p + str(ielec)]  = nb_selec
                        tmp_h5py.create_dataset('dist_%s_' %p + str(ielec), data=dist, chunks=True)
                        del dist, rho
                    else:
                        if result['pca_%s_' %p + str(ielec)] is None:
                            n_neighb                    = len(edges[nodes[ielec]])
                            dimension                   = basis['proj_%s' %p].shape[1] * n_neighb
                            result['pca_%s_' %p + str(ielec)] = numpy.identity(dimension, dtype=numpy.float32)
                        result['rho_%s_' %p  + str(ielec)] = numpy.zeros(len(result['data_%s_' %p + str(ielec)]), dtype=numpy.float32)
                        result['norm_%s_' %p + str(ielec)] = 1
                        result['sub_%s_' %p + str(ielec)]  = numpy.dot(result['data_%s_' %p + str(ielec)], result['pca_%s_' %p + str(ielec)])
                else:
                    if len(result['tmp_%s_' %p + str(ielec)]) > 1:
                        data      = numpy.dot(result['tmp_%s_' %p + str(ielec)], result['pca_%s_' %p + str(ielec)])
                        rho, dist, sdist, nb_selec = algo.rho_estimation(result['sub_%s_' %p + str(ielec)], update=(data, result['sdist_%s_' %p + str(ielec)]), mratio=m_ratio)
                        result['rho_%s_' %p  + str(ielec)]  = rho
                        result['sdist_%s_' %p + str(ielec)] = sdist
                        #result['norm_%s_' %p + str(ielec)] += nb_selec
                        del dist, rho

                if gpass == nb_repeats:

                    result.pop('tmp_%s_' %p + str(ielec))                    
                    n_data  = len(result['data_%s_' %p + str(ielec)])
                    n_min   = numpy.maximum(20, int(nclus_min*n_data))

                    if p == 'pos':
                        flag = 'positive'
                    elif p == 'neg':
                        flag = 'negative'

                    if (n_data > 1):
                        dist     = tmp_h5py.get('dist_%s_' %p + str(ielec))[:]
                        result['rho_%s_' %p + str(ielec)]  = -result['rho_%s_' %p + str(ielec)] + result['rho_%s_' %p + str(ielec)].max() 
                        result['rho_%s_' %p + str(ielec)] /= result['norm_%s_' %p + str(ielec)]
                        cluster_results[p][ielec]['groups'], r, d, c = algo.clustering(result['rho_%s_' %p + str(ielec)], dist,
                                                                                      m_ratio,
                                                                                      smart_select=smart_select,
                                                                                      n_min=n_min,
                                                                                      max_clusters=max_clusters)

                        # Now we perform a merging step, for clusters that look too similar
                        data = result['sub_%s_' %p + str(ielec)]
                        cluster_results[p][ielec]['groups'], merged = algo.merging(cluster_results[p][ielec]['groups'],
                                                                            sim_same_elec,
                                                                            data)

                        if make_plots not in ['None', '']:
                            save     = [plot_path, '%s_%d.%s' %(p, ielec, make_plots)]
                            injected = None
                            if test_clusters:
                                injected = numpy.zeros(len(result['data_%s_' %p + str(ielec)]), dtype=numpy.bool)
                                key = 'spikes_' + str(ielec)
                                thresh = 2
                                if injected_spikes.has_key(key):
                                    for icount, spike in enumerate(result['loc_times_' + str(ielec)]):
                                        idx = numpy.where(numpy.abs(spike - injected_spikes['spikes_' + str(ielec)]) < thresh)[0]
                                        if len(idx) > 0:
                                            if icount < (len(injected) - 1):
                                                injected[icount] = True

                            mask = numpy.where(cluster_results[p][ielec]['groups'] > -1)[0]
                            sel  = numpy.unique(cluster_results[p][ielec]['groups'][mask])
                            data = numpy.dot(result['data_%s_' %p + str(ielec)], result['pca_%s_' %p + str(ielec)])
                            plot.view_clusters(data, r, d, c,
                                                   cluster_results[p][ielec]['groups'], injected=injected,
                                                   save=save, smart_select=smart_select)

                        keys = ['loc_times_' + str(ielec), 'all_times_' + str(ielec), 'rho_%s_' %p + str(ielec), 'norm_%s_' %p + str(ielec)]
                        for key in keys:
                            if result.has_key(key):
                                result.pop(key)
                        mask                                = numpy.where(cluster_results[p][ielec]['groups'] > -1)[0]
                        cluster_results[p][ielec]['n_clus'] = len(numpy.unique(cluster_results[p][ielec]['groups'][mask]))
                        n_clusters                          = []
                        result['clusters_%s_' %p + str(ielec)] = cluster_results[p][ielec]['groups']
                        for i in numpy.unique(cluster_results[p][ielec]['groups'][mask]):
                            n_clusters += [numpy.sum(cluster_results[p][ielec]['groups'][mask] == i)]
            
                        line = ["Node %d: %d-%d %s templates on channel %d from %d spikes: %s" %(comm.rank, merged[0], merged[1], flag, ielec, n_data, str(n_clusters))]
                        print_and_log(line, 'default', logger)
                        if (merged[0]-merged[1]) == max_clusters:
                            local_hits += 1
                        local_mergings += merged[1]
                    else:
                        cluster_results[p][ielec]['groups'] = numpy.zeros(0, dtype=numpy.int32)
                        cluster_results[p][ielec]['n_clus'] = 0
                        result['clusters_%s_' %p + str(ielec)] = numpy.zeros(0, dtype=numpy.int32)
                        line = ["Node %d: not enough %s spikes on channel %d" %(comm.rank, flag, ielec)]
                        print_and_log(line, 'default', logger)

                    local_nb_clusters += cluster_results[p][ielec]['n_clus']

        if gpass >= 1:
            tmp_h5py.close()
        gpass += 1
    
    try:
        os.remove(result['dist_file'])
    except Exception:
        pass

    comm.Barrier()

    gdata      = gather_array(numpy.array([local_hits], dtype=numpy.float32), comm, 0)
    gdata2     = gather_array(numpy.array([local_mergings], dtype=numpy.float32), comm, 0)
    gdata3     = gather_array(numpy.array([local_nb_clusters], dtype=numpy.float32), comm, 0)

    mean_channels = 0

    if comm.rank == 0:
        total_hits        = int(numpy.sum(gdata))
        total_mergings    = int(numpy.sum(gdata2))
        total_nb_clusters = int(numpy.sum(gdata3))
        lines = ["Number of clusters found : %d" %total_nb_clusters,
                 "Number of local merges   : %d" %total_mergings]
        if few_elts:
            lines += ["Not enough spikes gathered: -put safety_space=False?"]
            if numpy.any(sdata > 0):
                lines += ["                            -decrease smart_search?"]
        if total_hits > 0 and not smart_select:
            lines += ["%d electrodes has %d clusters: -increase max_clusters?" %(total_hits, max_clusters)]
            lines += ["                              -increase sim_same_elec?"]
        print_and_log(lines, 'info', logger)

        print_and_log(["Estimating the templates with the %s procedure ..." %extraction], 'default', logger)

    if extraction in ['median-raw', 'median-pca', 'mean-raw', 'mean-pca']:

        total_nb_clusters = int(comm.bcast(numpy.array([int(numpy.sum(gdata3))], dtype=numpy.int32), root=0)[0])
        offsets    = numpy.zeros(comm.size, dtype=numpy.int32)
        for i in xrange(comm.size-1):
            offsets[i+1] = comm.bcast(numpy.array([local_nb_clusters], dtype=numpy.int32), root=i)
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

        data_file.close()
        
        to_explore = xrange(comm.rank, N_e, comm.size)

        if (comm.rank == 0):
            to_explore = get_tqdm_progressbar(to_explore)

        for ielec in to_explore:
        
            for p in search_peaks:

                #print "Dealing with cluster", ielec
                n_data   = len(result['data_%s_' %p + str(ielec)])
                n_neighb = len(edges[nodes[ielec]])
                data     = result['data_%s_' %p + str(ielec)].reshape(n_data, basis['proj_%s' %p].shape[1], n_neighb)
                mask     = numpy.where(cluster_results[p][ielec]['groups'] > -1)[0]
                loc_pad  = count_templates
                myamps   = []
                indices  = inv_nodes[edges[nodes[ielec]]]
                        
                for group in numpy.unique(cluster_results[p][ielec]['groups'][mask]):
                    electrodes[g_count] = ielec
                    myslice          = numpy.where(cluster_results[p][ielec]['groups'] == group)[0]
                    if p == 'pos':
                        myslice2     = numpy.where(result['peaks_' + str(ielec)] == 0)[0]
                    elif p == 'neg':
                        myslice2     = numpy.where(result['peaks_' + str(ielec)] == 1)[0]
                    if extraction == 'median-pca':
                        sub_data         = numpy.take(data, myslice, axis=0)
                        first_component  = numpy.median(sub_data, axis=0)
                        tmp_templates    = numpy.dot(first_component.T, basis['rec_%s' %p])
                    elif extraction == 'mean-pca':
                        sub_data         = numpy.take(data, myslice, axis=0)
                        first_component  = numpy.mean(sub_data, axis=0)
                        tmp_templates    = numpy.dot(first_component.T, basis['rec_%s' %p])
                    elif extraction == 'median-raw':                
                        labels_i         = numpy.random.permutation(myslice)[:min(len(myslice), 1000)]
                        times_i          = numpy.take(result['times_' + str(ielec)][myslice2], labels_i)
                        sub_data         = io.get_stas(params, times_i, labels_i, ielec, neighs=indices, nodes=nodes, pos=p)
                        first_component  = numpy.median(sub_data, 0)
                        tmp_templates    = first_component
                    elif extraction == 'mean-raw':                
                        labels_i         = numpy.random.permutation(myslice)[:min(len(myslice), 1000)]
                        times_i          = numpy.take(result['times_' + str(ielec)][myslice2], labels_i)
                        sub_data         = io.get_stas(sub_data, times_i, labels_i, ielec, neighs=indices, nodes=nodes, pos=p)
                        first_component  = numpy.mean(sub_data, 0)
                        tmp_templates    = first_component

                    if p == 'neg':
                        tmpidx           = divmod(tmp_templates.argmin(), tmp_templates.shape[1])
                    elif p == 'pos':
                        tmpidx           = divmod(tmp_templates.argmax(), tmp_templates.shape[1])

                    shift            = template_shift - tmpidx[1]
                    templates        = numpy.zeros((N_e, N_t), dtype=numpy.float32)
                    if shift > 0:
                        templates[indices, shift:] = tmp_templates[:, :-shift]
                    elif shift < 0:
                        templates[indices, :shift] = tmp_templates[:, -shift:]
                    else:
                        templates[indices, :] = tmp_templates

                    mean_channels += len(indices)
                    if comp_templates:
                        to_delete  = []
                        for i in indices:
                            if (numpy.abs(templates[i, :]).max() < 0.5*(thresholds[i]/spike_thresh)):
                                templates[i, :] = 0
                                to_delete += [i]
                        mean_channels -= len(to_delete)

                    templates  = templates.ravel()
                    dx         = templates.nonzero()[0].astype(numpy.int32)

                    temp_x     = numpy.concatenate((temp_x, dx))
                    temp_y     = numpy.concatenate((temp_y, count_templates*numpy.ones(len(dx), dtype=numpy.int32)))
                    temp_data  = numpy.concatenate((temp_data, templates[dx]))

                    norms[g_count] = numpy.sqrt(numpy.sum(templates.ravel()**2)/(N_e*N_t))

                    x, y, z          = sub_data.shape
                    sub_data_flat    = sub_data.reshape(x, y*z)
                    first_flat       = first_component.reshape(y*z, 1)
                    amplitudes       = numpy.dot(sub_data_flat, first_flat)
                    amplitudes      /= numpy.sum(first_flat**2)

                    variation        = numpy.median(numpy.abs(amplitudes - numpy.median(amplitudes)))
                    
                    physical_limit   = noise_thr*(-thresholds[indices[tmpidx[0]]])/tmp_templates.min()
                    amp_min          = min(0.8, max(physical_limit, numpy.median(amplitudes) - dispersion[0]*variation))
                    amp_max          = max(1.2, numpy.median(amplitudes) + dispersion[1]*variation)
                    amps_lims[g_count] = [amp_min, amp_max]
                    myamps            += [[amp_min, amp_max]]

                    for i in xrange(x):
                        sub_data_flat[i, :] -= amplitudes[i]*first_flat[:, 0]

                    if len(sub_data_flat) > 1:
                        pca              = PCA(1)
                        res_pca          = pca.fit_transform(sub_data_flat.astype(numpy.double)).astype(numpy.float32)
                        second_component = pca.components_.T.astype(numpy.float32).reshape(y, z)
                    else:
                        second_component = sub_data_flat.reshape(y, z)/numpy.sum(sub_data_flat**2)

                    if extraction in ['median-pca', 'mean-pca']:
                        tmp_templates = numpy.dot(second_component.T, basis['rec_%s' %p])
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

                    if comp_templates:
                        for i in to_delete:
                            sub_templates[i, :] = 0

                    sub_templates = sub_templates.ravel()
                    dx            = sub_templates.nonzero()[0].astype(numpy.int32)

                    temp_x     = numpy.concatenate((temp_x, dx))
                    temp_y     = numpy.concatenate((temp_y, offset*numpy.ones(len(dx), dtype=numpy.int32)))
                    temp_data  = numpy.concatenate((temp_data, sub_templates[dx]))

                    norms[g_count + g_offset] = numpy.sqrt(numpy.sum(sub_templates.ravel()**2)/(N_e*N_t))

                    count_templates += 1
                    g_count         += 1

                if make_plots not in ['None', '']:
                    if n_data > 1:
                        save     = [plot_path, '%s_%d.%s' %(p, ielec, make_plots)]
                        idx      = numpy.where(indices == ielec)[0][0]
                        sub_data = numpy.take(data,idx, axis=2)
                        nb_temp  = cluster_results[p][ielec]['n_clus']
                        vidx     = numpy.where((temp_y >= loc_pad) & (temp_y < loc_pad+nb_temp))[0] 
                        sub_tmp  = scipy.sparse.csr_matrix((temp_data[vidx], (temp_x[vidx], temp_y[vidx]-loc_pad)), shape=(N_e*N_t, nb_temp))
                        sub_tmp  = sub_tmp.toarray().reshape(N_e, N_t, nb_temp)
                        sub_tmp  = sub_tmp[ielec, :, :]
                        plot.view_waveforms_clusters(numpy.dot(sub_data, basis['rec_%s' %p]), cluster_results[p][ielec]['groups'],
                            thresholds[ielec], sub_tmp,
                            numpy.array(myamps), save=save)

                result['data_' + str(ielec)] = numpy.concatenate((result['data_' + str(ielec)], result['data_%s_'%p + str(ielec)]))
                if len(result['clusters_' + str(ielec)]) > 0:
                    max_offset = numpy.max(result['clusters_' + str(ielec)]) + 1
                else:
                    max_offset = 0
                mask = result['clusters_%s_' %p + str(ielec)] > -1
                result['clusters_%s_' %p + str(ielec)][mask] += max_offset
                result['clusters_' + str(ielec)] = numpy.concatenate((result['clusters_' + str(ielec)], result['clusters_%s_' %p + str(ielec)]))
                
            all_indices = numpy.zeros(0, dtype=numpy.int32)
            for p in search_peaks:
                if p == 'pos':
                    target = 0
                elif p == 'neg':
                    target = 1
                all_indices = numpy.concatenate((all_indices, numpy.where(result['peaks_' + str(ielec)] == target)[0]))

            result['times_' + str(ielec)] = result['times_' + str(ielec)][all_indices]
            result['peaks_' + str(ielec)] = result['peaks_' + str(ielec)][all_indices]

            io.write_datasets(cfile, to_write, result, ielec)


        #At the end we should have a templates variable to store.
        cfile.close()
        del result, amps_lims
        
        comm.Barrier()

        if local_nb_clusters > 0:
            mean_channels /= local_nb_clusters

        gdata4 = gather_array(numpy.array([mean_channels], dtype=numpy.float32), comm, 0)

        if comm.rank == 0:
            idx           = numpy.where(gdata4 != 0)[0]
            mean_channels = numpy.mean(gdata4[idx])
            if mean_channels < 3 and params.getfloat('clustering', 'cc_merge') != 1:
                print_and_log(["Templates on few channels only, cc_merge should be 1"], 'info', logger)

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

    if total_nb_clusters > 0:

        if comm.rank == 0:
            print_and_log(["Merging similar templates..."], 'default', logger)
    
        merged1 = algo.merging_cc(params, nb_cpu=nb_cpu, nb_gpu=nb_gpu, use_gpu=use_gpu)
    
        comm.Barrier()

        if remove_mixture:
            if comm.rank == 0:
                print_and_log(["Removing mixtures..."], 'default', logger)
            merged2 = algo.delete_mixtures(params, nb_cpu=nb_cpu, nb_gpu=nb_gpu, use_gpu=use_gpu)
        else:
            merged2 = [0, 0]

    else:
        merged1 = [0, 0]
        merged2 = [0, 0]

    if comm.rank == 0:

        print_and_log(["Number of global merges    : %d" %merged1[1], 
                          "Number of mixtures removed : %d" %merged2[1]], 'info', logger)    
    
    
    comm.Barrier()
    io.get_overlaps(params, erase=True, nb_cpu=nb_cpu, nb_gpu=nb_gpu, use_gpu=use_gpu)
