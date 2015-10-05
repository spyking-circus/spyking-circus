from .shared.utils import *

def main(filename, params, nb_cpu, use_gpu):
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
    safety_time      = int(params.getfloat('basis', 'safety_time')*sampling_rate*1e-3)
    max_elts_elec    = params.getint('basis', 'max_elts')
    nb_elts          = int(params.getfloat('basis', 'nb_elts')*N_e*max_elts_elec)
    output_dim       = params.getfloat('basis', 'output_dim')
    elt_count        = 0
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)
    #################################################################


    if comm.rank == 0:
        print "Searching spikes to construct the PCA basis..."

    if do_spatial_whitening or do_temporal_whitening:
        spatial_whitening  = io.load_data(params, 'spatial_whitening')
        temporal_whitening = io.load_data(params, 'temporal_whitening')

    borders, nb_chunks, chunk_len, last_chunk_len = io.analyze_data(params)

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
        pbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()], maxval=nb_elts).start()



    for gcount, gidx in enumerate(chunks_to_load):

        if (elt_count < nb_elts):
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

            all_peaktimes       = numpy.array(all_peaktimes, dtype=numpy.int32)
            all_minimas         = numpy.array(all_minimas, dtype=numpy.int32)

            #print "Removing the useless borders..."
            local_borders   = (template_shift, local_shape - template_shift)
            idx             = (all_peaktimes >= local_borders[0]) & (all_peaktimes < local_borders[1])
            all_peaktimes   = all_peaktimes[idx]
            all_minimas     = all_minimas[idx]

            local_peaktimes = numpy.lib.arraysetops.unique(all_peaktimes)

            if len(local_peaktimes) > 0:

                diff_times      = local_peaktimes[-1]-local_peaktimes[0]
                all_times       = numpy.zeros((N_e, diff_times+1), dtype=numpy.bool)
                min_times       = numpy.maximum(local_peaktimes - local_peaktimes[0] - safety_time, 0)
                max_times       = numpy.minimum(local_peaktimes - local_peaktimes[0] + safety_time + 1, diff_times)

                n_times         = len(local_peaktimes)
                argmax_peak     = numpy.random.permutation(numpy.arange(n_times))
                best_electrode  = numpy.argmin(local_chunk[local_peaktimes[argmax_peak]], 1)

                #print "Selection of the peaks with spatio-temporal masks..."
                for idx, elec in zip(argmax_peak, best_electrode):
                    if elt_count == nb_elts:
                        break
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
        numpy.savez(file_out + '.basis', proj=pca.get_projmatrix().astype(numpy.float32), rec=pca.get_recmatrix().astype(numpy.float32))
        print "A basis with %s dimensions has been built..." %pca.get_projmatrix().shape[1]
