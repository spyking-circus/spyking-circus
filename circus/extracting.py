from utils import *

def main(filename, params, nb_cpu, use_gpu):
    numpy.random.seed(426236)

    #################################################################
    sampling_rate  = params.getint('data', 'sampling_rate')
    N_e            = params.getint('data', 'N_e')
    N_t            = params.getint('data', 'N_t')
    N_total        = params.getint('data', 'N_total')
    template_shift = params.getint('data', 'template_shift')
    chunk_size     = params.getint('data', 'chunk_size')
    file_out       = params.get('data', 'file_out')
    file_out_suff  = params.get('data', 'file_out_suff')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    nodes, edges   = io.get_nodes_and_edges(params)
    safety_time    = int(params.getfloat('extracting', 'safety_time')*sampling_rate*1e-3)
    max_elts_temp  = params.getint('extracting', 'max_elts')
    nb_elts        = int(params.getfloat('extracting', 'nb_elts')*N_e*max_elts_temp)
    output_dim     = params.getfloat('extracting', 'output_dim')
    tmp_limits     = params.get('fitting', 'amp_limits').replace('(', '').replace(')', '').split(',')
    amp_limits     = map(float, tmp_limits)
    elt_count      = 0
    elts           = numpy.zeros((N_t, nb_elts), dtype=numpy.float32)
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)
    #################################################################

    if comm.rank == 0:
        print "Getting templates from already found clusters..."

    clusters, spiketimes, N_clusters     = io.load_data(params, 'spike-cluster')
    inv_clusters                         = numpy.zeros(clusters.max()+1, dtype=numpy.int32)
    inv_clusters[numpy.unique(clusters)] = numpy.argsort(numpy.unique(clusters))

    if do_spatial_whitening or do_temporal_whitening:
        spatial_whitening  = io.load_data(params, 'spatial_whitening')
        temporal_whitening = io.load_data(params, 'temporal_whitening')

    result         = {}
    for i in xrange(N_clusters):
        result['temp_' + str(i)]  = []

    max_elts_temp /= comm.size
    nb_elts       /= comm.size

    borders, nb_chunks, chunk_len, last_chunk_len = io.analyze_data(params)

    # I guess this is more relevant, to take signals from all over the recordings
    all_chunks = numpy.random.permutation(numpy.arange(nb_chunks))

    if comm.rank == 0:
        pbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()], maxval=nb_elts).start()


    for gidx in all_chunks[numpy.arange(comm.rank, nb_chunks, comm.size)]:

        if (elt_count < nb_elts):
            #print "Node", comm.rank, "is analyzing chunk", gidx, "/", nb_chunks, " ..."
            local_chunk, local_shape = io.load_chunk(params, gidx, chunk_len, nodes=nodes)

            if do_spatial_whitening:
                local_chunk = numpy.dot(local_chunk, spatial_whitening)
            if do_temporal_whitening:
                for i in xrange(N_e):
                    local_chunk[:, i] = numpy.convolve(local_chunk[:, i], temporal_whitening, 'same')

            #print "Extracting the peaks..."
            idx             = numpy.where((spiketimes >= gidx*chunk_size) & (spiketimes < (gidx+1)*chunk_size))[0]
            local_peaktimes = spiketimes[idx] - gidx*chunk_size

            #print "Removing the useless borders..."
            local_borders   = (template_shift, chunk_size - template_shift)
            idx             = (local_peaktimes >= local_borders[0]) & (local_peaktimes < local_borders[1])
            local_peaktimes = local_peaktimes[idx]
            local_clusters  = inv_clusters[clusters[idx]]

            if len(local_peaktimes) > 0:
                all_times       = numpy.zeros((N_e, local_peaktimes[-1]-local_peaktimes[0]+1), dtype=numpy.bool)
                min_times       = numpy.maximum(local_peaktimes - local_peaktimes[0] - safety_time, 0)
                max_times       = numpy.minimum(local_peaktimes - local_peaktimes[0] + safety_time + 1, local_peaktimes[-1]-local_peaktimes[0])

                abs_chunks      = local_chunk[local_peaktimes]
                argmax_peak     = numpy.random.permutation(numpy.arange(len(local_peaktimes)))
                clusters_id     = local_clusters[argmax_peak]
                best_electrode  = numpy.argmin(abs_chunks[argmax_peak], 1)
                local_peaktimes = local_peaktimes[argmax_peak]

                #print "Selection of the peaks with spatio-temporal masks..."
                #for idx, temp, elec in zip(argmax_peak, clusters_id, best_electrode):
                for idx in xrange(len(local_peaktimes)):
                    if elt_count == nb_elts:
                        break
                    elec    = best_electrode[idx]
                    temp    = clusters_id[idx]
                    indices = inv_nodes[edges[nodes[elec]]]
                    myslice = all_times[indices, min_times[idx]:max_times[idx]]
                    peak    = local_peaktimes[idx]
                    if not myslice.any():
                        if (len(result['temp_' + str(temp)]) < max_elts_temp):
                            elts[:, elt_count]           = local_chunk[peak-template_shift:peak+template_shift+1, elec]
                            elt_count                   += 1
                            result['temp_' + str(temp)] += [local_chunk[peak-template_shift:peak+template_shift+1, :]]
                        all_times[indices, min_times[idx]:max_times[idx]] = True

            if comm.rank == 0:
                pbar.update(elt_count)

    if comm.rank == 0:
        pbar.finish()

    total_nb_elts = 0
    for temp in xrange(N_clusters):
        total_nb_elts += len(result['temp_' + str(temp)])

    gdata = gather_array(numpy.array([total_nb_elts], dtype=numpy.float32), comm, 0)
    if comm.rank == 0:
        print "We have found a total of", int(numpy.sum(gdata)), "over", int(nb_elts*comm.size)

    #print "Spikes extracted in", time.time() - t_start, "s"

    #CLUSTERING: once we have been through enough chunks (we don't need all of them), we run a clustering for each electrode.
    #print "Clustering the data..."
    total_nb_clusters = 0
    cluster_results   = {}

    gdata = gather_array(elts[:, :elt_count], comm, 0)
    if comm.rank == 0:
        pca        = mdp.nodes.PCANode(output_dim=output_dim)
        res_pca    = pca(elts.astype(numpy.double).T)
        numpy.savez(file_out + '.basis', proj=pca.get_projmatrix().astype(numpy.float32), rec=pca.get_recmatrix().astype(numpy.float32))
        print "A basis with %s dimensions has been built..." %pca.get_projmatrix().shape[1]

    comm.Barrier()

    basis_proj, basis_rec = io.load_data(params, 'basis')

    if comm.rank == 0:
        print "Gathering data among nodes and start extracting the templates..."

    for temp in xrange(N_clusters):
        for i in xrange(len(result['temp_' + str(temp)])):
            result['temp_' + str(temp)][i] = numpy.dot(basis_proj.T, result['temp_' + str(temp)][i])
        result['temp_' + str(temp)] = numpy.array(result['temp_' + str(temp)], dtype=numpy.float32)
        n_data                      = len(result['temp_' + str(temp)])
        result['temp_' + str(temp)] = result['temp_' + str(temp)].reshape(n_data, N_e*basis_proj.shape[1])
        result['temp_' + str(temp)] = gather_array(numpy.array(result['temp_' + str(temp)], dtype=numpy.float32), comm, numpy.mod(temp, comm.size), 1)


    total_nb_clusters = 0
    for temp in xrange(comm.rank, N_clusters, comm.size):
        if len(result['temp_' + str(temp)]) > 0:
            total_nb_clusters += 1

    #print total_nb_clusters, "found in", time.time() - t_start, "s"

    comm.Barrier()
    if comm.rank == 0:
        print "Extracting the templates..."
    t_start         = time.time()
    templates       = numpy.zeros((N_e, N_t, 2*total_nb_clusters), dtype=numpy.float32)
    count_templates = 0
    amplitudes_lims = []


    for temp in xrange(comm.rank, N_clusters, comm.size):
        n_data           = len(result['temp_' + str(temp)])
        if n_data > 0:
            data             = result['temp_' + str(temp)].reshape(n_data, basis_proj.shape[1], N_e)
            first_component  = numpy.median(data, axis=0)
            templates[:, :, count_templates] = numpy.dot(first_component.T, basis_rec)
            x, y, z          = data.shape
            data_flat        = data.reshape(x, y*z)
            first_flat       = first_component.reshape(y*z, 1)
            amplitudes       = numpy.dot(data_flat, first_flat)
            amplitudes      /= numpy.sum(first_flat**2)
            for i in xrange(x):
                data_flat[i, :] -= amplitudes[i]*first_flat[:, 0]

            xamp, yamp       = numpy.histogram(amplitudes, len(amplitudes))
            amp_data         = xamp/float(numpy.sum(xamp))
            amp_min          = max(amp_limits[0], yamp[1:][numpy.where(numpy.cumsum(amp_data) > 0.001)[0][0]])
            amp_max          = min(amp_limits[1], yamp[1:][numpy.where(numpy.cumsum(amp_data) > 0.999)[0][0]])
            amplitudes_lims += [[amp_min, amp_max]]

            if len(data_flat) > 1:
                pca              = mdp.nodes.PCANode(output_dim=1)
                res_pca          = pca(data_flat.astype(numpy.double))
                second_component = pca.get_projmatrix().reshape(y, z)
            else:
                second_component = data_flat.reshape(y, z)/numpy.sum(data_flat**2)
            templates[:, :, total_nb_clusters + count_templates] = numpy.dot(second_component.T, basis_rec)
            count_templates += 1


    #At the end we should have a templates variable to store.
    numpy.save(file_out_suff + '.templates-%d' %comm.rank, templates)
    numpy.save(file_out_suff + '.amplitudes-%d' %comm.rank, amplitudes_lims)
    comm.Barrier()

    if comm.rank == 0:
        ts         = [numpy.load(file_out_suff + '.templates-%d.npy' %i) for i in xrange(comm.size)]
        bs         = [numpy.load(file_out_suff + '.amplitudes-%d.npy' %i).tolist() for i in xrange(comm.size)]
        result     = {}
        n_clusters = numpy.sum([ts[i].shape[2] for i in xrange(comm.size)])/2
        print "Number of templates found is", n_clusters
        templates  = numpy.zeros((N_e, N_t, 2*n_clusters), dtype=numpy.float32)
        count      = 0
        amplitudes = []
        for i in xrange(comm.size):
            middle = ts[i].shape[2]/2
            templates[:,:,count:count+middle] = ts[i][:,:,:middle]
            templates[:,:,n_clusters+count:n_clusters+count+middle] = ts[i][:,:,middle:]
            count      += middle
            amplitudes += bs[i]
            os.remove(file_out_suff + '.templates-%d.npy' %i)
            os.remove(file_out_suff + '.amplitudes-%d.npy' %i)

        if os.path.exists(file_out_suff + '.templates.mat'):
            os.remove(file_out_suff + '.templates.mat')
        hdf5storage.savemat(file_out_suff + '.templates', {'templates' : templates})
        #print "Templates extracted and saved in", time.time() - t_start, "s"

        if os.path.exists(file_out_suff + '.clusters.mat'):
            os.remove(file_out_suff + '.clusters.mat')

        hdf5storage.savemat(file_out_suff + '.clusters',   result)
        hdf5storage.savemat(file_out_suff + '.limits', {'limits' : amplitudes})

        io.get_overlaps(params)
