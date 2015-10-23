from .shared.utils import *

def main(filename, params, nb_cpu, nb_gpu, use_gpu):
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
    output_dim     = params.getfloat('extracting', 'output_dim')
    cc_merge       = params.getfloat('extracting', 'cc_merge')
    noise_thr      = params.getfloat('extracting', 'noise_thr')
    tmp_limits     = params.get('fitting', 'amp_limits').replace('(', '').replace(')', '').split(',')
    amp_limits     = map(float, tmp_limits)
    elt_count      = 0
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)
    #################################################################

    if comm.rank == 0:
        print "Extracting templates from already found clusters..."

    thresholds                           = io.load_data(params, 'thresholds')
    clusters, spiketimes, N_clusters     = io.load_data(params, 'spike-cluster')
    inv_clusters                         = numpy.zeros(clusters.max()+1, dtype=numpy.int32)
    inv_clusters[numpy.unique(clusters)] = numpy.argsort(numpy.unique(clusters))

    if do_spatial_whitening or do_temporal_whitening:
        spatial_whitening  = io.load_data(params, 'spatial_whitening')
        temporal_whitening = io.load_data(params, 'temporal_whitening')

    result         = {}
    for i in xrange(N_clusters):
        result['data_' + str(i)]  = numpy.zeros((0, N_e * N_t), dtype=numpy.float32)
        result['times_' + str(i)] = numpy.zeros(0, dtype=numpy.int32)

    borders, nb_chunks, chunk_len, last_chunk_len = io.analyze_data(params)

    # I guess this is more relevant, to take signals from all over the recordings
    all_chunks = numpy.random.permutation(numpy.arange(nb_chunks))

    nb_templates = numpy.sum(comm.rank == numpy.mod(numpy.arange(N_clusters), comm.size))
    nb_elts      = max_elts_temp * nb_templates 

    if comm.rank == 0:
        pbar = get_progressbar(nb_elts)

    for gidx in all_chunks:

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
            local_offset    = gidx*chunk_size
            local_peaktimes = spiketimes[idx] - local_offset

            #print "Removing the useless borders..."
            local_borders   = (template_shift, chunk_size - template_shift)
            idx             = (local_peaktimes >= local_borders[0]) & (local_peaktimes < local_borders[1])
            local_peaktimes = local_peaktimes[idx]
            local_clusters  = inv_clusters[clusters[idx]]

            if len(local_peaktimes) > 0:
                all_times       = numpy.zeros((N_e, local_peaktimes[-1]-local_peaktimes[0]+1), dtype=numpy.bool)
                min_times       = numpy.maximum(local_peaktimes - local_peaktimes[0] - safety_time, 0)
                max_times       = numpy.minimum(local_peaktimes - local_peaktimes[0] + safety_time + 1, local_peaktimes[-1]-local_peaktimes[0])

                n_times         = len(local_peaktimes)
                argmax_peak     = numpy.random.permutation(numpy.arange(n_times))
                clusters_id     = local_clusters[argmax_peak]
                best_electrode  = numpy.argmin(local_chunk[local_peaktimes[argmax_peak]], 1)

                myslice         = numpy.mod(clusters_id, comm.size) == comm.rank
                argmax_peak     = argmax_peak[myslice]
                best_electrode  = best_electrode[myslice]
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
                        if (len(result['data_' + str(temp)]) < max_elts_temp):
                            elt_count                    += 1
                            sub_mat                       = local_chunk[peak-template_shift:peak+template_shift+1, :]
                            sub_mat                       = sub_mat.reshape(1, N_e * N_t)
                            result['data_' + str(temp)]   = numpy.vstack((result['data_' + str(temp)], sub_mat))
                            to_add                        = numpy.array([peak + local_offset], dtype=numpy.int32)
                            result['times_' + str(temp)]  = numpy.concatenate((result['times_' + str(temp)], to_add))
                        all_times[indices, min_times[idx]:max_times[idx]] = True

            if comm.rank == 0:
                pbar.update(elt_count)

    if comm.rank == 0:
        pbar.finish()

    total_nb_elts = 0
    for temp in xrange(N_clusters):
        total_nb_elts += len(result['data_' + str(temp)])

    gdata = gather_array(numpy.array([total_nb_elts], dtype=numpy.float32), comm, 0)
    if comm.rank == 0:
        print "We found", int(numpy.sum(gdata)), "spikes over", int(nb_elts), "requested"

    #print "Spikes extracted in", time.time() - t_start, "s"

    comm.Barrier()

    local_nb_clusters = 0
    for temp in xrange(comm.rank, N_clusters, comm.size):
        if len(result['data_' + str(temp)]) > 0:
            local_nb_clusters += 1

    #print total_nb_clusters, "found in", time.time() - t_start, "s"

    comm.Barrier()
    if comm.rank == 0:
        print "Extracting the templates..."
    t_start         = time.time()
    templates       = numpy.zeros((N_e, N_t, 2*local_nb_clusters), dtype=numpy.float32)
    count_templates = 0
    amps_lims       = []
    electrodes      = []


    for temp in xrange(comm.rank, N_clusters, comm.size):
        n_data           = len(result['data_' + str(temp)])
        if n_data > 0:
            data             = result['data_' + str(temp)].reshape(n_data, N_e, N_t)
            tmp_templates    = numpy.median(data, axis=0)
            tmpidx           = numpy.where(tmp_templates == tmp_templates.min())
            temporal_shift   = template_shift - tmpidx[1][0]
            electrodes      += [tmpidx[0][0]]
            indices          = inv_nodes[edges[nodes[electrodes[-1]]]]

            if temporal_shift > 0:
                templates[indices, temporal_shift:, count_templates] = tmp_templates[indices, :-temporal_shift]
            elif temporal_shift < 0:
                templates[indices, :temporal_shift, count_templates] = tmp_templates[indices, -temporal_shift:]
            else:
                templates[indices, :, count_templates] = tmp_templates[indices, :]

            x, y, z          = data.shape
            data_flat        = data.reshape(x, y*z)
            first_flat       = tmp_templates.reshape(y*z, 1)
            amplitudes       = numpy.dot(data_flat, first_flat)
            amplitudes      /= numpy.sum(first_flat**2)
            for i in xrange(x):
                data_flat[i, :] -= amplitudes[i]*first_flat[:, 0]

            variations       = 6*numpy.median(numpy.abs(amplitudes - numpy.median(amplitudes)))
            physical_limit   = noise_thr*(-thresholds[tmpidx[0][0]])/tmp_templates.min()
            amp_min          = max(physical_limit, numpy.median(amplitudes) - variations)
            amp_max          = min(amp_limits[1], numpy.median(amplitudes) + variations)
            amps_lims       += [[amp_min, amp_max]]

            if len(data_flat) > 1:
                pca              = mdp.nodes.PCANode(output_dim=1)
                res_pca          = pca(data_flat.astype(numpy.double))
                tmp_templates    = pca.get_projmatrix().reshape(y, z)
            else:
                tmp_templates    = data_flat.reshape(y, z)/numpy.sum(data_flat**2)
            
            if temporal_shift > 0:
                templates[indices, temporal_shift:, local_nb_clusters + count_templates] = tmp_templates[indices, :-temporal_shift]
            elif temporal_shift < 0:
                templates[indices, :temporal_shift, local_nb_clusters + count_templates] = tmp_templates[indices, -temporal_shift:]
            else:
                templates[indices, :, local_nb_clusters + count_templates] = tmp_templates[indices, :]

            count_templates += 1


    result['amplitudes'] = numpy.array(amps_lims, dtype=numpy.float32)
    result['electrodes'] = numpy.array(electrodes, dtype=numpy.int32)

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
        count      = 0
        amplitudes = []
        for i in xrange(comm.size):
            middle = ts[i].shape[2]/2
            templates[:,:,count:count+middle] = ts[i][:,:,:middle]
            templates[:,:,n_clusters+count:n_clusters+count+middle] = ts[i][:,:,middle:]
            count      += middle
            electrodes += cs[i]
            amplitudes += bs[i]
            os.remove(file_out_suff + '.templates-%d.npy' %i)
            os.remove(file_out_suff + '.electrodes-%d.npy' %i)
            os.remove(file_out_suff + '.amplitudes-%d.npy' %i)
            os.remove(file_out_suff + '.data-%d.pic' %i)
            for j in range(i, N_clusters, comm.size):

                result['data_' + str(j)]     = rs[i]['data_' + str(j)]
                result['debug_' + str(j)]    = numpy.zeros((2, len(result['data_' + str(j)])), dtype=numpy.float32)
                result['times_' + str(j)]    = rs[i]['times_' + str(j)]

        amplitudes             = numpy.array(amplitudes)
        templates, amplitudes, result, merged = algo.merging_cc(templates, amplitudes, result, cc_merge)

        io.print_info(["Number of global merges  : %d" %merged[1]])

        if os.path.exists(file_out_suff + '.templates.mat'):
            os.remove(file_out_suff + '.templates.mat')
        hdf5storage.savemat(file_out_suff + '.templates', {'templates' : templates})
        #print "Templates extracted and saved in", time.time() - t_start, "s"

        if os.path.exists(file_out_suff + '.clusters.mat'):
            os.remove(file_out_suff + '.clusters.mat')

        hdf5storage.savemat(file_out_suff + '.clusters',   result)
        hdf5storage.savemat(file_out_suff + '.limits', {'limits' : amplitudes})

        io.get_overlaps(params, erase=True)
