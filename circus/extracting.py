from .shared.utils import *
import circus.shared.algorithms as algo

def main(filename, params, nb_cpu, nb_gpu, use_gpu):
    numpy.random.seed(426236)
    
    import h5py
    parallel_hdf5 = h5py.get_config().mpi

    #################################################################
    data_file      = io.get_data_file(params)
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
    noise_thr      = params.getfloat('extracting', 'noise_thr')
    tmp_limits     = params.get('fitting', 'amp_limits').replace('(', '').replace(')', '').split(',')
    amp_limits     = map(float, tmp_limits)
    elt_count      = 0
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)
    #################################################################

    if comm.rank == 0:
        io.print_and_log(["Extracting templates from already found clusters..."], 'default', params)

    thresholds                           = io.load_data(params, 'thresholds')
    basis_proj, basis_rec                = io.load_data(params, 'basis')
    clusters, spiketimes, N_clusters     = io.load_data(params, 'spike-cluster')
    inv_clusters                         = numpy.zeros(clusters.max()+1, dtype=numpy.int32)
    inv_clusters[numpy.unique(clusters)] = numpy.argsort(numpy.unique(clusters))

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

    if do_spatial_whitening:
        spatial_whitening  = io.load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = io.load_data(params, 'temporal_whitening')

    if use_gpu and do_spatial_whitening:
        spatial_whitening = cmt.CUDAMatrix(spatial_whitening)

    result         = {}
    for i in xrange(N_clusters):
        result['data_tmp_' + str(i)]  = numpy.zeros((0, N_e * basis_proj.shape[1]), dtype=numpy.float32)
        result['times_' + str(i)]     = numpy.zeros(0, dtype=numpy.int32)

    borders, nb_chunks, chunk_len, last_chunk_len = data_file.analyze(chunk_size)

    # I guess this is more relevant, to take signals from all over the recordings
    all_chunks = numpy.random.permutation(numpy.arange(nb_chunks))

    nb_templates = numpy.sum(comm.rank == numpy.mod(numpy.arange(N_clusters), comm.size))
    nb_elts      = max_elts_temp * nb_templates 

    if comm.rank == 0:
        pbar = get_progressbar(nb_elts)

    for gidx in all_chunks:

        if (elt_count < nb_elts):
            #print "Node", comm.rank, "is analyzing chunk", gidx, "/", nb_chunks, " ..."
            local_chunk, local_shape = data_file.get_data(gidx, chunk_len, chunk_size, nodes=nodes)

            if do_spatial_whitening:
                if use_gpu:
                    local_chunk = cmt.CUDAMatrix(local_chunk)
                    local_chunk = local_chunk.dot(spatial_whitening).asarray()
                else:
                    local_chunk = numpy.dot(local_chunk, spatial_whitening)
            if do_temporal_whitening:
                local_chunk = scipy.ndimage.filters.convolve1d(local_chunk, temporal_whitening, axis=0, mode='constant')

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
                local_peaktimes = local_peaktimes[argmax_peak]
                
                #print "Selection of the peaks with spatio-temporal masks..."
                for idx in xrange(len(local_peaktimes)):
                    
                    if elt_count == nb_elts:
                        break

                    temp = clusters_id[idx]

                    if numpy.mod(temp, comm.size) == comm.rank:

                        elec = numpy.argmin(local_chunk[local_peaktimes[idx]])
                        indices = inv_nodes[edges[nodes[elec]]]
                        myslice = all_times[indices, min_times[idx]:max_times[idx]]
                        peak    = local_peaktimes[idx]
                        if not myslice.any():
                            if (len(result['data_tmp_' + str(temp)]) < max_elts_temp):
                                elt_count += 1
                                sub_mat    = local_chunk[peak-template_shift:peak+template_shift+1, :]
                                sub_mat    = numpy.dot(basis_rec, sub_mat)
                                nx, ny     = sub_mat.shape
                                sub_mat    = sub_mat.reshape((1, nx * ny))
                                result['data_tmp_' + str(temp)] = numpy.vstack((result['data_tmp_' + str(temp)], sub_mat))
                                to_add                          = numpy.array([peak + local_offset], dtype=numpy.int32)
                                result['times_' + str(temp)]    = numpy.concatenate((result['times_' + str(temp)], to_add))
                            all_times[indices, min_times[idx]:max_times[idx]] = True

            if comm.rank == 0:
                pbar.update(elt_count)

    if comm.rank == 0:
        pbar.finish()

    total_nb_elts = 0
    for temp in xrange(N_clusters):
        total_nb_elts += len(result['data_tmp_' + str(temp)])

    gdata = gather_array(numpy.array([total_nb_elts], dtype=numpy.float32), comm, 0)
    if comm.rank == 0:
        io.print_and_log(["Found %d spikes over %d requested" %(int(numpy.sum(gdata)), int(nb_elts))], 'default', params)

    #print "Spikes extracted in", time.time() - t_start, "s"

    comm.Barrier()

    local_nb_clusters = 0
    for temp in xrange(comm.rank, N_clusters, comm.size):
        if len(result['data_tmp_' + str(temp)]) > 0:
            local_nb_clusters += 1

    #print total_nb_clusters, "found in", time.time() - t_start, "s"
    gdata3     = gather_array(numpy.array([local_nb_clusters], dtype=numpy.float32), comm, 0)

    comm.Barrier()
    if comm.rank == 0:
        io.print_and_log(["Extracting the templates..."], 'default', params)
    
    total_nb_clusters = int(comm.bcast(numpy.array([int(numpy.sum(gdata3))], dtype=numpy.int32), root=0)[0])
    offsets    = numpy.zeros(comm.size, dtype=numpy.int32)
    for i in xrange(comm.size-1):
        offsets[i+1] = comm.bcast(numpy.array([local_nb_clusters], dtype=numpy.int32), root=i)

    if parallel_hdf5:
        node_pad   = numpy.sum(offsets[:comm.rank+1])        
        hfile      = h5py.File(file_out_suff + '.templates.hdf5', 'w', driver='mpio', comm=comm, libver='latest')
        norms      = hfile.create_dataset('norms', shape=(2*total_nb_clusters, ), dtype=numpy.float32, chunks=True)
        electrodes = hfile.create_dataset('electrodes', shape=(total_nb_clusters, ), dtype=numpy.int32, chunks=True)
        amps_lims  = hfile.create_dataset('limits', shape=(total_nb_clusters, 2), dtype=numpy.float32, chunks=True)
        g_count    = node_pad
        g_offset   = total_nb_clusters
    else:
        node_pad   = 0
        hfile      = h5py.File(file_out_suff + '.templates-%d.hdf5' %comm.rank, 'w', libver='latest')
        electrodes = hfile.create_dataset('electrodes', shape=(local_nb_clusters, ), dtype=numpy.int32, chunks=True)
        norms      = hfile.create_dataset('norms', shape=(2*local_nb_clusters, ), dtype=numpy.float32, chunks=True)
        amps_lims  = hfile.create_dataset('limits', shape=(local_nb_clusters, 2), dtype=numpy.float32, chunks=True)
        g_count    = 0
        g_offset   = local_nb_clusters
    
    cfile           = h5py.File(file_out_suff + '.clusters-%d.hdf5' %comm.rank, 'w', libver='latest')
    count_templates = node_pad

    temp_x     = numpy.zeros(0, dtype=numpy.int32)
    temp_y     = numpy.zeros(0, dtype=numpy.int32)
    temp_data  = numpy.zeros(0, dtype=numpy.float32)

    if comm.rank == 0:
        pbar = get_progressbar(local_nb_clusters)

    for temp in xrange(comm.rank, N_clusters, comm.size):
        n_data           = len(result['data_tmp_' + str(temp)])
        if n_data > 0:
            data                = result['data_tmp_' + str(temp)].reshape(n_data, basis_proj.shape[1], N_e)
            first_component     = numpy.median(data, axis=0)
            tmp_templates       = numpy.dot(first_component.T, basis_rec)
            electrodes[g_count] = indices[tmpidx[0][0]]
            indices             = inv_nodes[edges[nodes[electrodes[-1]]]]
            templates           = numpy.zeros((N_e, N_t), dtype=numpy.float32)
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

            x, y, z          = data.shape
            data_flat        = data.reshape(x, y*z)
            first_flat       = first_component.reshape(y*z, 1)
            amplitudes       = numpy.dot(data_flat, first_flat)
            amplitudes      /= numpy.sum(first_flat**2)
            for i in xrange(x):
                data_flat[i, :] -= amplitudes[i]*first_flat[:, 0]

            variations       = 10*numpy.median(numpy.abs(amplitudes - numpy.median(amplitudes)))
            physical_limit   = noise_thr*(-thresholds[indices[tmpidx[0][0]]])/tmp_templates.min()
            amp_min          = max(physical_limit, numpy.median(amplitudes) - variations)
            amp_max          = min(amp_limits[1], numpy.median(amplitudes) + variations)
            amps_lims[g_count] = [amp_min, amp_max]

            if len(data_flat) > 1:
                pca              = PCA(1)
                res_pca          = pca.fit_transform(data_flat.astype(numpy.double))
                second_component = pca.components_.T.astype(numpy.float32).reshape(y, z)
            else:
                second_component = data_flat.reshape(y, z)/numpy.sum(data_flat**2)
            
            tmp_templates = numpy.dot(second_component.T, basis_rec)
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

        io.write_datasets(cfile, to_write, result, ielec)

        if comm.rank == 0:
            pbar.update(count_templates)

    if comm.rank == 0:
        pbar.finish()

    #At the end we should have a templates variable to store.
    cfile.close()
    del result, templates, amps_lims
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
                loc_temp    = ts[i].get('templates')
                middle      = loc_temp.shape[2]//2
                norms[count:count+middle]                               = loc_norms[:middle]
                norms[n_clusters+count:n_clusters+count+middle]         = loc_norms[middle:]
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

    data_file.close()