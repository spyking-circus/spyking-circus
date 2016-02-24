from .shared.utils import *
import h5py

def main(filename, params, nb_cpu, nb_gpu, use_gpu, file_name, benchmark):
    numpy.random.seed(45135)

    data_path      = os.path.dirname(os.path.abspath(file_name))
    data_suff, ext = os.path.splitext(os.path.basename(os.path.abspath(file_name)))
    file_out, ext  = os.path.splitext(os.path.abspath(file_name))

    if benchmark not in ['fitting', 'clustering', 'synchrony', 'smart-search', 'drifts']:
        if comm.rank == 0:
            io.print_and_log(['Benchmark need to be in [fitting, clustering, synchrony, smart-search, drifts]'], 'error', params)
        sys.exit(0)

    def write_benchmark(filename, benchmark, cells, rates, amplitudes, sampling, probe, trends=None):
        import cPickle
        to_write = {'benchmark' : benchmark}
        to_write['cells']      = cells
        to_write['rates']      = rates
        to_write['probe']      = probe
        to_write['amplitudes'] = amplitudes
        to_write['sampling']   = sampling
        if benchmark == 'drifts':
            to_write['drifts'] = trends
        cPickle.dump(to_write, open(filename + '.pic', 'w'))

    templates = io.load_data(params, 'templates')
    sim_same_elec   = 0.8
    trends          = None

    if benchmark == 'fitting':
        nb_insert       = 25
        n_cells         = numpy.random.random_integers(0, templates.shape[1]/2-1, nb_insert)
        rate            = nb_insert*[10]
        amplitude       = numpy.linspace(0.5, 5, nb_insert)
    if benchmark == 'clustering':
        n_point         = 10
        n_cells         = numpy.random.random_integers(0, templates.shape[1]/2-1, n_point**2)
        x, y            = numpy.mgrid[0:n_point,0:n_point]
        rate            = numpy.arange(0.5, 20, 2)[x.flatten()]
        amplitude       = numpy.arange(0.5, 5.5, 0.5)[y.flatten()]
    if benchmark == 'synchrony':
        nb_insert       = 5
        corrcoef        = 0.2
        n_cells         = nb_insert*[numpy.random.random_integers(0, templates.shape[1]/2-1, 1)[0]]
        rate            = 10./corrcoef
        amplitude       = 2
    if benchmark == 'smart-search':
        nb_insert       = 10
        n_cells         = nb_insert*[numpy.random.random_integers(0, templates.shape[1]/2-1, 1)[0]]
        rate            = 1 + 5*numpy.arange(nb_insert)
        amplitude       = 2
    if benchmark == 'drifts':
        n_point         = 5
        n_cells         = numpy.random.random_integers(0, templates.shape[1]/2-1, n_point**2)
        x, y            = numpy.mgrid[0:n_point,0:n_point]
        rate            = 5*numpy.ones(n_point)[x.flatten()]
        amplitude       = numpy.linspace(0.5, 5, n_point)[y.flatten()]
        trends          = numpy.random.randn(n_point**2)

    if comm.rank == 0:
        if os.path.exists(file_out):
            shutil.rmtree(file_out)

    if n_cells is None:
        n_cells    = 1
        cells      = [numpy.random.permutation(numpy.arange(n_cells))[0]]
    elif not numpy.iterable(n_cells):
        cells      = [n_cells]
        n_cells    = 1
    else:
        cells      = n_cells
        n_cells    = len(cells)

    if numpy.iterable(rate):
        assert len(rate) == len(cells), "Should have the same number of rates and cells"
    else:
        rate = [rate]*len(cells)

    if numpy.iterable(amplitude):
        assert len(amplitude) == len(cells), "Should have the same number of amplitudes and cells"
    else:
        amplitude = [amplitude]*len(cells)

    N_e             = params.getint('data', 'N_e')
    sampling_rate   = params.getint('data', 'sampling_rate')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    nodes, edges     = io.get_nodes_and_edges(params)
    N_t              = params.getint('data', 'N_t')
    N_total          = params.getint('data', 'N_total')
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    N_tm_init             = templates.shape[1]/2
    thresholds            = io.load_data(params, 'thresholds')
    limits                = io.load_data(params, 'limits')
    best_elecs            = io.load_data(params, 'electrodes')
    norms                 = io.load_data(params, 'norm-templates')

    if comm.rank == 0:
        if not os.path.exists(file_out):
            os.makedirs(file_out)

    if comm.rank == 0:
        write_benchmark(file_out, benchmark, cells, rate, amplitude, sampling_rate, params.get('data', 'mapping'), trends)


    comm.Barrier()

    if do_spatial_whitening:
        spatial_whitening  = io.load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = io.load_data(params, 'temporal_whitening')

    N_total        = params.getint('data', 'N_total')
    N_e            = params.getint('data', 'N_e')
    chunk_size     = params.getint('data', 'chunk_size')
    data_offset    = params.getint('data', 'data_offset')
    dtype_offset   = params.getint('data', 'dtype_offset')
    data_dtype     = params.get('data', 'data_dtype')
    myfile         = MPI.File()
    scalings       = []
    data_mpi       = get_mpi_type('float32')
    if comm.rank == 0:
        io.copy_header(data_offset, params.get('data', 'data_file'), file_name)

    comm.Barrier()
    g              = myfile.Open(comm, file_name, MPI.MODE_RDWR)
    g.Set_view(data_offset, data_mpi, data_mpi)

    for gcount, cell_id in enumerate(cells):
        best_elec   = best_elecs[cell_id]
        indices     = inv_nodes[edges[nodes[best_elec]]]
        count       = 0
        new_indices = []
        all_elecs   = numpy.random.permutation(numpy.arange(N_e))
        reference   = templates[:, cell_id].toarray().reshape(N_e, N_t)
        while len(new_indices) != len(indices) or (similarity >= sim_same_elec):   
            similarity  = 0
            if count == len(all_elecs):
                if comm.rank == 0:
                    io.print_and_log(["No electrode to move template %d (max similarity is %g)" %(cell_id, similarity)], 'error', params)
                sys.exit(0)
            else:
                n_elec = all_elecs[count]
                if benchmark not in ['synchrony', 'smart-search']:
                    local_test = n_elec != best_elec
                else:
                    local_test = n_elec == best_elec

                if local_test:
                    new_indices = inv_nodes[edges[nodes[n_elec]]]
                    idx = numpy.where(new_indices != best_elec)[0]
                    new_indices[idx] = numpy.random.permutation(new_indices[idx])

                    if len(new_indices) == len(indices):
                        new_temp                 = numpy.zeros(reference.shape, dtype=numpy.float32)
                        new_temp[new_indices, :] = reference[indices, :]
                        gmin = new_temp.min()
                        data = numpy.where(new_temp == gmin)
                        scaling = -thresholds[data[0][0]]/gmin
                        for i in xrange(templates.shape[1]/2):
                            match = templates[:, i].toarray().reshape(N_e, N_t)
                            d = numpy.corrcoef(match.flatten(), scaling*new_temp.flatten())[0, 1]
                            if d > similarity:
                                similarity = d
                else:
                    new_indices = []
            count += 1

        #if comm.rank == 0:
        #    print "Template", cell_id, "is shuffled from electrode", best_elec, "to", n_elec, "(max similarity is %g)" %similarity

        N_tm           = templates.shape[1]/2
        to_insert      = numpy.zeros(reference.shape, dtype=numpy.float32)
        to_insert[new_indices] = scaling*amplitude[gcount]*templates[:, cell_id].toarray().reshape(N_e, N_t)[indices]
        to_insert2     = numpy.zeros(reference.shape, dtype=numpy.float32)
        to_insert2[new_indices] = scaling*amplitude[gcount]*templates[:, cell_id + N_tm].toarray().reshape(N_e, N_t)[indices]

        mynorm     = numpy.sqrt(numpy.sum(to_insert.flatten()**2)/(N_e*N_t))
        mynorm2    = numpy.sqrt(numpy.sum(to_insert2.flatten()**2)/(N_e*N_t))
        to_insert  = to_insert.flatten()
        to_insert2 = to_insert2.flatten()

        limits     = numpy.vstack((limits, limits[cell_id]))
        best_elecs = numpy.concatenate((best_elecs, [n_elec]))

        norms      = numpy.insert(norms, N_tm, mynorm)
        norms      = numpy.insert(norms, 2*N_tm+1, mynorm2)
        scalings  += [scaling]
        
        templates = templates.tocoo()
        xdata     = templates.row
        ydata     = templates.col
        zdata     = templates.data
        idx       = numpy.where(ydata >= N_tm)[0]
        ydata[idx] += 1

        dx    = to_insert.nonzero()[0].astype(numpy.int32)
        xdata = numpy.concatenate((xdata, dx))
        ydata = numpy.concatenate((ydata, N_tm*numpy.ones(len(dx), dtype=numpy.int32)))
        zdata = numpy.concatenate((zdata, to_insert[dx]))

        dx    = to_insert2.nonzero()[0].astype(numpy.int32)
        xdata = numpy.concatenate((xdata, dx))
        ydata = numpy.concatenate((ydata, (2*N_tm + 1)*numpy.ones(len(dx), dtype=numpy.int32)))
        zdata = numpy.concatenate((zdata, to_insert2[dx]))
        templates = scipy.sparse.csc_matrix((zdata, (xdata, ydata)), shape=(N_e*N_t, 2*(N_tm+1)))

    borders, nb_chunks, chunk_len, last_chunk_len = io.analyze_data(params, chunk_size)
    if last_chunk_len > 0:
        nb_chunks += 1

    if comm.rank == 0:
        io.print_and_log(["Generating benchmark data [%s] with %d cells" %(benchmark, n_cells)], 'info', params)
        io.purge(file_out, '.data')

    template_shift = int((N_t-1)/2)
    all_chunks     = numpy.arange(nb_chunks)
    to_process     = all_chunks[numpy.arange(comm.rank, nb_chunks, comm.size)]
    loc_nb_chunks  = len(to_process)
    numpy.random.seed(comm.rank)

    if comm.rank == 0:
        pbar = get_progressbar(loc_nb_chunks)

    spiketimes_file     = open(os.path.join(file_out, data_suff + '.spiketimes-%d.data' %comm.rank), 'wb')
    amplitudes_file     = open(os.path.join(file_out, data_suff + '.amplitudes-%d.data' %comm.rank), 'wb')
    templates_file      = open(os.path.join(file_out, data_suff + '.templates-%d.data' %comm.rank), 'wb')
    real_amps_file      = open(os.path.join(file_out, data_suff + '.real_amps-%d.data' %comm.rank), 'wb')
    voltages_file       = open(os.path.join(file_out, data_suff + '.voltages-%d.data' %comm.rank), 'wb')

    for count, gidx in enumerate(to_process):

        if (last_chunk_len > 0) and (gidx == (nb_chunks - 1)):
            chunk_len  = last_chunk_len
            chunk_size = last_chunk_len/N_total

        result         = {'spiketimes' : [], 'amplitudes' : [], 'templates' : [], 'real_amps' : [], 'voltages' : []}
        offset         = gidx*chunk_size
        local_chunk, local_shape = io.load_chunk(params, gidx, chunk_len, chunk_size, nodes=nodes)

        if do_spatial_whitening:
            local_chunk = numpy.dot(local_chunk, spatial_whitening)
        if do_temporal_whitening:
            local_chunk = scipy.ndimage.filters.convolve1d(local_chunk, temporal_whitening, axis=0, mode='constant')

        if benchmark is 'synchrony':
            mips = numpy.random.rand(chunk_size) < rate[0]/float(sampling_rate)

        for idx in xrange(len(cells)):
            if benchmark == 'synchrony':
                sidx       = numpy.where(mips == True)[0]
                spikes     = numpy.zeros(chunk_size, dtype=numpy.bool)
                spikes[sidx[numpy.random.rand(len(sidx)) < corrcoef]] = True
            else:
                spikes     = numpy.random.rand(chunk_size) < rate[idx]/float(sampling_rate)

            if benchmark == 'drifts':
                amplitudes = numpy.ones(len(spikes)) + trends[idx]*((spikes + offset)/(5*60*float(sampling_rate)))
            else:
                amplitudes = numpy.ones(len(spikes))

            spikes[:N_t]   = False
            spikes[-N_t:]  = False
            spikes         = numpy.where(spikes == True)[0]
            n_template     = N_tm_init + idx
            loc_template   = templates[:, n_template].toarray().reshape(N_e, N_t)
            first_flat     = loc_template.T.flatten()
            norm_flat      = numpy.sum(first_flat**2)
            refractory     = int(5*1e-3*sampling_rate)         
            t_last         = -refractory
            for scount, spike in enumerate(spikes):
                if (spike - t_last) > refractory:
                    local_chunk[spike-template_shift:spike+template_shift+1, :] += amplitudes[scount]*loc_template.T
                    amp        = numpy.dot(local_chunk[spike-template_shift:spike+template_shift+1, :].flatten(), first_flat)
                    amp       /= norm_flat
                    result['real_amps']  += [amp]
                    result['spiketimes'] += [spike + offset]
                    result['amplitudes'] += [(amplitudes[scount], 0)]
                    result['templates']  += [n_template]
                    result['voltages']   += [local_chunk[spike, best_elecs[idx]]]
                    t_last                = spike

        spikes_to_write     = numpy.array(result['spiketimes'], dtype=numpy.int32)
        amplitudes_to_write = numpy.array(result['amplitudes'], dtype=numpy.float32)
        templates_to_write  = numpy.array(result['templates'], dtype=numpy.int32)
        real_amps_to_write  = numpy.array(result['real_amps'], dtype=numpy.float32)
        voltages_to_write   = numpy.array(result['voltages'], dtype=numpy.float32)

        spiketimes_file.write(spikes_to_write.tostring())   
        amplitudes_file.write(amplitudes_to_write.tostring())
        templates_file.write(templates_to_write.tostring())
        real_amps_file.write(real_amps_to_write.tostring())
        voltages_file.write(voltages_to_write.tostring())

        #print count, 'spikes inserted...'
        new_chunk    = numpy.zeros((chunk_size, N_total), dtype=numpy.float32)
        new_chunk[:, nodes] = local_chunk

        new_chunk   = new_chunk.flatten()
        g.Write_at(gidx*chunk_len, new_chunk)

        if comm.rank == 0:
            pbar.update(count)

    spiketimes_file.close()
    amplitudes_file.close()
    templates_file.close()
    real_amps_file.close()
    voltages_file.close()

    if comm.rank == 0:
        pbar.finish()


    g.Close()
    comm.Barrier()

    file_params = file_out + '.params'

    if comm.rank == 0:

        result_path = os.path.join(file_out, 'injected') 
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        shutil.copy2(params.get('data', 'data_file_noext') + '.params', file_params)
        shutil.copy2(params.get('data', 'file_out') + '.basis.hdf5', os.path.join(result_path, data_suff + '.basis.hdf5'))

        mydata = h5py.File(os.path.join(file_out, data_suff + '.templates.hdf5'), 'w')

        templates = templates.tocoo()
        mydata.create_dataset('temp_x', data=templates.row)
        mydata.create_dataset('temp_y', data=templates.col)
        mydata.create_dataset('temp_data', data=templates.data)
        mydata.create_dataset('temp_shape', data=numpy.array([N_e, N_t, templates.shape[1]], dtype=numpy.int32))
        mydata.create_dataset('limits', data=limits)
        mydata.create_dataset('norms', data=norms)
        mydata.close()

        mydata = h5py.File(os.path.join(file_out, data_suff + '.clusters.hdf5'), 'w')
        mydata.create_dataset('electrodes', data=best_elecs)
        mydata.close()


    comm.Barrier()
    if comm.rank == 0:
        io.collect_data(comm.size, io.load_parameters(file_name), erase=True, with_real_amps=True, with_voltages=True)
        io.change_flag(file_name, 'temporal', 'False')
        io.change_flag(file_name, 'spatial', 'False')
        io.change_flag(file_name, 'data_dtype', 'float32')
        io.change_flag(file_name, 'dtype_offset', 'auto')
        shutil.move(os.path.join(file_out, data_suff + '.result.hdf5'), os.path.join(result_path, data_suff + '.result.hdf5'))
                
        numpy.save(os.path.join(result_path, data_suff + '.scalings'), scalings)

        file_name_noext, ext = os.path.splitext(file_name)

        shutil.copy2(os.path.join(result_path, data_suff + '.basis.hdf5'), os.path.join(file_out, data_suff + '.basis.hdf5'))

        if benchmark not in ['fitting', 'synchrony']:
            shutil.move(os.path.join(file_out, data_suff + '.templates.hdf5'), os.path.join(result_path, data_suff + '.templates.hdf5'))
