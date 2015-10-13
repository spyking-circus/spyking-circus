from .shared.utils import *

def main(filename, params, nb_cpu, nb_gpu, use_gpu):
    numpy.random.seed(451235)

    def write_benchmark(filename, benchmark, cells, rates, amplitudes, sampling, probe):
        import cPickle
        to_write = {'benchmark' : benchmark}
        to_write['cells']      = cells
        to_write['rates']      = rates
        to_write['probe']      = probe
        to_write['amplitudes'] = amplitudes
        to_write['sampling']   = sampling
        cPickle.dump(to_write, open(filename + '.pic', 'w'))

    benchmark = 'clustering'
    templates = io.load_data(params, 'templates')
    sim_same_elec   = 0.8

    if benchmark == 'fitting':
        nb_insert       = 25
        n_cells         = numpy.random.random_integers(0, templates.shape[2]/2-1, nb_insert)
        file_name       = 'synthetic/fake_1'
        rate            = nb_insert*[10]
        amplitude       = numpy.linspace(0.5, 5, nb_insert)
    if benchmark == 'clustering':
        n_point         = 5
        n_cells         = numpy.random.random_integers(0, templates.shape[2]/2-1, n_point**2)
        file_name       = 'synthetic/fake_2'
        x, y            = numpy.mgrid[0:n_point,0:n_point]
        rate            = numpy.linspace(0.5, 20, n_point)[x.flatten()]
        amplitude       = numpy.linspace(0.5, 5, n_point)[y.flatten()]
    if benchmark == 'synchrony':
        nb_insert       = 5
        corrcoef        = 0.2
        n_cells         = nb_insert*[numpy.random.random_integers(0, templates.shape[2]/2-1, 1)[0]]
        file_name       = 'synthetic/fake_3'
        rate            = 10./corrcoef
        amplitude       = 2

    if comm.rank == 0:
        if os.path.exists(file_name):
            shutil.rmtree(file_name)

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
    clusters         = io.load_data(params, 'clusters')
    gain             = params.getfloat('data', 'gain')
    N_total          = params.getint('data', 'N_total')
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    N_tm_init             = templates.shape[2]/2
    thresholds            = io.load_data(params, 'thresholds')

    if comm.rank == 0:
        write_benchmark(file_name, benchmark, cells, rate, amplitude, sampling_rate, params.get('data', 'mapping'))

    if comm.rank == 0:
        if not os.path.exists(file_name):
            os.makedirs(file_name)

    comm.Barrier()

    if do_spatial_whitening or do_temporal_whitening:
        spatial_whitening  = io.load_data(params, 'spatial_whitening')
        temporal_whitening = io.load_data(params, 'temporal_whitening')

    N_total        = params.getint('data', 'N_total')
    N_e            = params.getint('data', 'N_e')
    file_out       = file_name + '/' + file_name.split('/')[-1]
    chunk_size     = params.getint('data', 'chunk_size')
    data_offset    = params.getint('data', 'data_offset')
    dtype_offset   = params.getint('data', 'dtype_offset')
    data_dtype     = params.get('data', 'data_dtype')
    myfile         = MPI.File()
    scalings       = []
    best_elecs     = []
    data_mpi       = get_mpi_type(data_dtype)
    if comm.rank == 0:
        file = open(file_name + '.raw', 'w')
        for i in xrange(data_offset):
            f.write('1')
        file.close()
    comm.Barrier()
    g              = myfile.Open(comm, file_name + '.raw', MPI.MODE_RDWR)
    g.Set_view(data_offset, data_mpi, data_mpi)

    for gcount, cell_id in enumerate(cells):
        best_elec   = clusters['electrodes'][cell_id]
        indices     = inv_nodes[edges[nodes[best_elec]]]
        count       = 0
        new_indices = []
        best_elecs += [0]
        all_elecs   = numpy.random.permutation(numpy.arange(N_e))
        while len(new_indices) != len(indices) or (similarity >= sim_same_elec):
            similarity  = 0
            if count == len(all_elecs):
                if comm.rank == 0:
                    print "No electrode to move template %d (max similarity is %g) !!!" %(cell_id, similarity)
                sys.exit(0)
            else:
                n_elec = all_elecs[count]
                best_elecs[-1] = n_elec
                if benchmark is not 'synchrony':
                    local_test = n_elec != best_elec
                else:
                    local_test = n_elec == best_elec

                if local_test:
                    new_indices = inv_nodes[edges[nodes[n_elec]]]
                    idx = numpy.where(new_indices != best_elec)[0]
                    new_indices[idx] = numpy.random.permutation(new_indices[idx])

                    if len(new_indices) == len(indices):
                        new_temp                 = templates[:, :, cell_id].copy()
                        new_temp[indices, :]     = 0
                        new_temp[new_indices, :] = templates[indices, :, cell_id]
                        gmin = templates[:, :, cell_id].min()
                        data = numpy.where(new_temp == gmin)
                        scaling = -thresholds[data[0][0]]/gmin
                        for i in xrange(templates.shape[2]/2):
                            d = numpy.corrcoef(templates[:, :, i].flatten(), scaling*new_temp.flatten())[0, 1]
                            if d > similarity:
                                similarity = d
                else:
                    new_indices = []
            count += 1
        if comm.rank == 0:
            print "Template", cell_id, "is shuffled from electrode", best_elec, "to", n_elec, "(max similarity is %g)" %similarity

        old_templates = templates.copy()
        N_tm          = old_templates.shape[2]/2
        new_line      = numpy.zeros((old_templates.shape[0], old_templates.shape[1]))
        templates     = numpy.dstack((old_templates[:, :, :N_tm], new_line))
        templates     = numpy.dstack((templates, old_templates[:, :, N_tm:]))
        templates     = numpy.dstack((templates, new_line))

        scalings                       += [scaling]
        templates[new_indices, :, N_tm] = scaling*amplitude[gcount]*old_templates[indices, :, cell_id]
        templates[new_indices, :, -1]   = scaling*amplitude[gcount]*old_templates[indices, :, cell_id + N_tm]

    borders, nb_chunks, chunk_len, last_chunk_len = io.analyze_data(params, chunk_size)
    if last_chunk_len > 0:
        nb_chunks += 1

    if comm.rank == 0:
        io.print_info(["Generating benchmark data [%s] with %d cells" %(benchmark, n_cells)])
        io.purge(file_out, '.data')

    template_shift = int((N_t-1)/2)
    all_chunks     = numpy.arange(nb_chunks)
    to_process     = all_chunks[numpy.arange(comm.rank, nb_chunks, comm.size)]
    loc_nb_chunks  = len(to_process)
    numpy.random.seed(comm.rank)

    if comm.rank == 0:
        pbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()], maxval=loc_nb_chunks).start()

    file_name_noext     = file_name.split('/')[-1]
    spiketimes_file     = open(file_name +'/'+file_name_noext+ '.spiketimes-%d.data' %comm.rank, 'w')
    amplitudes_file     = open(file_name +'/'+file_name_noext+ '.amplitudes-%d.data' %comm.rank, 'w')
    templates_file      = open(file_name +'/'+file_name_noext+ '.templates-%d.data' %comm.rank, 'w')
    real_amps_file      = open(file_name +'/'+file_name_noext+ '.real_amps-%d.data' %comm.rank, 'w')
    voltages_file       = open(file_name +'/'+file_name_noext+ '.voltages-%d.data' %comm.rank, 'w')

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
            for i in xrange(N_e):
                local_chunk[:, i] = numpy.convolve(local_chunk[:, i], temporal_whitening, 'same')

        if benchmark is 'synchrony':
            mips = numpy.random.rand(chunk_size) < rate[0]/float(sampling_rate)

        for idx in xrange(len(cells)):
            if benchmark is 'synchrony':
                sidx       = numpy.where(mips == True)[0]
                spikes     = numpy.zeros(chunk_size, dtype=numpy.bool)
                spikes[sidx[numpy.random.rand(len(sidx)) < corrcoef]] = True
            else:
                spikes     = numpy.random.rand(chunk_size) < rate[idx]/float(sampling_rate)
            spikes[:N_t]   = False
            spikes[-N_t:]  = False
            spikes         = numpy.where(spikes == True)[0]
            n_template     = N_tm_init + idx
            first_flat     = templates[:, :, n_template].T.flatten()
            norm_flat      = numpy.sum(first_flat**2)
            for scount, spike in enumerate(spikes):
                local_chunk[spike-template_shift:spike+template_shift+1, :] += templates[:, :, n_template].T
                amp        = numpy.dot(local_chunk[spike-template_shift:spike+template_shift+1, :].flatten(), first_flat)
                amp       /= norm_flat
                result['real_amps']  += [amp]
                result['spiketimes'] += [spike + offset]
                result['amplitudes'] += [(1, 0)]
                result['templates']  += [n_template]
                result['voltages']   += [local_chunk[spike, best_elecs[idx]]]

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
        local_chunk /= gain
        local_chunk += dtype_offset
        local_chunk  = local_chunk.astype(data_dtype)
        new_chunk    = numpy.zeros((chunk_size, N_total), dtype=data_dtype)
        new_chunk[:, nodes] = local_chunk

        new_chunk   = new_chunk.reshape(local_shape * N_total)
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

    file_params = file_name + '.params'

    if comm.rank == 0:

        if not os.path.exists(file_name + '/injected'):
            os.makedirs(file_name + '/injected')

        shutil.copy2(params.get('data', 'data_file_noext') + '.params', file_params)

        for name in ['.basis.npz', '.thresholds.npy', '.limits.mat', '.whitening.mat']:
            shutil.copy2(params.get('data', 'file_out') + name, file_name+'/injected/'+file_name.split('/')[-1]+name)

        hdf5storage.savemat(file_out + '.templates', {'templates' : templates})

    comm.Barrier()
    if comm.rank == 0:
        io.collect_data(comm.size, io.load_parameters(file_params), erase=True, with_real_amps=True, with_voltages=True)
        io.change_flag(file_name + '.raw', 'temporal', 'False')
        io.change_flag(file_name + '.raw', 'spatial', 'False')
        shutil.move(file_out + '.templates.mat', file_name+'/injected/templates.mat')
        shutil.move(file_out + '.spiketimes.mat', file_name+'/injected/spiketimes.mat')
        shutil.move(file_out + '.real_amps.mat', file_name+'/injected/real_amps.mat')
        shutil.move(file_out + '.amplitudes.mat', file_name+'/injected/amplitudes.mat')
        shutil.move(file_out + '.voltages.mat', file_name+'/injected/voltages.mat')
        data = hdf5storage.loadmat(file_name+'/injected/'+file_name.split('/')[-1]+'.limits.mat')['limits']
        for count, cell in enumerate(cells):
            data = numpy.vstack((data, data[cell]))
        hdf5storage.savemat(file_name+'/injected/'+file_name.split('/')[-1]+'.limits.mat', {'limits' : data})
        numpy.save(file_name+'/injected/'+file_name.split('/')[-1]+'.scalings', scalings)
        numpy.save(file_name+'/injected/elecs', best_elecs)

        file_name_noext = file_name.split('/')[-1]

        for name in ['.basis.npz', '.thresholds.npy', '.limits.mat', '.whitening.mat']:
            shutil.copy2(file_name+'/injected/'+file_name.split('/')[-1]+name, file_name+'/'+file_name_noext+ name)

        if benchmark in ['fitting', 'synchrony']:
            os.system('cp %s %s' %(file_name+'/injected/templates.mat', file_name+'/'+file_name_noext+'.templates.mat'))
