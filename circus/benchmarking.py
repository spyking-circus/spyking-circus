from .shared.utils import *
import h5py
from circus.shared.probes import get_nodes_and_edges
from circus.shared.parser import CircusParser
from circus.shared.messages import print_and_log, init_logging

def main(params, nb_cpu, nb_gpu, use_gpu, file_name, benchmark, sim_same_elec):
    """
    Useful tool to create synthetic datasets for benchmarking.
    
    Arguments
    ---------
    benchmark : {'fitting', 'clustering', 'synchrony', 'pca-validation', 'smart-search', 'drifts'}
        
    """
    if sim_same_elec is None:
        sim_same_elec = 0.8

    logger         = init_logging(params.logfile)
    logger         = logging.getLogger('circus.benchmarking')

    numpy.random.seed(265)
    file_name      = os.path.abspath(file_name)
    data_path      = os.path.dirname(file_name)
    data_suff, ext = os.path.splitext(os.path.basename(file_name))
    file_out, ext  = os.path.splitext(file_name)

    if ext == '':
        ext = '.dat'
        file_name += ext
    
    if ext != '.dat':
        if comm.rank == 0:
            print_and_log(['Benchmarking produces raw files: select a .dat extension'], 'error', logger)
        sys.exit(1)

    if benchmark not in ['fitting', 'clustering', 'synchrony', 'smart-search', 'drifts']:
        if comm.rank == 0:
            print_and_log(['Benchmark need to be in [fitting, clustering, synchrony, smart-search, drifts]'], 'error', logger)
        sys.exit(1)

    # The extension `.p` or `.pkl` or `.pickle` seems more appropriate than `.pic`.
    # see: http://stackoverflow.com/questions/4530111/python-saving-objects-and-using-pickle-extension-of-filename
    # see: https://wiki.python.org/moin/UsingPickle
    def write_benchmark(filename, benchmark, cells, rates, amplitudes, sampling, probe, trends=None):
        """Save benchmark parameters in a file to remember them."""
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

    # Retrieve some key parameters.
    templates = io.load_data(params, 'templates')
    N_tm = templates.shape[1] / 2
    trends          = None

    # Normalize some variables.
    if benchmark == 'fitting':
        nb_insert       = 25
        n_cells         = numpy.random.random_integers(0, N_tm - 1, nb_insert)
        rate            = nb_insert * [10]
        amplitude       = numpy.linspace(0.5, 5, nb_insert)
    if benchmark == 'clustering':
        n_point         = 5
        n_cells         = numpy.random.random_integers(0, N_tm - 1, n_point ** 2)
        x, y            = numpy.mgrid[0:n_point, 0:n_point]
        rate            = numpy.linspace(0.5, 20, n_point)[x.flatten()]
        amplitude       = numpy.linspace(0.5, 5, n_point)[y.flatten()]
    if benchmark == 'synchrony':
        nb_insert       = 5
        corrcoef        = 0.2
        n_cells         = nb_insert * [numpy.random.random_integers(0, N_tm - 1, 1)[0]]
        rate            = 10. / corrcoef
        amplitude       = 2
    if benchmark == 'pca-validation':
        nb_insert       = 10
        n_cells         = numpy.random.random_integers(0, N_tm - 1, nb_insert)
        rate_min        = 0.5
        rate_max        = 20.0
        rate            = rate_min + (rate_max - rate_min) * numpy.random.random_sample(nb_insert)
        amplitude_min   = 0.5
        amplitude_max   = 5.0
        amplitude       = amplitude_min + (amplitude_max - amplitude_min) * numpy.random.random_sample(nb_insert)
    if benchmark == 'smart-search':
        nb_insert       = 10
        n_cells         = nb_insert*[numpy.random.random_integers(0, templates.shape[1]//2-1, 1)[0]]
        rate            = 1 + 5*numpy.arange(nb_insert)
        amplitude       = 2
    if benchmark == 'drifts':
        n_point         = 5
        n_cells         = numpy.random.random_integers(0, templates.shape[1]//2-1, n_point**2)
        x, y            = numpy.mgrid[0:n_point,0:n_point]
        rate            = 5*numpy.ones(n_point)[x.flatten()]
        amplitude       = numpy.linspace(0.5, 5, n_point)[y.flatten()]
        trends          = numpy.random.randn(n_point**2)

    # Delete the output directory tree if this output directory exists.
    if comm.rank == 0:
        if os.path.exists(file_out):
            shutil.rmtree(file_out)

    # Check and normalize some variables.
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
        rate = [rate] * len(cells)

    if numpy.iterable(amplitude):
        assert len(amplitude) == len(cells), "Should have the same number of amplitudes and cells"
    else:
        amplitude = [amplitude] * len(cells)

    # Retrieve some additional key parameters.
    data_file        = params.get_data_file(source=True)
    import copy
    tmp_params       = copy.deepcopy(data_file_in._params)
    N_e              = params.getint('data', 'N_e')
    N_total          = params.nb_channels
    nodes, edges     = get_nodes_and_edges(params)
    N_t              = params.getint('detection', 'N_t')
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    N_tm_init             = templates.shape[1]//2
    thresholds            = io.load_data(params, 'thresholds')
    limits                = io.load_data(params, 'limits')
    best_elecs            = io.load_data(params, 'electrodes')
    norms                 = io.load_data(params, 'norm-templates')

    # Create output directory if it does not exist.
    if comm.rank == 0:
        if not os.path.exists(file_out):
            os.makedirs(file_out)

    # Save benchmark parameters in a file to remember them.
    if comm.rank == 0:
        write_benchmark(file_out, benchmark, cells, rate, amplitude,
                        params.rate, params.get('data', 'mapping'), trends)

    # Synchronize all the threads/processes.
    comm.Barrier()

    if do_spatial_whitening:
        spatial_whitening  = io.load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = io.load_data(params, 'temporal_whitening')

    # Retrieve some additional key parameters.
    chunk_size     = params.getint('data', 'chunk_size')
    scalings       = []
    
    params.set('data', 'data_file', file_name)

    data_file_out = params.get_data_file(is_empty=True)
    data_file_out.allocate(shape=data_file.shape)
    data_file_in._params = tmp_params

    # Synchronize all the threads/processes.
    comm.Barrier()

    # For each wanted synthesized cell insert a generated template in the set of
    # existing template.
    for gcount, cell_id in enumerate(cells):
        best_elec   = best_elecs[cell_id]
        indices     = inv_nodes[edges[nodes[best_elec]]]
        count       = 0
        new_indices = []
        all_elecs   = numpy.random.permutation(numpy.arange(N_e))
        reference   = templates[:, cell_id].toarray().reshape(N_e, N_t)
        # Initialize the similarity (i.e. default value).
        similarity = 1.0
        # Find the first eligible template for the wanted synthesized cell.
        while len(new_indices) != len(indices) or (similarity > sim_same_elec): 
            similarity  = 0
            if count == len(all_elecs):
                if comm.rank == 0:
                    print_and_log(["No electrode to move template %d (max similarity is %g)" %(cell_id, similarity)], 'error', logger)
                sys.exit(1)
            else:
                # Get the next shuffled electrode.
                n_elec = all_elecs[count]

                if benchmark not in ['synchrony', 'smart-search']:
                    # Process if the shuffled electrode and the nearest electrode
                    # to the synthesized cell are not identical.
                    local_test = n_elec != best_elec
                else:
                    # Process if the shuffled electrode and the nearest electrode
                    # to the synthesized cell are identical.
                    local_test = n_elec == best_elec

                if local_test:
                    # Shuffle the neighboring electrodes whithout modifying
                    # the nearest electrode to the synthesized cell.
                    new_indices = inv_nodes[edges[nodes[n_elec]]]
                    idx = numpy.where(new_indices != best_elec)[0]
                    new_indices[idx] = numpy.random.permutation(new_indices[idx])

                    if len(new_indices) == len(indices):
                        # Shuffle the templates on the neighboring electrodes.
                        new_temp = numpy.zeros(reference.shape,
                                               dtype=numpy.float32)
                        new_temp[new_indices, :] = reference[indices, :]
                        # Compute the scaling factor which normalize the
                        # shuffled template.
                        gmin = new_temp.min()
                        data = numpy.where(new_temp == gmin)
                        scaling = -thresholds[data[0][0]]/gmin
                        for i in xrange(templates.shape[1]//2):
                            match = templates[:, i].toarray().reshape(N_e, N_t)
                            d = numpy.corrcoef(match.flatten(),
                                               scaling * new_temp.flatten())[0, 1]
                            if d > similarity:
                                similarity = d
                else:
                    new_indices = []
            # Go to the next shuffled electrode.
            count += 1

        #if comm.rank == 0:
        #    print "Template", cell_id, "is shuffled from electrode", best_elec, "to", n_elec, "(max similarity is %g)" %similarity

        N_tm           = templates.shape[1]//2
        to_insert      = numpy.zeros(reference.shape, dtype=numpy.float32)
        to_insert[new_indices] = scaling*amplitude[gcount]*templates[:, cell_id].toarray().reshape(N_e, N_t)[indices]
        to_insert2     = numpy.zeros(reference.shape, dtype=numpy.float32)
        to_insert2[new_indices] = scaling*amplitude[gcount]*templates[:, cell_id + N_tm].toarray().reshape(N_e, N_t)[indices]

        ## Insert the selected template.
        
        # Retrieve the number of existing templates in the dataset.
        N_tm           = templates.shape[1] / 2

        # Generate the template of the synthesized cell from the selected
        # template, the target amplitude and the rescaling (i.e. threshold of
        # the target electrode).
        to_insert = numpy.zeros(reference.shape, dtype=numpy.float32)
        to_insert[new_indices] = scaling * amplitude[gcount] * templates[:, cell_id].toarray().reshape(N_e, N_t)[indices]
        to_insert = to_insert.flatten()
        to_insert2 = numpy.zeros(reference.shape, dtype=numpy.float32)
        to_insert2[new_indices] = scaling * amplitude[gcount] * templates[:, cell_id + N_tm].toarray().reshape(N_e, N_t)[indices]
        to_insert2 = to_insert2.flatten()

        # Compute the norm of the generated template.
        mynorm     = numpy.sqrt(numpy.sum(to_insert ** 2) / (N_e * N_t))
        mynorm2    = numpy.sqrt(numpy.sum(to_insert2 ** 2) / (N_e * N_t))

        # Insert the limits of the generated template.
        limits     = numpy.vstack((limits, limits[cell_id]))
        # Insert the best electrode of the generated template.
        best_elecs = numpy.concatenate((best_elecs, [n_elec]))

        # Insert the norm of the generated template (i.e. central component and
        # orthogonal component).
        norms      = numpy.insert(norms, N_tm, mynorm)
        norms      = numpy.insert(norms, 2 * N_tm + 1, mynorm2)
        # Insert the scaling of the generated template.
        scalings  += [scaling]

        # Retrieve the data about the existing templates.
        templates = templates.tocoo()
        xdata     = templates.row
        ydata     = templates.col
        zdata     = templates.data

        # Shift by one the orthogonal components of the existing templates.
        idx       = numpy.where(ydata >= N_tm)[0]
        ydata[idx] += 1

        # Insert the central component of the selected template.
        dx    = to_insert.nonzero()[0].astype(numpy.int32)
        xdata = numpy.concatenate((xdata, dx))
        ydata = numpy.concatenate((ydata, N_tm * numpy.ones(len(dx), dtype=numpy.int32)))
        zdata = numpy.concatenate((zdata, to_insert[dx]))

        # Insert the orthogonal component of the selected template.
        dx    = to_insert2.nonzero()[0].astype(numpy.int32)
        xdata = numpy.concatenate((xdata, dx))
        ydata = numpy.concatenate((ydata, (2 * N_tm + 1) * numpy.ones(len(dx), dtype=numpy.int32)))
        zdata = numpy.concatenate((zdata, to_insert2[dx]))

        # Recontruct the matrix of templates.
        templates = scipy.sparse.csc_matrix((zdata, (xdata, ydata)), shape=(N_e * N_t, 2 * (N_tm + 1)))

    # Remove all the expired data.
    if benchmark == 'pca-validation':
        # Remove all the expired data.
        N_tm_init = 0
        N_tm = templates.shape[1] / 2

        limits = limits[N_tm - nb_insert:, :]
        best_elecs = best_elecs[N_tm - nb_insert:]
        norms = numpy.concatenate((norms[N_tm-nb_insert:N_tm], norms[2*N_tm-nb_insert:2*N_tm]))
        scalings = scalings
        
        templates = templates.tocoo()
        xdata = templates.row
        ydata = templates.col
        zdata = templates.data
        
        idx_cen = numpy.logical_and(N_tm - nb_insert <= ydata, ydata < N_tm)
        idx_cen = numpy.where(idx_cen)[0]
        idx_ort = numpy.logical_and(2 * N_tm - nb_insert <= ydata, ydata < 2 * N_tm)
        idx_ort = numpy.where(idx_ort)[0]
        ydata[idx_cen] = ydata[idx_cen] - (N_tm - nb_insert)
        ydata[idx_ort] = ydata[idx_ort] - 2 * (N_tm - nb_insert)
        idx = numpy.concatenate((idx_cen, idx_ort))
        xdata = xdata[idx]
        ydata = ydata[idx]
        zdata = zdata[idx]
        templates = scipy.sparse.csc_matrix((zdata, (xdata, ydata)), shape=(N_e * N_t, 2 * nb_insert))
        
    # Retrieve the information about the organisation of the chunks of data.
    nb_chunks, last_chunk_len = data_file.analyze(chunk_size)

    # Display informations about the generated benchmark.
    if comm.rank == 0:
        print_and_log(["Generating benchmark data [%s] with %d cells" %(benchmark, n_cells)], 'info', logger)
        purge(file_out, '.data')


    template_shift = params.getint('detection', 'template_shift')
    all_chunks     = numpy.arange(nb_chunks)
    to_process     = all_chunks[numpy.arange(comm.rank, nb_chunks, comm.size)]
    loc_nb_chunks  = len(to_process)
    numpy.random.seed(comm.rank)

    to_explore = xrange(comm.rank, nb_chunks, comm.size)

    # Initialize the progress bar about the generation of the benchmark.
    if comm.rank == 0:
        to_explore = get_tqdm_progressbar(to_explore)

    # Open the file for collective I/O.
    #g = myfile.Open(comm, file_name, MPI.MODE_RDWR)
    #g.Set_view(data_offset, data_mpi, data_mpi)
    data_file_out.open(mode='r+')

    # Open the thread/process' files to collect the results.
    spiketimes_filename = os.path.join(file_out, data_suff + '.spiketimes-%d.data' %comm.rank)
    spiketimes_file = open(spiketimes_filename, 'wb')
    amplitude_filename = os.path.join(file_out, data_suff + '.amplitudes-%d.data' %comm.rank)
    amplitudes_file = open(amplitude_filename, 'wb')
    templates_filename = os.path.join(file_out, data_suff + '.templates-%d.data' %comm.rank)
    templates_file = open(templates_filename, 'wb')
    real_amps_filename = os.path.join(file_out, data_suff + '.real_amps-%d.data' %comm.rank)
    real_amps_file = open(real_amps_filename, 'wb')
    voltages_filename = os.path.join(file_out, data_suff + '.voltages-%d.data' %comm.rank)
    voltages_file = open(voltages_filename, 'wb')

    # For each chunk of data associate to the current thread/process generate
    # the new chunk of data (i.e. with considering the added synthesized cells).
    for count, gidx in enumerate(to_explore):

        #if (last_chunk_len > 0) and (gidx == (nb_chunks - 1)):
        #    chunk_len  = last_chunk_len
        #    chunk_size = last_chunk_len // N_total

        result         = {'spiketimes' : [], 'amplitudes' : [], 
                          'templates' : [], 'real_amps' : [],
                          'voltages' : []}
        offset         = gidx * chunk_size
        local_chunk, t_offset = data_file.get_data(gidx, chunk_size, nodes=nodes)

        if benchmark == 'pca-validation':
            # Clear the current data chunk.
            local_chunk = numpy.zeros(local_chunk.shape, dtype=local_chunk.dtype)

        # Handle whitening if necessary.
        if do_spatial_whitening:
            local_chunk = numpy.dot(local_chunk, spatial_whitening)
        if do_temporal_whitening:
            local_chunk = scipy.ndimage.filters.convolve1d(local_chunk,
                                                           temporal_whitening,
                                                           axis=0,
                                                           mode='constant')

        if benchmark is 'synchrony':
            # Generate some spike indices (i.e. times) at the given rate for
            # 'synchrony' mode. Each synthesized cell will use a subset of this
            # spike times.
            mips = numpy.random.rand(chunk_size) < rate[0] / float(params.rate)

        # For each synthesized cell generate its spike indices (i.e.times) and
        # add them to the dataset.
        for idx in xrange(len(cells)):
            if benchmark is 'synchrony':
                # Choose a subset of the spike indices generated before. The
                # size of this subset is parameterized by the target correlation
                # coefficients.
                sidx       = numpy.where(mips == True)[0]
                spikes     = numpy.zeros(chunk_size, dtype=numpy.bool)
                spikes[sidx[numpy.random.rand(len(sidx)) < corrcoef]] = True
            else:
                # Generate some spike indices at the given rate.
                spikes     = numpy.random.rand(chunk_size) < rate[idx] / float(params.rate)
            if benchmark == 'drifts':
                amplitudes = numpy.ones(len(spikes)) + trends[idx]*((spikes + offset)/(5*60*float(params.rate)))
            else:
                amplitudes = numpy.ones(len(spikes))
            # Padding with `False` to avoid the insertion of partial spikes at
            # the edges of the signal.
            spikes[:N_t]   = False
            spikes[-N_t:]  = False
            # Find the indices of the spike samples.
            spikes         = numpy.where(spikes == True)[0]
            n_template     = N_tm_init + idx
            loc_template   = templates[:, n_template].toarray().reshape(N_e, N_t)
            first_flat     = loc_template.T.flatten()
            norm_flat      = numpy.sum(first_flat ** 2)
            # For each index (i.e. spike sample location) add the spike to the
            # chunk of data.
            refractory     = int(5 * 1e-3 * params.rate)         
            t_last         = - refractory
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

        # Write the results into the thread/process' files.
        spikes_to_write     = numpy.array(result['spiketimes'], dtype=numpy.uint32)
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
        #new_chunk    = numpy.zeros((chunk_size, N_total), dtype=numpy.float32)
        #new_chunk[:, nodes] = local_chunk

        # Overwrite the new chunk of data using explicit offset. 
        #new_chunk   = new_chunk.flatten()
        #g.Write_at(gidx * chunk_len, new_chunk)
        data_file_out.set_data(offset, local_chunk)

        # Update the progress bar about the generation of the benchmark.
        
    # Close the thread/process' files.
    spiketimes_file.flush()
    os.fsync(spiketimes_file.fileno())
    spiketimes_file.close()

    amplitudes_file.flush()
    os.fsync(amplitudes_file.fileno())
    amplitudes_file.close()

    templates_file.flush()
    os.fsync(templates_file.fileno())
    templates_file.close()

    real_amps_file.flush()
    os.fsync(real_amps_file.fileno())
    real_amps_file.close()

    voltages_file.flush()
    os.fsync(voltages_file.fileno())
    voltages_file.close()


    # Close the file for collective I/O.
    data_file_out.close()
    data_file.close()

    
    # Synchronize all the threads/processes.
    comm.Barrier()

    
    ## Eventually, perform all the administrative tasks.
    ## (i.e. files and folders management).

    file_params = file_out + '.params'

    if comm.rank == 0:
        # Create `injected` directory if it does not exist
        result_path = os.path.join(file_out, 'injected') 
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # Copy initial configuration file from `<dataset1>.params` to `<dataset2>.params`.
        shutil.copy2(params.get('data', 'data_file_noext') + '.params', file_params)
        new_params = CircusParser(file_name)
        # Copy initial basis file from `<dataset1>/<dataset1>.basis.hdf5` to
        # `<dataset2>/injected/<dataset2>.basis.hdf5.
        shutil.copy2(params.get('data', 'file_out') + '.basis.hdf5',
                     os.path.join(result_path, data_suff + '.basis.hdf5'))


        # Save templates into `<dataset>/<dataset>.templates.hdf5`.
        mydata = h5py.File(os.path.join(file_out, data_suff + '.templates.hdf5'), 'w')
        templates = templates.tocoo()
        mydata.create_dataset('temp_x', data=templates.row)
        mydata.create_dataset('temp_y', data=templates.col)
        mydata.create_dataset('temp_data', data=templates.data)
        mydata.create_dataset('temp_shape', data=numpy.array([N_e, N_t, templates.shape[1]],
                                                             dtype=numpy.int32))
        mydata.create_dataset('limits', data=limits)
        mydata.create_dataset('norms', data=norms)
        mydata.close()

        # Save electrodes into `<dataset>/<dataset>.clusters.hdf5`.
        mydata = h5py.File(os.path.join(file_out, data_suff + '.clusters.hdf5'), 'w')
        mydata.create_dataset('electrodes', data=best_elecs)
        mydata.close()

    comm.Barrier()
    if comm.rank == 0:
        # Gather data from all threads/processes.
        f_next, extension = os.path.splitext(file_name)
        file_out_bis = os.path.join(f_next, os.path.basename(f_next))
        #new_params.set('data', 'file_out', file_out_bis) # Output file without suffix
        #new_params.set('data', 'file_out_suff', file_out_bis  + params.get('data', 'suffix'))
    
        new_params.get_data_file()
        io.collect_data(comm.size, new_params, erase=True, with_real_amps=True, with_voltages=True, benchmark=True)
        # Change some flags in the configuration file.
        new_params.write('whitening', 'temporal', 'False') # Disable temporal filtering
        new_params.write('whitening', 'spatial', 'False') # Disable spatial filtering
        new_params.write('data', 'data_dtype', 'float32') # Set type of the data to float32
        new_params.write('data', 'dtype_offset', 'auto') # Set padding for data to auto
        # Move results from `<dataset>/<dataset>.result.hdf5` to
        # `<dataset>/injected/<dataset>.result.hdf5`.
        
        shutil.move(os.path.join(file_out, data_suff + '.result.hdf5'), os.path.join(result_path, data_suff + '.result.hdf5'))
                
        # Save scalings into `<dataset>/injected/<dataset>.scalings.npy`.
        numpy.save(os.path.join(result_path, data_suff + '.scalings'), scalings)

        file_name_noext, ext = os.path.splitext(file_name)

        # Copy basis from `<dataset>/injected/<dataset>.basis.hdf5` to
        # `<dataset>/<dataset>.basis.hdf5`.
        shutil.copy2(os.path.join(result_path, data_suff + '.basis.hdf5'),
                     os.path.join(file_out, data_suff + '.basis.hdf5'))

        if benchmark not in ['fitting', 'synchrony']:
            # Copy templates from `<dataset>/<dataset>.templates.hdf5` to
            # `<dataset>/injected/<dataset>.templates.hdf5`
            shutil.move(os.path.join(file_out, data_suff + '.templates.hdf5'),
                        os.path.join(result_path, data_suff + '.templates.hdf5'))
