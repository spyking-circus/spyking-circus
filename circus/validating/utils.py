import matplotlib.pyplot as plt
from scipy import sparse

from ..shared.utils import *



def get_neighbors(params, chan=43, radius=120):
    if radius is None:
        pass
    else:
        radius = 120 # um
        _ = params.set('data', 'radius', str(radius))
    N_total = params.getint('data', 'N_total')
    nodes, edges = io.get_nodes_and_edges(params)
    if chan is None:
        # Select all the channels.
        chans = nodes
    else:
        # Select only the neighboring channels of the best channel.
        inv_nodes = numpy.zeros(N_total, dtype=numpy.int32)
        inv_nodes[nodes] = numpy.argsort(nodes)
        chans = inv_nodes[edges[nodes[chan]]]
    return chans



def extract_juxta_spikes(params):
    # TODO: replace with a routine on the juxta cellular trace.
    # TODO: rewrite 'io.load_data'.
    spike_times_gt, _ = io.load_data(params, 'triggers')
    return spike_times_gt



def extract_extra_thresholds(params):
    
    # TODO: test (i.e. check validity of the computation of the standard deviations).
    
    data_filename = params.get('data', 'data_file')
    data_dtype = params.get('data', 'data_dtype')
    chunk_size = params.getint('data', 'chunk_size')
    # chunk_size = params.getint('whitening', 'chunk_size')
    N_total = params.getint('data', 'N_total')
    
    mpi_file = MPI.File()
    mpi_input = mpi_file.Open(comm, data_filename, MPI.MODE_RDONLY)
    _, nb_chunks, chunk_len, last_chunk_len = io.analyze_data(params, chunk_size)
    
    def extract(data_len, gidx):
        loc_chunk = numpy.zeros(data_len, dtype=data_dtype)
        mpi_input.Read_at(gidx * chunk_len, loc_chunk)
        loc_shape = chunk_size
        loc_chunk = loc_chunk.reshape(loc_shape, N_total)
        mean = numpy.mean(loc_chunk, axis=0)
        var = numpy.mean(numpy.power(loc_chunk - mean, 2), axis=0)
        return mean, var
    
    all_chunks = numpy.arange(nb_chunks)
    loc_indices = numpy.arange(comm.rank, nb_chunks, comm.size)
    loc_all_chunks = all_chunks[loc_indices]
    loc_nb_chunks = len(loc_all_chunks)
    
    if comm.rank == 0:
        io.print_and_log(["Extract extracellular thresholds"], level='info', logger=params)
    
    if comm.rank == 0:
        pbar = get_progressbar(loc_nb_chunks)
    
    data_len = chunk_len
    means = numpy.zeros((N_total, loc_nb_chunks))
    vars = numpy.zeros((N_total, loc_nb_chunks))
    # For each chunk attributed to the current CPU.
    for count, gidx in enumerate(loc_all_chunks):
        means[:, count], vars[:, count] = extract(data_len, gidx)
        if comm.rank == 0:
            pbar.update(count)
    mean = numpy.mean(means, axis=1)
    var = numpy.std(vars, axis=1)
    
    comm.Barrier()
    
    loc_nbs_chunks = comm.gather(loc_nb_chunks, root=0)
    means = comm.gather(mean, root=0)
    vars = comm.gather(var, root=0)
    
    def weighted_mean(weights, values):
        norm_weights = [float(weight) / float(sum(weights)) for weight in weights]
        weighted_values = [norm_weight * value for (norm_weight, value) in zip(norm_weights, values)]
        weighted_mean = sum(weighted_values)
        return weighted_mean
    
    if comm.rank == 0:
        mean = weighted_mean(loc_nbs_chunks, means)
        var = weighted_mean(loc_nbs_chunks, vars)
    
    if comm.rank == 0:
        if last_chunk_len > 0:
            data_len = last_chunk_len
            gidx = nb_chunks
            chunk_size = last_chunk_len // N_total
            last_mean, last_var = extract(data_len, gidx)
            mean = (float(nb_chunks * chunk_len) * mean + float(last_chunk_len) * last_mean) \
                   / float(nb_chunks * chunk_len + last_chunk_len)
            var = (float(nb_chunks * chunk_len) * var + float(last_chunk_len) * last_var) \
                  / float(nb_chunks * chunk_len + last_chunk_len)
        n = nb_chunks * chunk_len + last_chunk_len
        std = numpy.sqrt(var * (float(n) / float(n - 1)))
    else:
        std = None
    
    if comm.rank == 0:
        pbar.finish()
    
    comm.Barrier()
    
    # Broadcast standard deviation to each CPU.
    std = comm.bcast(std, root=0)
    
    return mean, std



def clean_excess(time, channel, value, loc_shape, N_total, safety_time, pos=None):
    
    index = numpy.argsort(value)
    
    x = sparse.coo_matrix((value, (time, channel)), shape=(loc_shape, N_total))
    x = sparse.csr_matrix(x)
    
    mask = numpy.zeros(index.size, dtype=bool)
    for i in xrange(0, index.size):
        t = time[index[i]]
        c = channel[index[i]]
        # TODO: improve time range computation
        time_min = max(0, t - safety_time)
        time_max = min(loc_shape, t + safety_time + 1)
        time_range = xrange(time_min, time_max)
        # TODO: improve space range computation
        # channel_range = xrange(c - w, c + w  + 1)
        channel_range = xrange(c, c + 1)
        xw = x[time_range, channel_range]
        if numpy.count_nonzero(xw.data) == 1:
            mask[i] = True
        else:
            x[t, c] = 0
    
    time = time[index[mask]]
    channel = channel[index[mask]]
    if pos is not None:
        pos  = pos[index[mask]]
    
    if pos is None:
        return time, channel
    else:
        return time, channel, pos



def extract_extra_spikes_(params):
    
    # TODO: change the 'analyze_data' call to take borders into account.
    # TODO: support 'safety_space'.
    
    
    sampling_rate = params.getint('data', 'sampling_rate')
    safety_time = params.getfloat('clustering', 'safety_time')
    safety_space = params.get('clustering', 'safety_space')
    data_filename = params.get('data', 'data_file')
    data_dtype = params.get('data', 'data_dtype')
    chunk_size = params.getint('data', 'chunk_size')
    # chunk_size = params.getint('whitening', 'chunk_size')
    N_total = params.getint('data', 'N_total')
    file_out_suff  = params.get('data', 'file_out_suff')
    
    
    mpi_file = MPI.File()
    mpi_input = mpi_file.Open(comm, data_filename, MPI.MODE_RDONLY)
    _, nb_chunks, chunk_len, last_chunk_len = io.analyze_data(params, chunk_size)
    
    # Convert 'safety_time' from milliseconds to number of samples.
    safety_time = int(safety_time * float(sampling_rate) * 1e-3)
    
    extra_means, extra_stds = extract_extra_thresholds(params)
    
    # Select only the neighboring channels of the best channel.
    chan = params.getint('validating', 'val_chan')
    chans = get_neighbors(params, chan=chan)
    
    def extract_chunk_spikes(data_len, gidx, k=6.0):
        
        loc_chunk = numpy.zeros(data_len, dtype=data_dtype)
        mpi_input.Read_at(gidx * chunk_len, loc_chunk)
        loc_shape = chunk_size
        loc_chunk = loc_chunk.reshape(loc_shape, N_total)
        
        chunk_excess = [] # (time, channel, value) to localized "above threshold" events
        chunk_spikes = [] # (time, channel) to localized spikes
        
        # Find (time, channel, value) couples which locates "above thresholds" events.
        loc_chunk = loc_chunk - extra_means
        loc_mask = (loc_chunk < - k * extra_stds)
        
        time, channel = numpy.where(loc_mask)
        value = loc_chunk + k * extra_stds
        value = - value[loc_mask]
        
        # Filter channels.
        mask = np.array([c in chans for c in channel])
        time = time[mask]
        channel = channel[mask]
        value = value[mask]
        
        # Clean excess.
        time, channel = clean_excess(time, channel, value, loc_shape, N_total, safety_time)
        
        
        # chunk_excess = numpy.vstack((time, channel, value))
        # # or
        # #chunk_excess = zip(list(time), list(channel), list(value))
        
        # index = numpy.argsort(value)
        
        # x = sparse.coo_matrix((index, (time, channel)), shape=(loc_shape, N_total))
        # x = sparse.csr_matrix(x)
        
        # mask = numpy.zeros(index.size, dtype=bool)
        # for i in xrange(0, index.size):
        #     t = time[index[i]]
        #     c = channel[index[i]]
        #     # TODO: improve time range computation
        #     time_min = max(0, t - safety_time)
        #     time_max = min(loc_shape, t + safety_time + 1)
        #     time_range = xrange(time_min, time_max)
        #     # TODO: improve space range comutation
        #     # channel_range = xrange(c - w, c + w  + 1)
        #     channel_range = xrange(c, c + 1)
        #     xw = x[time_range, channel_range]
        #     if xw.nnz == 0:
        #         mask[i] = True
        #     else:
        #         x[t, c] = 0
        #         x.eliminate_zeros()
        #     # print(xw.nnz)
        #     if not mask[i]:
        #         print("{} {} {}".format(t, c, mask[i]))
        
        return time, channel
    
    all_chunks = numpy.arange(nb_chunks)
    loc_indices = numpy.arange(comm.rank, nb_chunks, comm.size)
    loc_all_chunks = all_chunks[loc_indices]
    loc_nb_chunks = len(loc_all_chunks)
    
    if comm.rank == 0:
        io.print_and_log(["Extract extracellular spikes"], level='info', logger=params)
    
    if comm.rank == 0:
        pbar = get_progressbar(loc_nb_chunks)
    
    data_len = chunk_len
    
    k = 8.0
    
    ##### TODO: remove test zone (i.e. plots of extracellular spike times).
    test = False
    
    if test:
        count = 0
        gidx = loc_all_chunks[0]
        loc_chunk = numpy.zeros(data_len, dtype=data_dtype)
        mpi_input.Read_at(gidx * chunk_len, loc_chunk)
        loc_shape = chunk_size
        loc_chunk = loc_chunk.reshape(loc_shape, N_total)
        
        loc_chunk = loc_chunk - extra_means
        loc_mask = (loc_chunk < - k * extra_stds)
        
        time, channel = numpy.where(loc_mask)
        value = loc_chunk + k * extra_stds
        value = - value[loc_mask]
        
        pos = numpy.random.rand(time.size) - 0.5
        
        # Filter wrong channels.
        mask = np.array([c not in [0, 2, 3, 5, 21, 22] for c in channel])
        time = time[mask]
        channel = channel[mask]
        value = value[mask]
        pos = pos[mask]
        
        safety_time_bis = 10 * safety_time
        new_time, new_channel, new_pos = clean_excess(time, channel, value, loc_shape, N_total, safety_time_bis, pos=pos)
        
        fig = plt.figure()
        ax = fig.gca()
        # For each first chunk plot one channel per figure.
        for j in xrange(0, loc_chunk.shape[1]):
            ax = fig.gca()
            ax.plot(loc_chunk[:, j])
            ax.plot([0, loc_chunk.shape[0] - 1], 2 * [extra_means[j]], 'k--')
            ax.plot([0, loc_chunk.shape[0] - 1], 2 * [extra_means[j] + extra_stds[j]], 'k--')
            ax.plot([0, loc_chunk.shape[0] - 1], 2 * [extra_means[j] - extra_stds[j]], 'k--')
            ax.plot([0, loc_chunk.shape[0] - 1], 2 * [extra_means[j] + k * extra_stds[j]], 'k--')
            ax.plot([0, loc_chunk.shape[0] - 1], 2 * [extra_means[j] - k * extra_stds[j]], 'k--')
            idx, = numpy.where(channel == j)
            y = - 350.0 * numpy.ones(idx.size) + 100.0 * pos[idx]
            ax.scatter(time[idx], y, c='r')
            new_idx, = numpy.where(new_channel == j)
            new_y = - 250.0 * numpy.ones(new_idx.size) + 100.0 * new_pos[new_idx]
            ax.scatter(new_time[new_idx], new_y, c='g')
            ax.set_xlim(0, loc_chunk.shape[0] - 1)
            ax.set_ylim(- 400.0, 400.0)
            plt.savefig("/tmp/check-{}-{}.png".format(j, comm.rank))
            fig.clear()
        
        sys.exit(0)
    ##### end test zone
    
    # Preallocation
    times = len(loc_all_chunks) * [None]
    channels = len(loc_all_chunks) * [None]
    # For each chunk attributed to the current CPU.
    for (count, gidx) in enumerate(loc_all_chunks):
        time, channel = extract_chunk_spikes(data_len, gidx, k=k)
        times[count] = time + gidx * chunk_len
        channels[count] = channel
        if comm.rank == 0:
            pbar.update(count)
    
    # Concatenate times and channels.
    times = numpy.hstack(times)
    channels = numpy.hstack(channels)
    print("{}: {}".format(comm.rank, times))
    print("{}: {}".format(comm.rank, channels))
    
    if comm.rank == 0:
        pbar.finish()
    
    comm.Barrier()
    
    # Gather times and channels.
    times = gather_array(times, comm, 0, dtype='int64')
    channels = gather_array(channels, comm, 0, dtype='int64')
    if comm.rank == 0:
        print("{}: {}".format(comm.rank, times))
        print("{}: {}".format(comm.rank, channels))
    
    if comm.rank == 0:
        # Sort times and channels according to time.
        idx = numpy.argsort(times)
        times = times[idx]
        channels = channels[idx]
    
    if comm.rank == 0:
        # Save results to files.
        for i in numpy.unique(channels):
            mask = (channels == i)
            triggers = times[mask]
            path = "{}.triggers.{}.npy".format(file_out_suff, i)
            numpy.save(path, triggers)
    
    comm.Barrier()
    
    return



def extract_extra_spikes(params):
    
    extra_done = params.getboolean('noedits', 'extra_done')
    do_extra = params.getboolean('validating', 'extra')
    
    if extra_done:
        if comm.rank == 0:
            msg = "Spike detection for extracellular traces has already been done"
            io.print_and_log([msg], 'info', params)
    elif do_extra:
        extract_extra_spikes_(params)
        # # TODO: uncomment when 'extract_extra_spikes' works properly.
        # if comm.rank == 0:
        #     io.change_flag(filename, 'extra_done', 'True')
        pass
    else:
        # TODO: log a meaningful message.
        pass
    
    return



def extract_juxta_spikes_(params):
    # TODO: complete with Kampff's script.
    assert False
    return



def extract_juxta_spikes(params):
    juxta_done = params.getboolean('noedits', 'juxta_done')
    do_juxta = params.getboolean('validating', 'juxta')
    
    if juxta_done:
        if comm.rank == 0:
            msg = "Spike detection for juxtacellular traces has already been done"
            io.print_and_log([msg], 'info', params)
    elif juxta_done:
        extract_juxta_spikes_(params)
        # # TODO: uncomment when 'extract_juxta_spikes' works properly.
        # if comm.rank == 0:
        #     io.change_flag(filename, 'juxta_done', 'True')
        pass
    else:
        # TODO: log a meaningful message.
        pass
    
    return
