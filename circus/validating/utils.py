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



def extract_extra_thresholds(params):
    """Compute the mean and the standard deviation for each extracellular channel"""
    
    data_filename = params.get('data', 'data_file')
    data_dtype = params.get('data', 'data_dtype')
    chunk_size = params.getint('data', 'chunk_size')
    # chunk_size = params.getint('whitening', 'chunk_size')
    N_total = params.getint('data', 'N_total')
    
    mpi_file = MPI.File()
    mpi_input = mpi_file.Open(comm, data_filename, MPI.MODE_RDONLY)
    _, nb_chunks, chunk_len, last_chunk_len = io.analyze_data(params, chunk_size)
    
    def weighted_mean(weights, values):
        """Compute a weighted mean for the given values"""
        norm_weights = [float(weight) / float(sum(weights)) for weight in weights]
        weighted_values = [norm_weight * value for (norm_weight, value) in zip(norm_weights, values)]
        weighted_mean = sum(weighted_values)
        return weighted_mean
    
    def extract_median(data_len, gidx):
        """Extract the medians from a chunk of extracellular traces"""
        loc_chunk = numpy.zeros(data_len, dtype=data_dtype)
        mpi_input.Read_at(gidx * chunk_len, loc_chunk)
        loc_shape = chunk_size
        loc_chunk = loc_chunk.reshape(loc_shape, N_total)
        median = numpy.median(loc_chunk, axis=0)
        return median
    
    def extract_median_absolute_deviation(data_len, gidx, median):
        """Extract the median absolute deviations from a chunk of extracellular traces"""
        loc_chunk = numpy.zeros(data_len, dtype=data_dtype)
        mpi_input.Read_at(gidx * chunk_len, loc_chunk)
        loc_shape = chunk_size
        loc_chunk = loc_chunk.reshape(loc_shape, N_total)
        mad = numpy.median(numpy.abs(loc_chunk - median), axis=0)
        return mad
    
    # Distribute chunks over the CPUs.
    all_chunks = numpy.arange(nb_chunks)
    loc_indices = numpy.arange(comm.rank, nb_chunks, comm.size)
    loc_all_chunks = all_chunks[loc_indices]
    loc_nb_chunks = len(loc_all_chunks)
    
    loc_nbs_chunks = comm.gather(loc_nb_chunks, root=0)
    
    if comm.rank == 0:
        io.print_and_log(["Extract extracellular medians"],
                         level='info', logger=params)
    
    if comm.rank == 0:
        pbar = get_progressbar(loc_nb_chunks)
    
    data_len = chunk_len
    medians = numpy.zeros((N_total, loc_nb_chunks))
    
    # For each chunk attributed to the current CPU.
    for count, gidx in enumerate(loc_all_chunks):
        medians[:, count] = extract_median(data_len, gidx)
        if comm.rank == 0:
            pbar.update(count)
    median = numpy.mean(medians, axis=1)
    
    comm.Barrier()
    
    medians = comm.gather(median, root=0)
    
    if comm.rank == 0:
        median = weighted_mean(loc_nbs_chunks, medians)
    
    if comm.rank == 0:
        if last_chunk_len > 0:
            # For last chunk attributed to the first CPU.
            data_len = last_chunk_len
            gidx = nb_chunks
            chunk_size = last_chunk_len // N_total
            last_median = extract_median(data_len, gidx)
            median = (float(nb_chunks * chunk_len) * median + float(last_chunk_len) * last_median) \
                     / float(nb_chunks * chunk_len + last_chunk_len)
    
    if comm.rank == 0:
        pbar.finish()
    
    # Broadcast medians to each CPU.
    median = comm.bcast(median, root=0)
    
    comm.Barrier()
    
    if comm.rank == 0:
        io.print_and_log(["Extract extracellular median absolute deviations"],
                         level='info', logger=params)
    
    if comm.rank == 0:
        pbar = get_progressbar(loc_nb_chunks)
    
    data_len = chunk_len
    mads = numpy.zeros((N_total, loc_nb_chunks))
    
    # For each chunk attributed to the current CPU.
    for count, gidx in enumerate(loc_all_chunks):
        mads[:, count] = extract_median_absolute_deviation(data_len, gidx, median)
        if comm.rank == 0:
            pbar.update(count)
    mad = numpy.mean(mads, axis=1)
    
    comm.Barrier()
    
    mads = comm.gather(mad, root=0)
    
    if comm.rank == 0:
        mad = weighted_mean(loc_nbs_chunks, mads)
    
    if comm.rank == 0:
        if last_chunk_len > 0:
            # For last chunk attributed to the first CPU.
            data_len = last_chunk_len
            gidx = nb_chunks
            chunk_size = last_chunk_len // N_total
            last_mad = extract_median_absolute_deviation(data_len, gidx, median)
            mad = (float(nb_chunks * chunk_len) * mad + float(last_chunk_len) * last_mad) \
                  / float(nb_chunks * chunk_len + last_chunk_len)
    
    if comm.rank == 0:
        pbar.finish()
    
    # Broadcast median absolute deviation to each CPU.
    mad = comm.bcast(mad, root=0)
    
    comm.Barrier()
    
    return median, mad



def clean_excess(params, time, channel, value, loc_shape, N_total, safety_space,
                 safety_time, pos=None):
    """Find spikes among excess for a given forbidden window (temporal and spatial)"""
    
    index = numpy.argsort(value)
    
    x = sparse.coo_matrix((value, (time, channel)), shape=(loc_shape, N_total))
    x = sparse.csr_matrix(x)
    
    mask = numpy.zeros(index.size, dtype=bool)
    # For each excess check if there is a higher excess the temporal and spatial
    # neighborhood.
    for i in xrange(0, index.size):
        t = time[index[i]]
        c = channel[index[i]]
        time_min = max(0, t - safety_time)
        time_max = min(loc_shape, t + safety_time + 1)
        time_range = xrange(time_min, time_max)
        if safety_space:
            channel_range = get_neighbors(params, chan=c)
        else:
            channel_range = [c]
        xw = x[time_range, :]
        xw = xw[:, channel_range]
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



def plot_extracted_extra_spikes(loc_all_chunks, data_len, mpi_input, data_dtype,
                                chunk_len, chunk_size, N_total, nodes,
                                extra_means, extra_stds, k, params, safety_space,
                                safety_time):
    """Temporary function to see if the computed thresholds for a given dataset are valid"""
    
    count = 0
    gidx = loc_all_chunks[0]
    loc_chunk = numpy.zeros(data_len, dtype=data_dtype)
    mpi_input.Read_at(gidx * chunk_len, loc_chunk)
    loc_shape = chunk_size
    loc_chunk = loc_chunk.reshape(loc_shape, N_total)
    
    # Consider the valid channels only.
    loc_chunk = loc_chunk[:, nodes]
    extra_means_ = extra_means[nodes]
    extra_stds_ = extra_stds[nodes]
    
    # Find (time, channel, value) couples which locates "above thresholds" events.
    loc_chunk = loc_chunk - extra_means_
    loc_mask = (loc_chunk < - k * extra_stds_)
    
    time, channel = numpy.where(loc_mask)
    value = loc_chunk + k * extra_stds_
    value = value / (k * extra_stds_) # normalize threshold excess
    value = - value[loc_mask]
    
    pos = numpy.random.rand(time.size) - 0.5
    
    # Clean excess.
    safety_time_bis = 10 * safety_time
    new_time, new_channel, new_pos = clean_excess(params, time, channel, value,
                                                  loc_shape, N_total,
                                                  safety_space, safety_time_bis,
                                                  pos=pos)
    
    fig = plt.figure()
    ax = fig.gca()
    # For each first chunk plot one channel per figure.
    for j in xrange(0, loc_chunk.shape[1]):
        ax = fig.gca()
        ax.plot(loc_chunk[:, j])
        ax.plot([0, loc_chunk.shape[0] - 1], 2 * [extra_means_[j]], 'k--')
        ax.plot([0, loc_chunk.shape[0] - 1], 2 * [extra_means_[j] + extra_stds_[j]], 'k--')
        ax.plot([0, loc_chunk.shape[0] - 1], 2 * [extra_means_[j] - extra_stds_[j]], 'k--')
        ax.plot([0, loc_chunk.shape[0] - 1], 2 * [extra_means_[j] + k * extra_stds_[j]], 'k--')
        ax.plot([0, loc_chunk.shape[0] - 1], 2 * [extra_means_[j] - k * extra_stds_[j]], 'k--')
        idx, = numpy.where(channel == j)
        y = + 350.0 * numpy.ones(idx.size) + 100.0 * pos[idx]
        ax.scatter(time[idx], y, c='r')
        new_idx, = numpy.where(new_channel == j)
        new_y = - 350.0 * numpy.ones(new_idx.size) + 100.0 * new_pos[new_idx]
        ax.scatter(new_time[new_idx], new_y, c='g')
        ax.set_xlim(0, loc_chunk.shape[0] - 1)
        ax.set_ylim(- 400.0, 400.0)
        plt.savefig("/tmp/check-{}-{}.png".format(j, comm.rank))
        fig.clear()
    
    return



def extract_extra_spikes_(params):
    """Detect spikes from the extracellular traces"""
    
    # TODO: change the 'analyze_data' call to take borders into account.
    
    sampling_rate = params.getint('data', 'sampling_rate')
    safety_time = params.getfloat('validating', 'safety_time')
    safety_space = params.get('validating', 'safety_space')
    data_filename = params.get('data', 'data_file')
    data_dtype = params.get('data', 'data_dtype')
    chunk_size = params.getint('data', 'chunk_size')
    # chunk_size = params.getint('whitening', 'chunk_size')
    N_total = params.getint('data', 'N_total')
    file_out_suff  = params.get('data', 'file_out_suff')
    
    
    mpi_file = MPI.File()
    mpi_input = mpi_file.Open(comm, data_filename, MPI.MODE_RDONLY)
    _, nb_chunks, chunk_len, last_chunk_len = io.analyze_data(params, chunk_size)
    nodes, _ = io.get_nodes_and_edges(params)
    N_elec = nodes.size
    
    # Convert 'safety_time' from milliseconds to number of samples.
    safety_time = int(safety_time * float(sampling_rate) * 1e-3)
    
    extra_means, extra_stds = extract_extra_thresholds(params)
    
    def extract_chunk_spikes(data_len, gidx, k=6.0):
        """Detect spikes from a chunk of the extracellular traces"""
        loc_chunk = numpy.zeros(data_len, dtype=data_dtype)
        mpi_input.Read_at(gidx * chunk_len, loc_chunk)
        loc_shape = chunk_size
        loc_chunk = loc_chunk.reshape(loc_shape, N_total)
        # Consider only the valid channels.
        loc_chunk = loc_chunk[:, nodes]
        extra_means_ = extra_means[nodes]
        extra_stds_ = extra_stds[nodes]
        # Find (time, channel, value) couples which locates "above thresholds" events.
        loc_chunk = loc_chunk - extra_means_
        loc_mask = (loc_chunk < - k * extra_stds_)
        time, channel = numpy.where(loc_mask)
        value = loc_chunk + k * extra_stds_
        value = value / (k * extra_stds_) # normalize threshold excess
        value = - value[loc_mask]
        # Clean excess.
        time, channel = clean_excess(params, time, channel, value, loc_shape,
                                     N_total, safety_space, safety_time)
        return time, channel
    
    # Distribute chunks over CPUs.
    all_chunks = numpy.arange(nb_chunks)
    loc_indices = numpy.arange(comm.rank, nb_chunks, comm.size)
    loc_all_chunks = all_chunks[loc_indices]
    loc_nb_chunks = len(loc_all_chunks)
    
    if comm.rank == 0:
        io.print_and_log(["Extract extracellular spikes"], level='info', logger=params)
    
    if comm.rank == 0:
        pbar = get_progressbar(loc_nb_chunks)
    
    data_len = chunk_len
    
    k = 7.0
    
    
    ##### TODO: remove test zone (i.e. plots of extracellular spike times).
    # nodes = range(0, 64) # all the channels
    # nodes = range(4, 48) + range(49, 64) # 'gt.params"'s channels
    # nodes = range(1, 2) + range(4, 5) + range(6, 21) + range(23, 64) # personal choice
    # nodes = range(4, 5) + range(6, 21) + range(23, 64) # personal choice (without juxta ?)
    plot_extracted_extra_spikes(loc_all_chunks, data_len, mpi_input, data_dtype,
                                chunk_len, chunk_size, N_total, nodes,
                                extra_means, extra_stds, k, params, safety_space,
                                safety_time)
    # sys.exit(0)
    ##### end test zone
    
    
    # Preallocation for results.
    times = len(loc_all_chunks) * [None]
    channels = len(loc_all_chunks) * [None]
    
    # For each chunk attributed to the current CPU.
    for (count, gidx) in enumerate(loc_all_chunks):
        time, channel = extract_chunk_spikes(data_len, gidx, k=k)
        times[count] = time + gidx * chunk_size
        channels[count] = channel
        if comm.rank == 0:
            pbar.update(count)
    
    # Concatenate times and channels.
    times = numpy.hstack(times)
    channels = numpy.hstack(channels)
    
    if comm.rank == 0:
        pbar.finish()
    
    comm.Barrier()
    
    # Gather times and channels.
    times = gather_array(times, comm, 0, dtype='int64')
    channels = gather_array(channels, comm, 0, dtype='int64')
    
    if comm.rank == 0:
        # Sort times and channels according to time.
        idx = numpy.argsort(times)
        times = times[idx]
        channels = channels[idx]
    
    if comm.rank == 0:
        msg = [
            "Total number of extracellular spikes extracted: {}".format(channels.size),
        ] + [
            "Number of extracellular spikes extracted on channel {}: {}".format(i, channels[channels == i].size) for i in numpy.unique(channels)
        ]
        io.print_and_log(msg, level='info', logger=params)
    
    if comm.rank == 0:
        # Save results to files.
        for i in numpy.unique(channels):
            mask = (channels == i)
            triggers = times[mask]
            path = "{}.triggers.{}.npy".format(file_out_suff, i)
            numpy.save(path, triggers)
    
    comm.Barrier()
    
    return



def extract_extra_spikes(filename, params):
    
    extra_done = params.getboolean('noedits', 'extra_done')
    do_extra = params.getboolean('validating', 'extra')
    
    if extra_done:
        if comm.rank == 0:
            msg = "Spike detection for extracellular traces has already been done"
            io.print_and_log([msg], 'info', params)
    elif do_extra:
        extract_extra_spikes_(params)
        if comm.rank == 0:
            io.change_flag(filename, 'extra_done', 'True')
    else:
        # TODO: log a meaningful message.
        pass
    
    return



def extract_juxta_spikes_(params):
    # TODO: complete with Kampff's script.
    #       (i.e. for the moment provide the 'triggers.npy' file)
    assert False
    return



def extract_juxta_spikes(filename, params):
    juxta_done = params.getboolean('noedits', 'juxta_done')
    do_juxta = params.getboolean('validating', 'juxta')
    
    if juxta_done:
        if comm.rank == 0:
            msg = "Spike detection for juxtacellular traces has already been done"
            io.print_and_log([msg], 'info', params)
    elif juxta_done:
        extract_juxta_spikes_(params)
        if comm.rank == 0:
            io.change_flag(filename, 'juxta_done', 'True')
        pass
    else:
        # TODO: log a meaningful message.
        pass
    
    return


