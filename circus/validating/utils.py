import h5py
import matplotlib.pyplot as plt
from scipy import sparse
from scipy import signal


import circus.shared.algorithms as algo
from ..shared.utils import *



def get_neighbors(params, chan=None):
    N_total = params.getint('data', 'N_total')
    nodes, edges = io.get_nodes_and_edges(params, validating=True)
    if chan is None:
        # Select all the channels.
        chans = nodes
    else:
        # Select only the neighboring channels of the best channel.
        inv_nodes = numpy.zeros(N_total, dtype=numpy.int32)
        inv_nodes[nodes] = numpy.argsort(nodes)
        chans = inv_nodes[edges[nodes[chan]]]
    return chans

def load_chunk(params, spike_times, chans=None):
    """Auxiliary function to load spike data given spike times."""
    # Load the parameters of the spike data.
    data_file = params.get('data', 'data_file')
    data_offset = params.getint('data', 'data_offset')
    data_dtype = params.get('data', 'data_dtype')
    chunk_size = params.getint('data', 'chunk_size')
    N_total = params.getint('data', 'N_total')
    N_t = params.getint('data', 'N_t')
    dtype_offset = params.getint('data', 'dtype_offset')
    if chans is None:
        chans, _ = io.get_nodes_and_edges(params)
    N_filt = chans.size
    ## Compute some additional parameters of the spike data.
    N_tr = spike_times.shape[0]
    datablock = numpy.memmap(data_file, offset=data_offset, dtype=data_dtype, mode='r')
    template_shift = int((N_t - 1) / 2)
    ## Load the spike data.
    spikes = numpy.zeros((N_t, N_filt, N_tr))
    for (count, idx) in enumerate(spike_times):
        chunk_len = chunk_size * N_total
        chunk_start = (idx - template_shift) * N_total
        chunk_end = (idx + template_shift + 1)  * N_total
        local_chunk = datablock[chunk_start:chunk_end]
        # Reshape, slice and cast data.
        local_chunk = local_chunk.reshape(N_t, N_total)
        local_chunk = local_chunk[:, chans]
        local_chunk = local_chunk.astype(numpy.float32)
        local_chunk -= dtype_offset
        # Save data.
        spikes[:, :, count] = local_chunk
    return spikes



# Extracellular ################################################################

def extract_extra_thresholds(params):
    """Compute the mean and the standard deviation for each extracellular channel"""
    
    data_filename = params.get('data', 'data_file')
    data_dtype = params.get('data', 'data_dtype')
    chunk_size = params.getint('data', 'chunk_size')
    # chunk_size = params.getint('whitening', 'chunk_size')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    N_total = params.getint('data', 'N_total')
    
    if do_spatial_whitening:
        spatial_whitening  = io.load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = io.load_data(params, 'temporal_whitening')
    
    mpi_file = MPI.File()
    mpi_input = mpi_file.Open(comm, data_filename, MPI.MODE_RDONLY)
    _, nb_chunks, chunk_len, last_chunk_len = io.analyze_data(params, chunk_size)
    nodes, _ = io.get_nodes_and_edges(params)
    N_elec = nodes.size
    
    def weighted_mean(weights, values):
        """Compute a weighted mean for the given values"""
        norm_weights = [float(weight) / float(sum(weights)) for weight in weights]
        weighted_values = [norm_weight * value for (norm_weight, value) in zip(norm_weights, values)]
        weighted_mean = sum(weighted_values)
        return weighted_mean
    
    def extract_median(chunk_len, chunk_size, gidx):
        """Extract the medians from a chunk of extracellular traces"""
        loc_chunk, loc_shape = io.load_chunk(params, gidx, chunk_len, chunk_size, nodes=nodes)
        # Whiten signal.
        if do_spatial_whitening:
            loc_chunk = numpy.dot(loc_chunk, spatial_whitening)
        if do_temporal_whitening:
            loc_chunk = scipy.ndimage.filters.convolve1d(loc_chunk, temporal_whitening, axis=0, mode='constant')
        median = numpy.median(loc_chunk, axis=0)
        return median
    
    def extract_median_absolute_deviation(chunk_len, chunk_size, gidx, median):
        """Extract the median absolute deviations from a chunk of extracellular traces"""
        loc_chunk, loc_shape = io.load_chunk(params, gidx, chunk_len, chunk_size, nodes=nodes)
        # Whiten signal.
        if do_spatial_whitening:
            loc_chunk = numpy.dot(loc_chunk, spatial_whitening)
        if do_temporal_whitening:
            loc_chunk = scipy.ndimage.filters.convolve1d(loc_chunk, temporal_whitening, axis=0, mode='constant')
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
    
    medians = numpy.zeros((N_elec, loc_nb_chunks))
    
    # For each chunk attributed to the current CPU.
    for count, gidx in enumerate(loc_all_chunks):
        medians[:, count] = extract_median(chunk_len, chunk_size, gidx)
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
            gidx = nb_chunks
            last_chunk_size = last_chunk_len // N_total
            last_median = extract_median(last_chunk_len, last_chunk_size, gidx)
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
    
    mads = numpy.zeros((N_elec, loc_nb_chunks))
    
    # For each chunk attributed to the current CPU.
    for count, gidx in enumerate(loc_all_chunks):
        mads[:, count] = extract_median_absolute_deviation(chunk_len, chunk_size, gidx, median)
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
            gidx = nb_chunks
            last_chunk_size = last_chunk_len // N_total
            last_mad = extract_median_absolute_deviation(last_chunk_len, last_chunk_size, gidx, median)
            mad = (float(nb_chunks * chunk_len) * mad + float(last_chunk_len) * last_mad) \
                  / float(nb_chunks * chunk_len + last_chunk_len)
    
    if comm.rank == 0:
        pbar.finish()
    
    # Broadcast median absolute deviation to each CPU.
    mad = comm.bcast(mad, root=0)
    
    comm.Barrier()
    
    return median, mad



def plot_extracted_extra_spikes(loc_all_chunks, data_len, mpi_input, data_dtype,
                                chunk_len, chunk_size, N_total, nodes,
                                extra_means, extra_stds, k, params, safety_space,
                                safety_time):
    """Temporary function to see if the computed thresholds for a given dataset are valid"""
    
    count = 0
    gidx = loc_all_chunks[0]
    
    loc_chunk, loc_shape = io.load_chunk(params, gidx, chunk_len, chunk_size, nodes=nodes)
    
    sampling_rate = params.getint('data', 'sampling_rate')
    dist_peaks     = params.getint('data', 'dist_peaks')
    skip_artefact  = params.getboolean('data', 'skip_artefact')
    template_shift = params.getint('data', 'template_shift')
    alignment      = params.getboolean('data', 'alignment')
    nodes, _ = io.get_nodes_and_edges(params)
    N_elec = nodes.size
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    if do_spatial_whitening:
        spatial_whitening  = io.load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = io.load_data(params, 'temporal_whitening')
    loc_chunk = numpy.zeros(data_len, dtype=data_dtype)
    mpi_input.Read_at(gidx * chunk_len, loc_chunk)
    loc_shape = chunk_size
    loc_chunk = loc_chunk.reshape(loc_shape, N_total)
    # Consider only the valid channels.
    loc_chunk = loc_chunk[:, nodes]
    # extra_means_ = extra_means[nodes]
    # extra_stds_ = extra_stds[nodes]
    extra_means_ = extra_means
    extra_stds_ = extra_stds
    # Whiten signal.
    if do_spatial_whitening:
        loc_chunk = numpy.dot(loc_chunk, spatial_whitening)
    if do_temporal_whitening:
        loc_chunk = scipy.ndimage.filters.convolve1d(loc_chunk, temporal_whitening, axis=0, mode='constant')
    # Preallocation for results.
    peak_times = N_elec * [None]
    peak_channels = N_elec * [None]
    # For each electrode.
    for e in xrange(0, N_elec):
        # Extract the peaks of the current chunk.
        threshold = k * extra_stds_[e]
        peak_times[e] = algo.detect_peaks(loc_chunk[:, e], threshold, valley=True, mpd=dist_peaks)
        peak_channels[e] = e * numpy.ones(peak_times[e].size, dtype='int')
        if skip_artefact:
            # Remove strong artifacts.
            peak_values = loc_chunk[peak_times[e], e]
            peak_indices = numpy.where(-10.0 * threshold <= peak_values)[0]
            peak_times[e] = peak_times[e][peak_indices]
    peak_times = numpy.concatenate(peak_times)
    peak_channels = numpy.concatenate(peak_channels)
    # Remove the useless borders.
    if alignment:
        loc_borders = (2 * template_shift, loc_shape - 2 * template_shift)
    else:
        loc_borders = (template_shift, loc_shape - template_shift)
    peak_flags = (loc_borders[0] <= peak_times) & (peak_times < loc_borders[1])
    peak_times = peak_times[peak_flags]
    peak_channels = peak_channels[peak_flags]
    # Filter unique peak times.
    loc_peak_times = numpy.unique(peak_times)
    n_times = len(loc_peak_times)
    loc_peak_flags = numpy.zeros(n_times, dtype='bool')
    loc_peak_elecs = numpy.zeros(n_times, dtype='int')
    if 0 < len(loc_peak_times):
        diff_times = loc_peak_times[-1] - loc_peak_times[0]
        all_times = numpy.zeros((N_elec, diff_times + 1), dtype='bool')
        min_times = numpy.maximum(loc_peak_times - loc_peak_times[0] - safety_time, 0)
        max_times = numpy.minimum(loc_peak_times - loc_peak_times[0] + safety_time + 1, diff_times)
        
        ##### TODO: remove temporary zone
        numpy.random.seed(42)
        ##### end temporary zone
        
        argmax_peak = numpy.random.permutation(numpy.arange(n_times))
        all_indices = loc_peak_times[argmax_peak]
        # Select peaks with spatio-temporal masks.
        for peak_index, peak_time in zip(argmax_peak, all_indices):
            # Select electrode showing lowest amplitude.
            elec = numpy.argmin(loc_chunk[peak_time, :])
            neighs = get_neighbors(params, chan=elec)
            if safety_space:
                mslice = all_times[neighs, min_times[peak_index]:max_times[peak_index]]
            else:
                mslice = all_times[elec, min_times[peak_index]:max_times[peak_index]]
            is_local_min = (elec in peak_channels[peak_times == peak_time])
            if is_local_min and not mslice.any():
                loc_peak_flags[peak_index] = True
                loc_peak_elecs[peak_index] = elec
                if safety_space:
                    all_times[neighs, min_times[peak_index]:max_times[peak_index]] = True
                    # all_times[elec, min_times[peak_index]:max_times[peak_index]] = True
                else:
                    all_times[elec, min_times[peak_index]:max_times[peak_index]] = True
    loc_peak_times = loc_peak_times[loc_peak_flags]
    loc_peak_elecs = loc_peak_elecs[loc_peak_flags]
    
    time = loc_peak_times
    channel = loc_peak_elecs
    
    pos = numpy.random.rand(time.size) - 0.5    

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
        # y = + 250.0 * numpy.ones(idx.size) + 100.0 * pos[idx]
        y = - 9.5 * numpy.ones(idx.size) + 1.0 * pos[idx]
        ax.scatter(time[idx], y, c='r')
        # new_idx, = numpy.where(new_channel == j)
        # new_y = - 350.0 * numpy.ones(new_idx.size) + 100.0 * new_pos[new_idx]
        # ax.scatter(new_time[new_idx], new_y, c='g')
        ax.set_xlim(0, loc_chunk.shape[0] - 1)
        # ax.set_ylim(- 400.0, 400.0)
        ax.set_ylim(-10.0, 10.0)
        plt.savefig("/tmp/check-{}-{}.png".format(j, comm.rank))
        fig.clear()
    
    return



def extract_extra_spikes_(params):
    """Detect spikes from the extracellular traces"""
    
    sampling_rate = params.getint('data', 'sampling_rate')
    dist_peaks     = params.getint('data', 'dist_peaks')
    skip_artefact  = params.getboolean('data', 'skip_artefact')
    template_shift = params.getint('data', 'template_shift')
    alignment      = params.getboolean('data', 'alignment')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    safety_time = params.getfloat('validating', 'safety_time')
    safety_space = params.getboolean('validating', 'safety_space')
    data_filename = params.get('data', 'data_file')
    data_dtype = params.get('data', 'data_dtype')
    chunk_size = params.getint('data', 'chunk_size')
    # chunk_size = params.getint('whitening', 'chunk_size')
    N_total = params.getint('data', 'N_total')
    file_out_suff  = params.get('data', 'file_out_suff')
    
    if do_spatial_whitening:
        spatial_whitening  = io.load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = io.load_data(params, 'temporal_whitening')
    
    mpi_file = MPI.File()
    mpi_input = mpi_file.Open(comm, data_filename, MPI.MODE_RDONLY)
    _, nb_chunks, chunk_len, last_chunk_len = io.analyze_data(params, chunk_size)
    nodes, _ = io.get_nodes_and_edges(params)
    N_elec = nodes.size
    
    # Convert 'safety_time' from milliseconds to number of samples.
    safety_time = int(safety_time * float(sampling_rate) * 1e-3)
    
    extra_medians, extra_mads = extract_extra_thresholds(params)
    
    if comm.rank == 0:
        # Save medians and median absolute deviations to BEER file.
        path = "{}.beer.hdf5".format(file_out_suff)
        beer_file = h5py.File(path, 'a', libver='latest')
        ## Save medians.
        extra_medians_key = "extra_medians"
        if extra_medians_key in beer_file.keys():
            del beer_file[extra_medians_key]
        beer_file.create_dataset(extra_medians_key, data=extra_medians)
        ## Save median absolute deviations.
        extra_mads_key = "extra_mads"
        if extra_mads_key in beer_file.keys():
            del beer_file[extra_mads_key]
        beer_file.create_dataset(extra_mads_key, data=extra_mads)
        beer_file.close()
    
    def extract_chunk_spikes(data_len, gidx, k=6.0):
        """Detect spikes from a chunk of the extracellular traces"""
        
        loc_chunk, loc_shape = io.load_chunk(params, gidx, chunk_len, chunk_size, nodes=nodes)
        # Whiten signal.
        if do_spatial_whitening:
            loc_chunk = numpy.dot(loc_chunk, spatial_whitening)
        if do_temporal_whitening:
            loc_chunk = scipy.ndimage.filters.convolve1d(loc_chunk, temporal_whitening,
                                                         axis=0, mode='constant')
        
        ##### TODO: uncomment or remove temporary zone
        # # For each electrode, center traces by removing the medians.
        # extra_medians = numpy.median(loc_chunk, axis=0)
        # loc_chunk = loc_chunk - extra_medians
        ##### end temporary zone
        
        # Preallocation for results.
        peak_times = N_elec * [None]
        peak_channels = N_elec * [None]
        # For each electrode.
        for e in xrange(0, N_elec):
            # Extract the peaks of the current chunk.
            threshold = k * extra_mads[e]
            peak_times[e] = algo.detect_peaks(loc_chunk[:, e], threshold, valley=True, mpd=dist_peaks)
            peak_channels[e] = e * numpy.ones(peak_times[e].size, dtype='int')
            if skip_artefact:
                # Remove strong artifacts.
                peak_values = loc_chunk[peak_times[e], e]
                peak_indices = numpy.where(-10.0 * threshold <= peak_values)[0]
                peak_times[e] = peak_times[e][peak_indices]
        peak_times = numpy.concatenate(peak_times)
        peak_channels = numpy.concatenate(peak_channels)
        # Remove the useless borders.
        if alignment:
            loc_borders = (2 * template_shift, loc_shape - 2 * template_shift)
        else:
            loc_borders = (template_shift, loc_shape - template_shift)
        peak_flags = (loc_borders[0] <= peak_times) & (peak_times < loc_borders[1])
        peak_times = peak_times[peak_flags]
        peak_channels = peak_channels[peak_flags]
        # Filter unique peak times.
        loc_peak_times = numpy.unique(peak_times)
        n_times = len(loc_peak_times)
        loc_peak_flags = numpy.zeros(n_times, dtype='bool')
        loc_peak_elecs = numpy.zeros(n_times, dtype='int')
        if 0 < len(loc_peak_times):
            diff_times = loc_peak_times[-1] - loc_peak_times[0]
            all_times = numpy.zeros((N_elec, diff_times + 1), dtype='bool')
            min_times = numpy.maximum(loc_peak_times - loc_peak_times[0] - safety_time, 0)
            max_times = numpy.minimum(loc_peak_times - loc_peak_times[0] + safety_time + 1, diff_times)
            # Shuffle peaks.
            argmax_peak = numpy.random.permutation(numpy.arange(n_times))
            all_indices = loc_peak_times[argmax_peak]
            # Select peaks with spatio-temporal masks.
            for peak_index, peak_time in zip(argmax_peak, all_indices):
                # Select electrode showing lowest amplitude.
                elec = numpy.argmin(loc_chunk[peak_time, :])
                neighs = get_neighbors(params, chan=elec)
                if safety_space:
                    mslice = all_times[neighs, min_times[peak_index]:max_times[peak_index]]
                else:
                    mslice = all_times[elec, min_times[peak_index]:max_times[peak_index]]
                is_local_min = (elec in peak_channels[peak_times == peak_time])
                if is_local_min and not mslice.any():
                    loc_peak_flags[peak_index] = True
                    loc_peak_elecs[peak_index] = elec
                    if safety_space:
                        all_times[neighs, min_times[peak_index]:max_times[peak_index]] = True
                        # all_times[elec, min_times[peak_index]:max_times[peak_index]] = True
                    else:
                        all_times[elec, min_times[peak_index]:max_times[peak_index]] = True
        loc_peak_times = loc_peak_times[loc_peak_flags]
        loc_peak_elecs = loc_peak_elecs[loc_peak_flags]
        
        return loc_peak_times, loc_peak_elecs
    
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
    
    k = 6.0
    
    
    ##### TODO: remove test zone (i.e. plots of extracellular spike times).
    # plot_extracted_extra_spikes(loc_all_chunks, data_len, mpi_input, data_dtype,
    #                             chunk_len, chunk_size, N_total, nodes,
    #                             extra_medians, extra_mads, k, params, safety_space,
    #                             safety_time)
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
        path = "{}.beer.hdf5".format(file_out_suff)
        beer_file = h5py.File(path, 'a', libver='latest')
        group_name = "extra_spiketimes"
        if group_name in beer_file.keys():
            del beer_file[group_name]
        beer_file.create_group(group_name)
        for i in numpy.arange(0, N_elec):
            mask = (channels == i)
            triggers = times[mask]
            beer_file.create_dataset("{}/elec_{}".format(group_name, i), data=triggers)
        beer_file.close()
    
    comm.Barrier()
    
    return
    
    

def extract_extra_spikes(filename, params):
    extra_done = params.getboolean('noedits', 'extra_done')
    do_extra = params.getboolean('validating', 'extra')
    
    if extra_done:
        if comm.rank == 0:
            msg = [
                "Spike detection for extracellular traces has already been done"
            ]
            io.print_and_log(msg, 'info', params)
    elif do_extra:
        extract_extra_spikes_(params)
        if comm.rank == 0:
            io.change_flag(filename, 'extra_done', 'True')
    else:
        msg = [
            "Extracellular spike times extraction disabled"
        ]
        io.print_and_log(msg, 'info', params)
    
    return



# Juxtacellular ################################################################

def highpass(data, BUTTER_ORDER=3, sampling_rate=10000, cut_off=500.0):
    Wn = (float(cut_off) / (float(sampling_rate) / 2.0), 0.95)
    b, a = signal.butter(BUTTER_ORDER, Wn, 'pass')
    return signal.filtfilt(b, a, data)


def extract_juxta_spikes_(params):
    '''Detect spikes from the extracellular traces'''
    
    file_out_suff = params.get('data', 'file_out_suff')
    dtype_offset = params.getint('data', 'dtype_offset')
    sampling_rate = params.getint('data', 'sampling_rate')
    dist_peaks = params.getint('data', 'dist_peaks')
    juxta_dtype = params.get('validating', 'juxta_dtype')
    
    juxta_filename = "{}.juxta.dat".format(file_out_suff)
    beer_path = "{}.beer.hdf5".format(file_out_suff)
    
    # Read juxtacellular trace.
    juxta_data = numpy.fromfile(juxta_filename, dtype=juxta_dtype)
    juxta_data = juxta_data.astype(numpy.float32)
    # juxta_data = juxta_data - dtype_offset
    juxta_data = numpy.ascontiguousarray(juxta_data)
    
    # Filter juxtacellular trace.
    juxta_data = highpass(juxta_data, sampling_rate=sampling_rate)
    
    # Compute median and median absolute deviation.
    juxta_median = numpy.median(juxta_data)
    juxta_ad = numpy.abs(juxta_data - juxta_median)
    juxta_mad = numpy.median(juxta_ad, axis=0)
    
    # Save medians and median absolute deviations to BEER file.
    beer_file = h5py.File(beer_path, 'a', libver='latest')
    if "juxta_median" in beer_file.keys():
        del beer_file["juxta_median"]
    beer_file.create_dataset("juxta_median", data=juxta_median)
    if "juxta_mad" in beer_file.keys():
        del beer_file["juxta_mad"]
    beer_file.create_dataset("juxta_mad", data=juxta_mad)
    beer_file.close()

    if comm.rank == 0:
        io.print_and_log(["Extract juxtacellular spikes"], level='info', logger=params)
    
    # Detect juxta spike times.
    k = 6.0
    data = juxta_data - juxta_median
    threshold = k * juxta_mad
    juxta_spike_times = algo.detect_peaks(data, threshold, valley=True, mpd=dist_peaks)
    
    # Save juxta spike times to BEER file.
    beer_file = h5py.File(beer_path, 'a', libver='latest')
    group_name = "juxta_spiketimes"
    if group_name in beer_file.keys():
        del beer_file[group_name]
    beer_file.create_group(group_name)
    key = "{}/elec_0".format(group_name)
    beer_file.create_dataset(key, data=juxta_spike_times)
    beer_file.close()
    
    return



def extract_juxta_spikes(filename, params):
    juxta_done = params.getboolean('noedits', 'juxta_done')
    do_juxta = params.getboolean('validating', 'juxta')
    
    if juxta_done:
        if comm.rank == 0:
            msg = [
                "Spike detection for juxtacellular traces has already been done"
            ]
            io.print_and_log(msg, 'info', params)
    elif do_juxta:
        extract_juxta_spikes_(params)
        if comm.rank == 0:
            io.change_flag(filename, 'juxta_done', 'True')
        pass
    else:
        if comm.rank == 0:
            msg = [
                "Juxtacellular spike times extraction disabled"
            ]
            io.print_and_log(msg, 'info', params)
    
    return
