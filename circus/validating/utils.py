import h5py
import matplotlib.pyplot as plt
from scipy import sparse
from scipy import signal


import circus.shared.algorithms as algo
from ..shared.utils import *
from ..shared import plot
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



def get_neighbors(params, chan=None):
    N_total = params.getint('data', 'N_total')
    nodes, edges = io.get_nodes_and_edges(params, validating=True)
    inv_nodes = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)
    if chan is None:
        # Select all the channels.
        chans = inv_nodes[nodes]
    else:
        # Select only the neighboring channels of the best channel.
        chans = inv_nodes[edges[nodes[chan]]]
    return nodes, chans

# def load_chunk(params, spike_times, chans=None):
#     """Auxiliary function to load spike data given spike times."""
#     # Load the parameters of the spike data.
#     data_file = params.get('data', 'data_file')
#     data_offset = params.getint('data', 'data_offset')
#     data_dtype = params.get('data', 'data_dtype')
#     chunk_size = params.getint('data', 'chunk_size')
#     alignment  = params.getboolean('detection', 'alignment')
#     N_total = params.getint('data', 'N_total')
#     do_temporal_whitening = params.getboolean('whitening', 'temporal')
#     do_spatial_whitening  = params.getboolean('whitening', 'spatial')
#     template_shift        = params.getint('data', 'template_shift')
#     N_t = params.getint('data', 'N_t')

#     if do_spatial_whitening:
#         spatial_whitening  = io.load_data(params, 'spatial_whitening')
#     if do_temporal_whitening:     
#         temporal_whitening = io.load_data(params, 'temporal_whitening')

#     if alignment:
#         cdata = numpy.linspace(-template_shift, template_shift, 5*N_t)
#         xdata = numpy.arange(-2*template_shift, 2*template_shift+1)

    
#     dtype_offset = params.getint('data', 'dtype_offset')
#     if chans is None:
#         chans, _ = io.get_nodes_and_edges(params)
#     N_filt = chans.size
#     ## Compute some additional parameters of the spike data.
#     N_tr = spike_times.shape[0]
#     datablock = numpy.memmap(data_file, offset=data_offset, dtype=data_dtype, mode='r')
#     template_shift = int((N_t - 1) // 2)
#     ## Load the spike data.
#     spikes = numpy.zeros((N_t, N_filt, N_tr), dtype=numpy.float32)
#     for (count, idx) in enumerate(spike_times):
#         chunk_len = chunk_size * N_total
#         #if alignment:
#         #    chunk_start = (idx - 2*template_shift)*N_total
#         #    chunk_end   = (idx + 2*template_shift+1)*N_total
#         if True:#else:
#             chunk_start = (idx - template_shift) * N_total
#             chunk_end = (idx + template_shift + 1)  * N_total
#         local_chunk = datablock[chunk_start:chunk_end]
#         # Reshape, slice and cast data.
#         #if alignment:
#         #    local_chunk = local_chunk.reshape(2*N_t - 1, N_total)
#         if True:#else:
#             local_chunk = local_chunk.reshape(N_t, N_total)
        
#         #if do_spatial_whitening:
#         #    local_chunk = numpy.dot(local_chunk, spatial_whitening)
#         #if do_temporal_whitening:
#         #    local_chunk = scipy.ndimage.filters.convolve1d(local_chunk, temporal_whitening, axis=0, mode='constant')

#         local_chunk = numpy.take(local_chunk, chans, axis=1)
#         local_chunk = local_chunk.astype(numpy.float32)
#         local_chunk -= dtype_offset

#         # Save data.
#         spikes[:, :, count] = local_chunk
#     return spikes


def get_juxta_stas(params, times_i, labels_i):
    '''Extract STAs from the juxtacellular trace.'''

    file_out_suff = params.get('data', 'file_out_suff')
    sampling_rate = params.getint('data', 'sampling_rate')
    N_t = params.getint('data', 'N_t')
    juxta_dtype = params.get('validating', 'juxta_dtype')
    
    juxta_filename = "{}.juxta.dat".format(file_out_suff)
    beer_path = "{}.beer.hdf5".format(file_out_suff)
    
    # Read juxtacellular trace.
    juxta_data = numpy.fromfile(juxta_filename, dtype=juxta_dtype)
    #juxta_data = juxta_data.astype(numpy.float32)
    # juxta_data = juxta_data - dtype_offset
    juxta_data = numpy.ascontiguousarray(juxta_data)
    
    # Filter juxtacellular trace.
    juxta_data  = highpass(juxta_data, sampling_rate=sampling_rate)
    juxta_data -= numpy.median(juxta_data)

    # Extract STAs.
    stas_shape = (len(times_i), N_t)
    stas = numpy.zeros(stas_shape)
    for i, time in enumerate(times_i):
        imin = time - (N_t - 1) / 2
        imax = time + (N_t - 1) / 2 + 1
        # TODO: check if imin < 0  or juxta_data.size < imax.
        stas[i] = juxta_data[imin:imax]
    
    return stas


def with_quadratic_feature(X_raw, pairwise=False):
    N = X_raw.shape[0]
    K = X_raw.shape[1]
    if pairwise:
        # With pairwise product of feature vector elements.
        M = K + K * (K + 1) // 2
        shape = (N, M)
    else:
        # Without pairwise product of feature vector elments.
        M = K
        shape = (N, M)
    
    # if comm.rank == 0:
    #     print("N, M: {}, {}".format(N, M))
    
    # X = numpy.zero(shape)
    X        = numpy.empty(shape, dtype=numpy.float32)
    X[:, :K] = X_raw
    
    ##### Initial try (~ 0.5s)
    if pairwise:
        # Add the pairwise product of feature vector elements.
        k = 0
        for i in xrange(K):
            for j in xrange(i, K):
                X[:, K + k] = X[:, i] * X[:, j]
                k = k + 1
    
    ##### Second try (~ 0.6s)
    # if pairwise:
    #     # Add the pairwise product of feature vector elements.
    #     k = 0
    #     for i in xrange(0, K):
    #         for j in xrange(i, K):
    #             X[:, K + k] = X[:, [i, j]].prod(axis=1)
    #             k = k + 1
    
    ##### Third try (~ 0.6s)
    # if pairwise:
    #     import itertools
    #     comb = itertools.combinations(range(K), 2)
    #     for k, c in enumerate(comb):
    #         X[:, K + k] = X[:, c].prod(axis=1)
    
    ##### Fourth try (~ 16s)
    # if pairwise:
    #     def func(x_raw):
    #         x = np.empty(M)
    #         x[:K] = x_raw
    #         k = 0
    #         for i in xrange(0, K):
    #             for j in xrange(i, K):
    #                 x[K + k] = x_raw[i] * x_raw[j]
    #                 k = k + 1
    #         return x
    #     X = numpy.apply_along_axis(func, 1, X_raw)
    
    return X



# Extracellular ################################################################

def extract_extra_thresholds(params):
    """Compute the mean and the standard deviation for each extracellular channel"""
    
    data_file      = io.get_data_file(params)
    data_file.open()

    chunk_size = params.getint('data', 'chunk_size')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    N_total = params.getint('data', 'N_total')
    
    if do_spatial_whitening:
        spatial_whitening  = io.load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = io.load_data(params, 'temporal_whitening')
    
    #mpi_file = MPI.File()
    #mpi_input = mpi_file.Open(comm, data_filename, MPI.MODE_RDONLY)
    nb_chunks, last_chunk_len = data_file.analyze(chunk_size)
    nodes, _ = io.get_nodes_and_edges(params)
    N_elec = nodes.size
    
    def weighted_mean(weights, values):
        """Compute a weighted mean for the given values"""
        norm_weights = [float(weight) / float(sum(weights)) for weight in weights]
        weighted_values = [norm_weight * value for (norm_weight, value) in zip(norm_weights, values)]
        weighted_mean = sum(weighted_values)
        return weighted_mean
    
    def extract_median(chunk_size, gidx):
        """Extract the medians from a chunk of extracellular traces"""
        loc_chunk = data_file.get_data(gidx, chunk_size, nodes=nodes)
        # Whiten signal.
        if do_spatial_whitening:
            loc_chunk = numpy.dot(loc_chunk, spatial_whitening)
        if do_temporal_whitening:
            loc_chunk = scipy.ndimage.filters.convolve1d(loc_chunk, temporal_whitening, axis=0, mode='constant')
        median = numpy.median(loc_chunk, axis=0)
        return median
    
    def extract_median_absolute_deviation(chunk_size, gidx, median):
        """Extract the median absolute deviations from a chunk of extracellular traces"""
        loc_chunk = data_file.get_data(gidx, chunk_size, nodes=nodes)
        # Whiten signal.
        if do_spatial_whitening:
            loc_chunk = numpy.dot(loc_chunk, spatial_whitening)
        if do_temporal_whitening:
            loc_chunk = scipy.ndimage.filters.convolve1d(loc_chunk, temporal_whitening, axis=0, mode='constant')
        mad = numpy.median(numpy.abs(loc_chunk - median), axis=0)
        return mad
    
    # Distribute chunks over the CPUs.
    all_chunks = numpy.arange(nb_chunks)
    loc_all_chunks = all_chunks[comm.rank::comm.size]
    loc_nb_chunks = len(loc_all_chunks)
    
    loc_nbs_chunks = comm.gather(loc_nb_chunks, root=0)
    
    if comm.rank == 0:
        io.print_and_log(["Computing extracellular medians..."],
                         level='default', logger=params)
    
    if comm.rank == 0:
        pbar = get_progressbar(loc_nb_chunks)
    
    medians = numpy.zeros((N_elec, loc_nb_chunks), dtype=numpy.float32)
    
    # For each chunk attributed to the current CPU.
    for count, gidx in enumerate(loc_all_chunks):
        medians[:, count] = extract_median(chunk_size, gidx)
        if comm.rank == 0:
            pbar.update(count)
    median = numpy.mean(medians, axis=1)
    
    comm.Barrier()
    
    medians = comm.gather(median, root=0)
    
    if comm.rank == 0:
        median = weighted_mean(loc_nbs_chunks, medians)
    
    if comm.rank == 0:
        pbar.finish()
    
    # Broadcast medians to each CPU.
    median = comm.bcast(median, root=0)
    
    comm.Barrier()
    
    if comm.rank == 0:
        io.print_and_log(["Computing extracellular thresholds..."],
                         level='default', logger=params)
    
    if comm.rank == 0:
        pbar = get_progressbar(loc_nb_chunks)
    
    mads = numpy.zeros((N_elec, loc_nb_chunks), dtype=numpy.float32)
    
    # For each chunk attributed to the current CPU.
    for count, gidx in enumerate(loc_all_chunks):
        mads[:, count] = extract_median_absolute_deviation(chunk_size, gidx, median)
        if comm.rank == 0:
            pbar.update(count)
    mad = numpy.mean(mads, axis=1)
    
    comm.Barrier()
    
    mads = comm.gather(mad, root=0)
    
    if comm.rank == 0:
        mad = weighted_mean(loc_nbs_chunks, mads)
    
    if comm.rank == 0:
        pbar.finish()
    
    # Broadcast median absolute deviation to each CPU.
    mad = comm.bcast(mad, root=0)
    
    comm.Barrier()
    data_file.close()
    
    return median, mad



# def plot_extracted_extra_spikes(loc_all_chunks, data_len, mpi_input, data_dtype,
#                                 chunk_len, chunk_size, N_total, nodes,
#                                 extra_means, extra_stds, k, params, safety_space,
#                                 safety_time):
#     """Temporary function to see if the computed thresholds for a given dataset are valid"""
    
#     count = 0
#     gidx = loc_all_chunks[0]
    
#     loc_chunk, loc_shape = io.load_chunk(params, gidx, chunk_len, chunk_size, nodes=nodes)
    
#     sampling_rate = params.getint('data', 'sampling_rate')
#     dist_peaks     = params.getint('data', 'dist_peaks')
#     skip_artefact  = params.getboolean('data', 'skip_artefact')
#     template_shift = params.getint('data', 'template_shift')
#     alignment      = params.getboolean('detection', 'alignment')
#     nodes, _ = io.get_nodes_and_edges(params)
#     N_elec = nodes.size
#     do_temporal_whitening = params.getboolean('whitening', 'temporal')
#     do_spatial_whitening  = params.getboolean('whitening', 'spatial')
#     if do_spatial_whitening:
#         spatial_whitening  = io.load_data(params, 'spatial_whitening')
#     if do_temporal_whitening:
#         temporal_whitening = io.load_data(params, 'temporal_whitening')
#     loc_chunk = numpy.zeros(data_len, dtype=data_dtype)
#     mpi_input.Read_at(gidx * chunk_len, loc_chunk)
#     loc_shape = chunk_size
#     loc_chunk = loc_chunk.reshape(loc_shape, N_total)
#     # Consider only the valid channels.
#     loc_chunk = loc_chunk[:, nodes]
#     # extra_means_ = extra_means[nodes]
#     # extra_stds_ = extra_stds[nodes]
#     extra_means_ = extra_means
#     extra_stds_ = extra_stds
#     # Whiten signal.
#     if do_spatial_whitening:
#         loc_chunk = numpy.dot(loc_chunk, spatial_whitening)
#     if do_temporal_whitening:
#         loc_chunk = scipy.ndimage.filters.convolve1d(loc_chunk, temporal_whitening, axis=0, mode='constant')
#     # Preallocation for results.
#     peak_times = N_elec * [None]
#     peak_channels = N_elec * [None]
#     # For each electrode.
#     for e in xrange(0, N_elec):
#         # Extract the peaks of the current chunk.
#         threshold = k * extra_stds_[e]
#         peak_times[e] = algo.detect_peaks(loc_chunk[:, e], threshold, valley=True, mpd=dist_peaks)
#         peak_channels[e] = e * numpy.ones(peak_times[e].size, dtype='int')
#         if skip_artefact:
#             # Remove strong artifacts.
#             peak_values = loc_chunk[peak_times[e], e]
#             peak_indices = numpy.where(-10.0 * threshold <= peak_values)[0]
#             peak_times[e] = peak_times[e][peak_indices]
#     peak_times = numpy.concatenate(peak_times)
#     peak_channels = numpy.concatenate(peak_channels)
#     # Remove the useless borders.
#     if alignment:
#         loc_borders = (2 * template_shift, loc_shape - 2 * template_shift)
#     else:
#         loc_borders = (template_shift, loc_shape - template_shift)
#     peak_flags = (loc_borders[0] <= peak_times) & (peak_times < loc_borders[1])
#     peak_times = peak_times[peak_flags]
#     peak_channels = peak_channels[peak_flags]
#     # Filter unique peak times.
#     loc_peak_times = numpy.unique(peak_times)
#     n_times = len(loc_peak_times)
#     loc_peak_flags = numpy.zeros(n_times, dtype='bool')
#     loc_peak_elecs = numpy.zeros(n_times, dtype='int')
#     if 0 < len(loc_peak_times):
#         diff_times = loc_peak_times[-1] - loc_peak_times[0]
#         all_times = numpy.zeros((N_elec, diff_times + 1), dtype='bool')
#         min_times = numpy.maximum(loc_peak_times - loc_peak_times[0] - safety_time, 0)
#         max_times = numpy.minimum(loc_peak_times - loc_peak_times[0] + safety_time + 1, diff_times)
        
#         ##### TODO: remove temporary zone
#         numpy.random.seed(42)
#         ##### end temporary zone
        
#         argmax_peak = numpy.random.permutation(numpy.arange(n_times))
#         all_indices = loc_peak_times[argmax_peak]
#         # Select peaks with spatio-temporal masks.
#         for peak_index, peak_time in zip(argmax_peak, all_indices):
#             # Select electrode showing lowest amplitude.
#             elec = numpy.argmin(loc_chunk[peak_time, :])
#             _, neighs = get_neighbors(params, chan=elec)
#             if safety_space:
#                 mslice = all_times[neighs, min_times[peak_index]:max_times[peak_index]]
#             else:
#                 mslice = all_times[elec, min_times[peak_index]:max_times[peak_index]]
#             is_local_min = (elec in peak_channels[peak_times == peak_time])
#             if is_local_min and not mslice.any():
#                 loc_peak_flags[peak_index] = True
#                 loc_peak_elecs[peak_index] = elec
#                 if safety_space:
#                     all_times[neighs, min_times[peak_index]:max_times[peak_index]] = True
#                 else:
#                     all_times[elec, min_times[peak_index]:max_times[peak_index]] = True
#     loc_peak_times = numpy.compress(loc_peak_flags, loc_peak_times)
#     loc_peak_elecs = numpy.compress(loc_peak_flags, loc_peak_elecs)
    
#     time    = loc_peak_times
#     channel = loc_peak_elecs
    
#     pos = numpy.random.rand(time.size) - 0.5    

#     fig = plt.figure()
#     ax = fig.gca()
#     # For each first chunk plot one channel per figure.
#     for j in xrange(0, loc_chunk.shape[1]):
#         ax = fig.gca()
#         ax.plot(loc_chunk[:, j])
#         ax.plot([0, loc_chunk.shape[0] - 1], 2 * [extra_means_[j]], 'k--')
#         ax.plot([0, loc_chunk.shape[0] - 1], 2 * [extra_means_[j] + extra_stds_[j]], 'k--')
#         ax.plot([0, loc_chunk.shape[0] - 1], 2 * [extra_means_[j] - extra_stds_[j]], 'k--')
#         ax.plot([0, loc_chunk.shape[0] - 1], 2 * [extra_means_[j] + k * extra_stds_[j]], 'k--')
#         ax.plot([0, loc_chunk.shape[0] - 1], 2 * [extra_means_[j] - k * extra_stds_[j]], 'k--')
#         idx, = numpy.where(channel == j)
#         # y = + 250.0 * numpy.ones(idx.size) + 100.0 * pos[idx]
#         y = - 9.5 * numpy.ones(idx.size) + 1.0 * pos[idx]
#         ax.scatter(time[idx], y, c='r')
#         # new_idx, = numpy.where(new_channel == j)
#         # new_y = - 350.0 * numpy.ones(new_idx.size) + 100.0 * new_pos[new_idx]
#         # ax.scatter(new_time[new_idx], new_y, c='g')
#         ax.set_xlim(0, loc_chunk.shape[0] - 1)
#         # ax.set_ylim(- 400.0, 400.0)
#         ax.set_ylim(-10.0, 10.0)
#         plt.savefig("/tmp/check-{}-{}.png".format(j, comm.rank))
#         fig.clear()
    
#     return



def extract_extra_spikes_(params):
    """Detect spikes from the extracellular traces"""
    
    data_file = io.get_data_file(params)
    data_file.open()
    sampling_rate  = data_file.rate
    dist_peaks     = params.getint('data', 'dist_peaks')
    spike_thresh   = params.getfloat('detection', 'spike_thresh')
    template_shift = params.getint('data', 'template_shift')
    alignment      = params.getboolean('detection', 'alignment')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    safety_time = params.getfloat('clustering', 'safety_time')
    safety_space = params.getboolean('clustering', 'safety_space')
    chunk_size = params.getint('data', 'chunk_size')
    # chunk_size = params.getint('whitening', 'chunk_size')
    N_total        = data_file.N_tot
    file_out_suff  = params.get('data', 'file_out_suff')
    
    if do_spatial_whitening:
        spatial_whitening  = io.load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = io.load_data(params, 'temporal_whitening')
    
    #mpi_file = MPI.File()
    #mpi_input = mpi_file.Open(comm, data_filename, MPI.MODE_RDONLY)
    nb_chunks, last_chunk_len = data_file.analyze(chunk_size)
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
            beer_file.pop(extra_medians_key)
        beer_file.create_dataset(extra_medians_key, data=extra_medians)
        ## Save median absolute deviations.
        extra_mads_key = "extra_mads"
        if extra_mads_key in beer_file.keys():
            beer_file.pop(extra_mads_key)
        beer_file.create_dataset(extra_mads_key, data=extra_mads)
        beer_file.close()
    
    def extract_chunk_spikes(gidx, extra_thresh, valley=True):
        """Detect spikes from a chunk of the extracellular traces"""
        
        loc_chunk = data_file.get_data(gidx, chunk_size, nodes=nodes)
        loc_shape = len(loc_chunk)
        
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
        for e in xrange(N_elec):
            # Extract the peaks of the current chunk.
            threshold = extra_thresh * extra_mads[e]
            peak_times[e] = algo.detect_peaks(loc_chunk[:, e], threshold, valley=valley, mpd=dist_peaks)
            peak_channels[e] = e * numpy.ones(peak_times[e].size, dtype='int')
            
            peak_values = loc_chunk[peak_times[e], e]
            if valley:
                peak_indices = numpy.where(-10.0 * threshold <= peak_values)[0]
            else:
                peak_indices = numpy.where(peak_values <= +10.0 * threshold)[0]
            peak_times[e] = peak_times[e][peak_indices]
            peak_channels[e] = peak_channels[e][peak_indices]
        
        peak_times = numpy.concatenate(peak_times)
        peak_channels = numpy.concatenate(peak_channels)
        # Remove the useless borders.
        if alignment:
            loc_borders = (2 * template_shift, loc_shape - 2 * template_shift)
        else:
            loc_borders = (template_shift, loc_shape - template_shift)
        peak_flags = (loc_borders[0] <= peak_times) & (peak_times < loc_borders[1])
        peak_times = numpy.compress(peak_flags, peak_times)
        peak_channels = numpy.compress(peak_flags, peak_channels)
        # Filter unique peak times.
        loc_peak_times = numpy.unique(peak_times)
        ##### TODO: remove debug zone
        # if gidx < 1:
        #     numpy.save("tmp/loc_peak_times_{}_{}_.npy".format(gidx, int(extra_thresh)), loc_peak_times)
        ##### end debug zone
        n_times = len(loc_peak_times)
        loc_peak_flags = numpy.zeros(n_times, dtype='bool')
        loc_peak_elecs = numpy.zeros(n_times, dtype='int')
        loc_peak_values = numpy.zeros(n_times, dtype='float')
        if 0 < len(loc_peak_times):
            diff_times = loc_peak_times[-1] - loc_peak_times[0]
            all_times = numpy.zeros((N_elec, diff_times + 1), dtype='bool')
            min_times = numpy.maximum(loc_peak_times - loc_peak_times[0] - safety_time, 0)
            max_times = numpy.minimum(loc_peak_times - loc_peak_times[0] + safety_time + 1, diff_times)
            # Shuffle peaks.
            ##### TODO: clean temporary zone
            # argmax_peak = numpy.random.permutation(numpy.arange(n_times))
            if valley:
                for i, loc_peak_time in enumerate(loc_peak_times):
                    loc_peak_values[i] = numpy.amin(loc_chunk[loc_peak_time, :])
                argmax_peak = numpy.argsort(loc_peak_values)
            else:
                for i, loc_peak_time in enumerate(loc_peak_times):
                    loc_peak_values[i] = numpy.amax(loc_chunk[loc_peak_time, :])
                argmax_peak = numpy.argsort(loc_peak_values)
                argmes_peak = argmax_peak[::-1]
            ##### end temporary zone
            all_indices = loc_peak_times[argmax_peak]
            # Select peaks with spatio-temporal masks.
            for peak_index, peak_time in zip(argmax_peak, all_indices):
                # Select electrode showing lowest amplitude.
                if valley:
                    elec = numpy.argmin(loc_chunk[peak_time, :])
                else:
                    elec = numpy.argmax(loc_chunk[peak_time, :])
                _, neighs = get_neighbors(params, chan=elec)
                if safety_space:
                    mslice = all_times[neighs, min_times[peak_index]:max_times[peak_index]]
                else:
                    mslice = all_times[elec, min_times[peak_index]:max_times[peak_index]]
                is_local_min = (elec in peak_channels[peak_times == peak_time])
                if is_local_min and not mslice.any():
                    loc_peak_flags[peak_index] = True
                    loc_peak_elecs[peak_index] = elec
                    if valley:
                        loc_peak_values[peak_index] = - loc_chunk[peak_time, elec]
                    else:
                        loc_peak_values[peak_index] = loc_chunk[peak_time, elec]
                    if safety_space:
                        all_times[neighs, min_times[peak_index]:max_times[peak_index]] = True
                        # all_times[elec, min_times[peak_index]:max_times[peak_index]] = True
                    else:
                        all_times[elec, min_times[peak_index]:max_times[peak_index]] = True
        loc_peak_times = numpy.compress(loc_peak_flags, loc_peak_times)
        loc_peak_elecs = numpy.compress(loc_peak_flags, loc_peak_elecs)
        loc_peak_values = numpy.compress(loc_peak_flags, loc_peak_values)

        ##### TODO: remove debug zone
        # if gidx < 1:
        #     numpy.save("tmp/loc_peak_times_{}_{}.npy".format(gidx, int(extra_thresh)), loc_peak_times)
        #     numpy.save("tmp/loc_peak_elecs_{}_{}.npy".format(gidx, int(extra_thresh)), loc_peak_elecs)
        #     numpy.save("tmp/loc_peak_values_{}_{}.npy".format(gidx, int(extra_thresh)), loc_peak_values)
        #     numpy.save("tmp/loc_chunk_{}_{}.npy".format(gidx, int(extra_thresh)), loc_chunk)
        ##### end debug zone
        
        return loc_peak_times, loc_peak_elecs, loc_peak_values
    
    # Distribute chunks over CPUs.
    all_chunks = numpy.arange(nb_chunks)
    loc_all_chunks = all_chunks[comm.rank::comm.size]
    loc_nb_chunks = len(loc_all_chunks)
    
    if comm.rank == 0:
        io.print_and_log(["Collecting extracellular spikes..."], level='default', logger=params)
    
    if comm.rank == 0:
        pbar = get_progressbar(loc_nb_chunks)
    
    extra_valley = True
    
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
    values = len(loc_all_chunks) * [None]
    
    # For each chunk attributed to the current CPU.
    for (count, gidx) in enumerate(loc_all_chunks):
        time, channel, value = extract_chunk_spikes(gidx, spike_thresh, valley=extra_valley)
        times[count] = time + gidx * chunk_size
        channels[count] = channel
        values[count] = value
        if comm.rank == 0:
            pbar.update(count)
    
    # Concatenate times, channels and values.
    times = numpy.hstack(times)
    channels = numpy.hstack(channels)
    values = numpy.hstack(values)
    
    if comm.rank == 0:
        pbar.finish()
    
    comm.Barrier()
    
    # Gather times, channels and values.
    times = gather_array(times.astype(numpy.int64), comm, 0, dtype='int64')
    channels = gather_array(channels.astype(numpy.int64), comm, 0, dtype='int64')
    values = gather_array(values.astype(numpy.float64), comm, 0, dtype='float64')
    
    if comm.rank == 0:
        # Sort times, channels and values according to time.
        idx = numpy.argsort(times)
        times = times[idx]
        channels = channels[idx]
        values = values[idx]
    
    if comm.rank == 0:
        msg = [
            "Total number of extracellular spikes extracted: {}".format(channels.size),
        ] 
        msg2 = [
            "Number of extracellular spikes extracted on channel {}: {}".format(i, channels[channels == i].size) for i in numpy.unique(channels)
        ]
        io.print_and_log(msg, level='info', logger=params)
        io.print_and_log(msg2, level='debug', logger=params)
    
    
    if comm.rank == 0:
        path = "{}.beer.hdf5".format(file_out_suff)
        beer_file = h5py.File(path, 'a', libver='latest')
        group_name = "extra_spiketimes"
        if group_name in beer_file.keys():
            beer_file.pop(group_name)
        beer_file.create_group(group_name)
        for i in numpy.arange(0, N_elec):
            mask = (channels == i)
            triggers = times[mask]
            beer_file.create_dataset("{}/elec_{}".format(group_name, i), data=triggers)
        group_name = "extra_spike_values"
        if group_name in beer_file.keys():
            beer_file.pop(group_name)
        beer_file.create_group(group_name)
        for i in numpy.arange(0, N_elec):
            mask = (channels == i)
            data = values[mask]
            beer_file.create_dataset("{}/elec_{}".format(group_name, i), data=data)
        beer_file.close()
    
    comm.Barrier()
    
    return
    
    

def extract_extra_spikes(filename, params):
    
    do_extra = True
    try:
        data = io.load_data(params, 'extra-triggers')
        do_extra = False
    except Exception as e:
        do_extra = True

    if not do_extra:
        if comm.rank == 0:
            msg = [
                "Spike detection for extracellular traces has already been done"
            ]
            io.print_and_log(msg, 'info', params)
    else:
        extract_extra_spikes_(params)

    return



# Juxtacellular ################################################################

def highpass(data, BUTTER_ORDER=3, sampling_rate=10000, cut_off=500.0):
    Wn = (float(cut_off) / (float(sampling_rate) / 2.0), 0.95)
    b, a = signal.butter(BUTTER_ORDER, Wn, 'pass')
    return signal.filtfilt(b, a, data)



def extract_juxta_spikes_(params):
    '''Detect spikes from the extracellular traces'''
    
    file_out_suff = params.get('data', 'file_out_suff')
    sampling_rate = params.getint('data', 'sampling_rate')
    dist_peaks = params.getint('data', 'dist_peaks')
    template_shift = params.getint('data', 'template_shift')
    juxta_dtype = params.get('validating', 'juxta_dtype')
    juxta_thresh = params.getfloat('validating', 'juxta_thresh')
    juxta_valley = params.getboolean('validating', 'juxta_valley')
    
    juxta_filename = "{}.juxta.dat".format(file_out_suff)
    beer_path = "{}.beer.hdf5".format(file_out_suff)
    
    # Read juxtacellular trace.
    juxta_data = numpy.fromfile(juxta_filename, dtype=juxta_dtype)
    #juxta_data = juxta_data.astype(numpy.float32)
    # juxta_data = juxta_data - dtype_offset
    juxta_data = numpy.ascontiguousarray(juxta_data)
    
    # Filter juxtacellular trace.
    juxta_data  = highpass(juxta_data, sampling_rate=sampling_rate)
    juxta_data -= numpy.median(juxta_data)

    # Compute median and median absolute deviation.
    juxta_median = numpy.median(juxta_data)
    juxta_ad     = numpy.abs(juxta_data - juxta_median)
    juxta_mad   = numpy.median(juxta_ad, axis=0)
    
    # Save medians and median absolute deviations to BEER file.
    beer_file = h5py.File(beer_path, 'a', libver='latest')
    if "juxta_median" in beer_file.keys():
        beer_file.pop("juxta_median")
    beer_file.create_dataset("juxta_median", data=juxta_median)
    if "juxta_mad" in beer_file.keys():
        beer_file.pop("juxta_mad")
    beer_file.create_dataset("juxta_mad", data=juxta_mad)
    beer_file.close()

    if comm.rank == 0:
        io.print_and_log(["Extract juxtacellular spikes"], level='debug', logger=params)
    
    # Detect juxta spike times.
    threshold = juxta_thresh * juxta_mad
    juxta_spike_times = algo.detect_peaks(juxta_data, threshold, valley=juxta_valley, mpd=dist_peaks)

    # Remove juxta spike times in the borders.
    juxta_spike_times = juxta_spike_times[template_shift <= juxta_spike_times]
    juxta_spike_times = juxta_spike_times[juxta_spike_times < juxta_data.size - template_shift]
    
    # Save juxta spike times to BEER file.
    beer_file = h5py.File(beer_path, 'a', libver='latest')
    group_name = "juxta_spiketimes"
    if group_name in beer_file.keys():
        beer_file.pop(group_name)
    beer_file.create_group(group_name)
    key = "{}/elec_0".format(group_name)
    beer_file.create_dataset(key, data=juxta_spike_times)
    beer_file.close()
    
    # Find juxta spike values of juxta spike times.
    juxta_spike_values = numpy.zeros_like(juxta_spike_times, dtype='float')
    for i, t in enumerate(juxta_spike_times):
        if juxta_valley:
            juxta_spike_values[i] = - juxta_data[t]
        else:
            juxta_spike_values[i] = + juxta_data[t]
    
    # Save juxta spike values to BEER file.
    beer_file = h5py.File(beer_path, 'a', libver='latest')
    group_name = "juxta_spike_values"
    if group_name in beer_file.keys():
        beer_file.pop(group_name)
    beer_file.create_group(group_name)
    key = "{}/elec_0".format(group_name)
    beer_file.create_dataset(key, data=juxta_spike_values)
    beer_file.close()
    
    return



def extract_juxta_spikes(filename, params):
    do_juxta = True
    try:
        data = io.load_data(params, 'juxta-triggers')
        do_juxta = False
    except Exception:
        do_juxta = True

    if not do_juxta:
        if comm.rank == 0:
            msg = [
                "Spike detection for juxtacellular traces has already been done"
            ]
            io.print_and_log(msg, 'info', params)
    elif do_juxta:
        extract_juxta_spikes_(params)
    return



# Validating utils #############################################################

class Projection(object):
    
    # TODO: test class.
    
    def __init__(self, tol=0):
        self.tol = tol
        self.fitted = False
    
    ##### TODO: clean temporary zone
    def fit(self, X, y):
        USE_OLD_VERSION = False
        if USE_OLD_VERSION:
            if type(X) is list:
                X = numpy.vstack(tuple(X))
            if type(y) is list:
                y = numpy.vstack(tuple(y))
                y = y.ravel()
            self.lda = LDA(n_components=1, tol=self.tol)
            self.lda = self.lda.fit(X, y)
            self.v1 = self.lda.scalings_[:, 0]
            self.v1 = self.v1 / numpy.linalg.norm(self.v1)
            self.mean = self.lda.xbar_
            self.v2 = numpy.ones(self.v1.size)
            self.v2 = self.v2 - numpy.dot(self.v1, self.v2) * self.v1 / numpy.linalg.norm(self.v1)
            self.v2 = self.v2 / numpy.linalg.norm(self.v2)
            self.fitted = True
            return self
        else:
            if type(X) is list:
                X = numpy.vstack(tuple(X))
            if type(y) is list:
                y = numpy.vstack(tuple(y))
                y = y.ravel()
            self.mean = numpy.mean(X, axis=0)
            uys = numpy.unique(y)
            uys = numpy.sort(uys)
            umus = uys.size * [None]
            for i, uy in enumerate(uys):
                uX = X[y == uy, :] - self.mean
                uy = y[y == uy]
                umus[i] = numpy.mean(uX, axis=0)
            uMu = numpy.stack(umus)
            if uMu.shape[0] == 2:
                self.v1 = uMu[1, :] - uMu[0, :]
                self.v1 = self.v1 / numpy.linalg.norm(self.v1)
                self.v2 = numpy.random.rand(self.v1.size)
                self.v2 = self.v2 / numpy.linalg.norm(self.v2)
                self.v2 = self.v2 - numpy.dot(self.v1, self.v2) * self.v1
                self.v2 = self.v2 / numpy.linalg.norm(self.v2)
                self.fitted = True
            else:
                raise NotImplementedError
            # TODO: complete.
            return self
    ##### end temporary zone
    
    def transform(self, X):
        if not self.fitted:
            raise Exception("Must be fitted first")
        if type(X) is list:
            X = numpy.vstack(tuple(X))
        x1 = numpy.dot(X - self.mean, self.v1).reshape(-1, 1)
        x2 = numpy.dot(X - self.mean, self.v2).reshape(-1, 1)
        x = numpy.hstack((x1, x2))
        return x
    
    def get_vectors(self):
        return self.v1, self.v2
    
    def get_mean(self):
        return self.mean

def accuracy_score(y_true, y_pred, class_weights=None):
    """Accuracy classification score."""
    mask = (y_true == y_pred)
    if class_weights is None:
        m = y_true[mask].size
        n = y_true.size
        score = float(m) / float(n)
    else:
        m1 = numpy.count_nonzero(y_true[mask])
        m0 = y_true[mask].size - m1
        n1 = numpy.count_nonzero(y_true)
        n0 = y_true.size - n1
        score = (class_weights[0] * float(m0) + class_weights[1] * float(m1)) \
                / (class_weights[0] * float(n0) + class_weights[1] * float(n1))
    return score

# Useful function to convert an ellispoid in standard form to an ellispoid
# in general form.
def ellipsoid_standard_to_general(t, s, O, verbose=False, logger=None):
    # Translation from standard matrix to general matrix.
    d = numpy.divide(1.0, numpy.power(s, 2.0))
    D = numpy.diag(d)
    A = O * D * O.T
    ##### TODO: remove test zone
    w, v = numpy.linalg.eigh(A)
    if verbose:
        msg = [
            # "# det(A)",
            # "%s" %(numpy.linalg.det(A),),
            "# Eigenvalues",
            "%s" %(w,),
        ]
        io.print_and_log(msg, level='default', logger=logger)
    ##### end test zone
    b = - 2.0 * numpy.dot(t, A)
    c = numpy.dot(t, numpy.dot(A, t)) - 1
    # Translation from general matrix to coefficients.
    N = t.size
    coefs = numpy.zeros(1 + N + (N + 1) * N / 2)
    coefs[0] = c
    for i in xrange(0, N):
        coefs[1 + i] = b[i]
    k = 0
    for i in xrange(0, N):
        coefs[1 + N + k] = A[i, i]
        k = k + 1
        for j in xrange(i + 1, N):
            # TODO: remove test zone
            # coefs[1 + N + k] = A[i, j]
            # coefs[1 + N + k] = A[j, i]
            coefs[1 + N + k] = A[i, j] + A[j, i]
            # end test zone
            k = k + 1
    return coefs

# Useful function to convert an ellispoid in general form to an ellispoid in
# standard form.
def ellipsoid_general_to_standard(coefs, verbose=False, logger=None):
    """
    Convert an ellipsoid in general form:
        a_{0}
        + a_{1} x1 + ... + a_{m} xm
        + a_{1, 1} * x1 * x1 + ... + a_{1, m} * x1 * xm
        + ...
        + a_{m, m} xm * xm
        = 0
    To standard form (TODO: check validity):
        (x1 - x1') * phi1(t_{1, 2}, ..., t_{m-1, m})
        + ...
        + (xm - xm') * phim(t_{1, 2}, ..., t_{m-1, m})
    The ellipse has center [x1', ..., xm']^T, semi-axes b1, ... and bm, and
    the angle to the semi-major axis is t.
    """
    # Convert to float.
    coefs = coefs.astype('float')
    K = coefs.size
    # Retrieve the number of dimension (i.e. N).
    # (i.e. solve: 1 + N + (N + 1) * N / 2 = K)
    N = int(- 1.5 + numpy.sqrt(1.5 ** 2.0 - 4.0 * 0.5 * (1.0 - float(K))))
    if verbose:
        msg = [
            "# K",
            "%s" %(K,),
            "# N",
            "%s" %(N,),
        ]
        io.print_and_log(msg, level='default', logger=logger)
    # Retrieve the matrix representation.
    A = numpy.zeros((N, N))
    k = 0
    for i in xrange(0, N):
        A[i, i] = coefs[1 + N + k]
        k = k + 1
        for j in xrange(i + 1, N):
            A[i, j] = coefs[1 + N + k] / 2.0
            A[j, i] = coefs[1 + N + k] / 2.0
            k = k + 1
    b = coefs[1:1+N]
    c = coefs[0]
    # Compute the center of the ellipsoid.
    center = - 0.5 * numpy.dot(numpy.linalg.inv(A), b)
    
    ##### TODO: remove test zone
    if verbose:
        msg = [
            "# Test of symmetry",
            "%s" %(numpy.all(A == A.T),),
        ]
        io.print_and_log(msg, level='default', logger=logger)
    ##### end test zone
    
    # Each eigenvector of A lies along one of the axes.
    evals, evecs = numpy.linalg.eigh(A)
    
    ##### TODO: remove print zone.
    if verbose:
        msg = [
            "# Semi-axes computation",
            "## det(A)",
            "%s" %(numpy.linalg.det(A),),
            "## evals",
            "%s" %(evals,),
        ]
        io.print_and_log(msg, level='default', logger=logger)
    ##### end print zone.
    
    # Semi-axes from reduced canonical equation.
    ##### TODO: remove test zone.
    # eaxis = numpy.sqrt(- c / evals)
    eaxis = numpy.sqrt(numpy.abs(-c / evals))
    ##### end test zone
    return center, eaxis, evecs

def ellipsoid_matrix_to_coefs(A, b, c):
    N = b.size
    K = 1 + N + (N + 1) * N / 2
    coefs = numpy.zeros(K)
    coefs[0] = c
    coefs[1:1+N] = b
    k = 0
    for i in xrange(0, N):
        coefs[1 + N + k] = A[i, i]
        k = k + 1
        for j in xrange(i + 1, N):
            coefs[1 + N + k] = A[i, j] + A[j, i]
            k = k + 1
    coefs = coefs.reshape(-1, 1)
    return coefs

def ellipsoid_coefs_to_matrix(coefs):
    K = coefs.size
    # Retrieve the number of dimension (i.e. N).
    # (i.e. solve: 1 + N + (N + 1) * N / 2 = K)
    N = int(- 1.5 + numpy.sqrt(1.5 ** 2.0 - 4.0 * 0.5 * (1.0 - float(K))))
    # Retrieve A.
    A = numpy.zeros((N, N))
    k = 0
    for i in xrange(0, N):
        A[i, i] = coefs[1 + N + k, 0]
        k = k + 1
        for j in xrange(i + 1, N):
            A[i, j] = coefs[1 + N + k, 0] / 2.0
            A[j, i] = coefs[1 + N + k, 0] / 2.0
            k = k + 1
    # Retrieve b.
    b = coefs[1:1+N, 0]
    # Retrieve c.
    c = coefs[0, 0]
    return A, b, c

def find_rotation(v1, v2, verbose=False, logger=None):
    '''Find a rotation which maps these two vectors of the two first vectors of
    the canonical basis.'''
    N = v1.size
    x = numpy.copy(v1)
    R = numpy.eye(N)
    for i in xrange(1, N):
        x1 = x[0]
        x2 = x[i]
        n = numpy.sqrt(x1 * x1 + x2 * x2)
        if n == 0.0:
            cos = 1.0
            sin = 0.0
        else:
            cos = x1 / n
            sin = x2 / n
        R_ = numpy.eye(N)
        R_[0, 0] = cos
        R_[0, i] = sin
        R_[i, 0] = - sin
        R_[i, i] = cos
        x = numpy.dot(R_, x)
        R = numpy.dot(R_, R)
    x = numpy.dot(R, v2)
    for i in xrange(2, N):
        x1 = x[1]
        x2 = x[i]
        n = numpy.sqrt(x1 * x1 + x2 * x2)
        if n == 0.0:
            cos = 1.0
            sin = 0.0
        else:
            cos = x1 / n
            sin = x2 / n
        R_ = numpy.eye(N)
        R_[1, 1] = cos
        R_[1, i] = sin
        R_[i, 1] = - sin
        R_[i, i] = cos
        x = numpy.dot(R_, x)
        R = numpy.dot(R_, R)
    if verbose:
        # u1 = numpy.dot(R, v1)
        # u1[numpy.abs(u1) < 1.0e-10] = 0.0
        # u2 = numpy.dot(R, v2)
        # u2[numpy.abs(u2) < 1.0e-10] = 0.0
        # msg = [
        #     "# R * v1",
        #     "%s" %(u1,),
        #     "# R * v2",
        #     "%s" %(u2,),
        # ]
        # io.print_and_log(msg, level='default', logger=logger)
        pass
    return R

def find_apparent_contour(A, b, c):
    '''Find the apparent contour of a classifier'''
    xs = [numpy.array([0.0, 0.0]),
          numpy.array([1.0, 0.0]),
          numpy.array([0.0, 1.0])]
    # Solve the linear system 2 * A.T * y + b = 0 for fixed couples (y_1, y_2).
    ys = []
    for x in xs:
        c1 = 2.0 * A[2:, 2:].T
        c2 = - (numpy.dot(2.0 * A[:2, 2:].T, x) + b[2:])
        yx = numpy.linalg.solve(c1, c2)
        ys.append(yx)
    # Solve the linear system to express (y_3, ..., y_m) with (y_1, y_2).
    k = ys[0].size
    c1 = numpy.eye(k)
    c1 = numpy.tile(c1, (3, 3))
    for (i, x) in enumerate(xs):
        for (j, v) in enumerate(x):
            c1[i*k:(i+1)*k, j*k:(j+1)*k] = v * c1[i*k:(i+1)*k, j*k:(j+1)*k]
    c2 = numpy.concatenate(tuple(ys))
    m = numpy.linalg.solve(c1, c2)
    # Reconstruct alpha.
    alpha_1 = numpy.eye(2)
    alpha_2 = numpy.hstack((m[0:k].reshape(-1, 1), m[k:2*k].reshape(-1, 1)))
    alpha = numpy.vstack((alpha_1, alpha_2))
    # Reconstruct beta.
    beta_1 = numpy.zeros(2)
    beta_2 = m[2*k:3*k]
    beta = numpy.concatenate((beta_1, beta_2))
    # Reconstruct the apparent contour.
    A_ = numpy.dot(alpha.T, numpy.dot(A, alpha))
    b_ = numpy.dot(alpha.T, 2.0 * numpy.dot(A, beta) + b)
    c_ = numpy.dot(numpy.dot(A, beta) + b, beta) + c
    return(A_, b_, c_)

def evaluate_ellipse(A, b, c, X):
    '''Compute ellipse values for various points'''
    x2 = numpy.sum(numpy.multiply(X.T, numpy.dot(A, X.T)), axis=0)
    x1 = numpy.dot(b, X.T)
    x0 = c
    d2 = x2 + x1 + x0
    return d2

def squared_Mahalanobis_distance(A, mu, X):
    '''Compute squared Mahalanobis distance for various points'''
    N = X.shape[0]
    d2 = numpy.zeros(N)
    for i in xrange(0, N):
        d2[i] = numpy.dot(X[i, :] - mu, numpy.dot(A, X[i, :] - mu))
    return d2

def get_class_weights(y_gt, y_ngt, y_noi=None, n=7):
    '''Compute different class weights for the stochastic gradient descent'''
    n_class_0 = float(y_gt.size)
    if y_noi is None:
        n_class_1 = float(y_ngt.size)
    else:
        n_class_1 = float(y_ngt.size + y_noi.size)
    n_samples = n_class_0 + n_class_1
    n_classes = 2.0
    alphas = numpy.linspace(2.0, 0.0, n + 2)[1:-1]
    betas = numpy.linspace(0.0, 2.0, n + 2)[1:-1]
    class_weights = []
    for i in xrange(0, n):
        alpha = alphas[i]
        beta = betas[i]
        weight_0 = alpha * n_samples / (n_classes * n_class_0)
        weight_1 = beta * n_samples / (n_classes * n_class_1)
        class_weight = {
            0: n_classes * weight_0 / (weight_0 + weight_1),
            1: n_classes * weight_1 / (weight_0 + weight_1),
        }
        class_weights.append(class_weight)
    return alphas, betas, class_weights

##### TODO: clean temporary zone
def get_class_weights_bis(n_class_0, n_class_1, n=7):
    '''Compute different class weights for the stochastic gradient descent'''
    n_class_0 = float(n_class_0)
    n_class_1 = float(n_class_1)
    n_samples = n_class_0 + n_class_1
    n_classes = 2.0
    alphas = numpy.linspace(2.0, 0.0, n + 2)[1:-1]
    betas = numpy.linspace(0.0, 2.0, n + 2)[1:-1]
    class_weights = []
    for i in xrange(0, n):
        alpha = alphas[i]
        beta = betas[i]
        weight_0 = alpha * n_samples / (n_classes * n_class_0)
        weight_1 = beta * n_samples / (n_classes * n_class_1)
        class_weight = {
            0: n_classes * weight_0 / (weight_0 + weight_1),
            1: n_classes * weight_1 / (weight_0 + weight_1),
        }
        class_weights.append(class_weight)
    return alphas, betas, class_weights
##### end temporary zone
