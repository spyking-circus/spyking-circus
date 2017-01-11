import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, h5py

from ..shared.utils import *
from ..shared.files import get_stas, get_stas_memshared
from ..shared import plot
from circus.shared.parser import CircusParser
from circus.shared.messages import print_and_log, init_logging
from circus.shared.mpi import SHARED_MEMORY, comm

try:
    import sklearn
except Exception:
    if comm.rank == 0:
        print "Sklearn is not installed! Install spyking-circus with the beer extension (see documentation)"
    sys.exit(1)

from sklearn.decomposition import PCA
# TODO: remove following line (i.e. remove warning).
# from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier


from matplotlib.cm import inferno
from matplotlib.patches import Rectangle

from .utils import *



def main(params, nb_cpu, nb_gpu, us_gpu):    
    
    # RETRIEVE PARAMETERS FOR VALIDATING #######################################
    
    logger         = init_logging(params.logfile)
    logger         = logging.getLogger('circus.validating')
    data_file      = params.data_file
    N_e            = params.getint('data', 'N_e')
    N_t            = params.getint('detection', 'N_t')
    N_total        = params.nb_channels
    sampling_rate  = params.rate

    template_shift = params.getint('detection', 'template_shift')
    file_out_suff = params.get('data', 'file_out_suff')
    nb_repeats = params.getint('clustering', 'nb_repeats')
    max_iter = params.getint('validating', 'max_iter')
    learning_rate_init = params.getfloat('validating', 'learning_rate')
    make_plots = params.get('validating', 'make_plots')
    roc_sampling = params.getint('validating', 'roc_sampling')
    plot_path = os.path.join(params.get('data', 'data_file_noext'), 'plots')
    test_size = params.getfloat('validating', 'test_size')
    matching_jitter = params.getfloat('validating', 'matching_jitter')
    
    verbose   = False
    skip_demo = False
    make_plots_snippets = False
    # test_method = 'full' # full test set
    test_method = 'downsampled' # downsampled test set
    
    # N_max = 1000000
    N_max = 12000
    # N_max = 6000
    # N_max = 3000
    # N_max = 1500
    
    # Compute 'time_min' and 'time_max'.

    time_min = template_shift
    time_max = (data_file.duration - 1) - template_shift
    
    # Initialize the random seed.
    _ = numpy.random.seed(0)
    
    
    
    ###### JUXTACELLULAR SPIKE DETECTION #######################################
    
    # Detect the spikes times of the juxtacellular trace.
    if comm.rank == 0:
        extract_juxta_spikes(params)
    comm.Barrier()

    beer_path  = "{}.beer.hdf5".format(file_out_suff)
    beer_file  = h5py.File(beer_path, 'a', libver='latest')
    group_name = "juxta_spike_values"
    key = "{}/elec_0".format(group_name)

    if len(beer_file.get(key)) == 0:
        if comm.rank == 0:
            print_and_log(['No juxta-cellular spikes have been found!'], 'error', logger)
        sys.exit(1)
    beer_file.close()
    
    # Retrieve the spike times of the juxtacellular trace.
    spike_times_juxta = io.load_data(params, 'juxta-triggers')
    
    
    
    ##### PLOT INFLUENCE OF JUXTACELLULAR THRESHOLD ############################
    
    # Compute the cumulative distribution of juxta spike times according to the threshold value.
    spike_values_juxta = io.load_data(params, 'juxta-values')
    juxta_thresh = params.getfloat('validating', 'juxta_thresh')
    juxta_mad = io.load_data(params, 'juxta-mad')
    
    spike_values_juxta = numpy.sort(spike_values_juxta) / juxta_mad
    threshs = numpy.concatenate((numpy.array([juxta_thresh]), spike_values_juxta))
    counts = numpy.arange(spike_values_juxta.size, -1, -1)
    unknown_zone = Rectangle((0.0, 0), juxta_thresh, spike_values_juxta.size,
                             hatch='/', facecolor='white', zorder=3)
    
    if comm.rank == 0:

        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        
        if make_plots not in ['None', '']:
            plot_filename = "beer-juxta-distribution.{}".format(make_plots)
            path = os.path.join(plot_path, plot_filename)
            import pylab
            fig = pylab.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_position((0.1, 0.15, 0.8, 0.75))
            ax.step(threshs, counts, 'k', where='post')
            ax.add_patch(unknown_zone)
            ax.grid(True)
            if len(spike_values_juxta) > 0:
                ax.set_xlim(0.0, numpy.amax(spike_values_juxta))
            ax.set_ylim(0, spike_values_juxta.size)
            ax.set_title("Juxtacellular threshold detection")
            ax.set_xlabel("threshold")
            ax.set_ylabel("number of spikes")
            fig.text(0.02, 0.02, "median absolute deviation: {:.2f}".format(juxta_mad))
            pylab.savefig(path)
            pylab.close()
    
    
    
    ############################################################################
    
    # Retrieve PCA basis.
    basis_proj, basis_rec = io.load_data(params, 'basis')
    N_p = basis_proj.shape[1]
    
    # Select only the neighboring channels of the best channel.
    chan = params.get('validating', 'nearest_elec')
    if chan == 'auto':
        ###### TODO: clean temporary zone
        # Set best channel as the channel with the highest change in amplitude.
        nodes, chans = get_neighbors(params, chan=None)
        spike_labels_juxta = numpy.zeros(len(spike_times_juxta))
        #juxta_spikes = load_chunk(params, spike_times_juxta, chans=None)
        juxta_spikes = get_stas(params, spike_times_juxta, spike_labels_juxta, 0, chans, nodes=nodes, auto_align=False).T
        spike_labels_juxta_ = numpy.zeros(len(spike_times_juxta))
        juxta_spikes_ = get_juxta_stas(params, spike_times_juxta, spike_labels_juxta).T
        
        USE_OLD_VERSION = False
        if USE_OLD_VERSION:
            tmp_juxta_spikes = juxta_spikes
            tmp_juxta_spikes_ = juxta_spikes_
        else:
            # Remove juxta spikes times for which we see some artifacts in the corresponding extra snipets.
            juxta_spike_times_selection = numpy.ones(juxta_spikes.shape[2], dtype=numpy.bool)
            for elec in xrange(0, juxta_spikes.shape[1]):
                median = numpy.median(juxta_spikes[:, elec, :])
                tmp_juxta_spikes = numpy.abs(juxta_spikes - median)
                mad_juxta_spikes = numpy.median(tmp_juxta_spikes)
                for spike_time_index in xrange(0, juxta_spikes.shape[2]):
                    # Since extra_valley is always true.
                    min_juxta_spikes = numpy.amin(juxta_spikes[:, elec, spike_time_index])
                    if min_juxta_spikes <= - 20.0 * juxta_thresh * mad_juxta_spikes:
                        # There is an artifact.
                        juxta_spike_times_selection[spike_time_index] = False
                        ##### TODO: remove debug zone
                        # print("##### Remove artifact (spike time index: {})".format(spike_time_index))
                        ##### end debug zone
            tmp_juxta_spikes = juxta_spikes[:, :, juxta_spike_times_selection]
            tmp_juxta_spikes_ = juxta_spikes_[:, juxta_spike_times_selection]
        mean_juxta_spikes = numpy.mean(tmp_juxta_spikes, axis=2) # average over spike times
        max_juxta_spikes = numpy.amax(mean_juxta_spikes, axis=0) # argmax over timestamps
        min_juxta_spikes = numpy.amin(mean_juxta_spikes, axis=0) # argmin over timestamps
        dif_juxta_spikes = max_juxta_spikes - min_juxta_spikes
        chan = numpy.argmax(dif_juxta_spikes)
        
        nodes, chans = get_neighbors(params, chan=chan)
        if comm.rank == 0:
            msg = ["Ground truth neuron is close to channel {} (set automatically)".format(chan)]
            print_and_log(msg, level='default', logger=logger)
        ##### TODO: clean temporary zone
    else:
        chanl = int(chan)
        nodes, chans = get_neighbors(params, chan=chan)
        ##### TODO: clean temporary zone
        elec = numpy.where(chans == chan)[0][0]
        chan = elec
        ##### end temporary zone
        spike_labels_juxta = numpy.zeros(len(spike_times_juxta))
        juxta_spikes = get_stas(params, spike_times_juxta, spike_labels_juxta, 0, chans, nodes=nodes, auto_align=False).T
        spike_labels_juxta_ = numpy.zeros(len(spike_times_juxta))
        juxta_spikes_ = get_juxta_stas(params, spike_times_juxta, spike_labels_juxta).T
        tmp_juxta_spikes = juxta_spikes
        tmp_juxta_spikes_ = juxta_spikes_
        if comm.rank == 0:
            msg = ["Ground truth neuron is close to channel {} (set manually)".format(chan)]
            print_and_log(msg, level='default', logger=logger)

    if comm.rank == 0:
        
        if make_plots not in ['None', '']:
            plot_filename = "beer-trigger-times.{}".format(make_plots)
            path = os.path.join(plot_path, plot_filename)
            plot.view_trigger_times(params, spike_times_juxta, tmp_juxta_spikes[:, chan, :], tmp_juxta_spikes_, save=path)
    
    
    
    ###### EXTRACELLULAR SPIKE DETECTION #######################################
    
    # Detect the spikes times of the "non ground truth cell".
    extract_extra_spikes(params)
    comm.Barrier()
    # Retrieve the spike times of the "non ground truth cell".
    spike_times_ngt_tmp = io.load_data(params, 'extra-triggers')
    
    
    
    ##### PLOT INFLUENCE OF EXTRACELLULAR THRESHOLD ############################
    
    # Compute the cumulative distribution of extra spike times according to the threshold values.
    spike_times_extra = spike_times_ngt_tmp
    spike_values_extra = io.load_data(params, 'extra-values')
    extra_thresh = params.getfloat('detection', 'spike_thresh')
    extra_mads = io.load_data(params, 'extra-mads')
    
    N_e = params.getint('data', 'N_e')
    threshs = N_e * [None]
    counts = N_e * [None]
    for e in xrange(0, N_e):
        spike_values_extra[e] = numpy.sort(spike_values_extra[e]) / extra_mads[e]
    xmax = max([0.0] + [numpy.amax(s) for s in spike_values_extra if 0 < s.size])
    ymax = max([s.size + 1 for s in spike_values_extra])
    for e in xrange(0, N_e):
        threshs[e] = numpy.concatenate((numpy.array([extra_thresh]), spike_values_extra[e], numpy.array([xmax])))
        counts[e] = numpy.concatenate((numpy.arange(spike_values_extra[e].size, -1, -1), numpy.array([0])))
    
    unknown_zone = Rectangle((0.0, 0), extra_thresh, ymax,
                             hatch='/', facecolor='white', zorder=3)

    if comm.rank == 0:
        
        if make_plots not in ['None', '']:
            plot_filename = "beer-extra-distributions.{}".format(make_plots)
            path = os.path.join(plot_path, plot_filename)
            import pylab
            fig = pylab.figure()
            ax = fig.add_subplot(1, 1, 1)
            for e in xrange(0, N_e):
                color = inferno(float(e) / float(N_e))
                ax.step(threshs[e], counts[e], color=color, where='post')
            ax.add_patch(unknown_zone)
            ax.grid(True)
            ax.set_xlim(0.0, xmax)
            ax.set_ylim(0, ymax)
            ax.set_title("Extracellular threshold detection")
            ax.set_xlabel("threshold")
            ax.set_ylabel("number of spikes")
            pylab.savefig(path)
            pylab.close()
    
    # Compute the cumulative distribution of extra spike times according to the threshold values.
    spike_times_extra = spike_times_ngt_tmp
    spike_values_extra = io.load_data(params, 'extra-values')
    extra_thresh = params.getfloat('detection', 'spike_thresh')
    extra_mads = io.load_data(params, 'extra-mads')
    
    N_e = params.getint('data', 'N_e')
    for e in xrange(0, N_e):
        spike_values_extra[e] = spike_values_extra[e] / extra_mads[e]
    spike_values_extra = numpy.concatenate(spike_values_extra)
    spike_values_extra = numpy.sort(spike_values_extra)
    counts = numpy.arange(spike_values_extra.size - 1, -1, -1)
    spike_values_extra = numpy.concatenate((numpy.array([extra_thresh]), spike_values_extra))
    counts = numpy.concatenate((numpy.array([counts[0]]), counts))
    xmax = numpy.amax(spike_values_extra)
    ymax = numpy.amax(counts)
    
    unknown_zone = Rectangle((0.0, 0), extra_thresh, ymax,
                             hatch='/', facecolor='white', zorder=3)

    if comm.rank == 0:
        
        if make_plots not in ['None', '']:
            plot_filename = "beer-extra-distributions-bis.{}".format(make_plots)
            path = os.path.join(plot_path, plot_filename)
            import pylab
            fig = pylab.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.step(spike_values_extra, counts, color='black', where='post')
            ax.add_patch(unknown_zone)
            ax.grid(True)
            ax.set_xlim(0.0, xmax)
            ax.set_ylim(0, ymax)
            ax.set_title("Extracellular threshold detection")
            ax.set_xlabel("threshold")
            ax.set_ylabel("number of spikes")
            pylab.savefig(path)
            pylab.close()
    
    
    
    ##### PLOT PROPORTION OF JUXTA SPIKE TIMES NEAR EXTRA SPIKE TIMES ##########
    
    # Compute the proportion of juxtacellular spikes present in the extracelllar
    # spikes according to the threshold value.
    spike_values_juxta = io.load_data(params, 'juxta-values')
    juxta_thresh = params.getfloat('validating', 'juxta_thresh')
    juxta_mad = io.load_data(params, 'juxta-mad')
    juxta_thresh = max(5.0, juxta_thresh)
    mask = juxta_thresh * juxta_mad <= spike_values_juxta
    spike_times_juxta = spike_times_juxta[mask]
    spike_values_juxta = spike_values_juxta[mask]
    
    spike_times_extra = spike_times_ngt_tmp
    spike_values_extra = io.load_data(params, 'extra-values')
    extra_thresh = params.getfloat('detection', 'spike_thresh')
    extra_mads = io.load_data(params, 'extra-mads')
    
    thresh = int(float(params.rate) * matching_jitter * 1.0e-3) # "matching threshold"
    
    for e in xrange(0, N_e):
        spike_values_extra[e] = spike_values_extra[e] / extra_mads[e]
    spike_times_extra = numpy.concatenate(spike_times_extra)
    spike_values_extra = numpy.concatenate(spike_values_extra)
    
    matches = []
    for spike_time_juxta in spike_times_juxta:
        idx = numpy.where(abs(spike_times_extra - spike_time_juxta) <= thresh)[0]
        if 0 < len(idx):
            matches.append(numpy.amax(spike_values_extra[idx]))
        else:
            pass
    matches = sorted(matches)
    counts = numpy.arange(len(matches) - 1, -1, -1)
    matches = numpy.concatenate((numpy.array([extra_thresh]), matches))
    if len(counts) > 0:
        counts = numpy.concatenate((numpy.array([counts[0]]), counts))
    counts = 100.0 * counts.astype('float') / float(spike_times_juxta.size)
    
    unknown_zone = Rectangle((0.0, 0), extra_thresh, 100.0,
                             hatch='/', facecolor='white', zorder=3, fill=False)

    if comm.rank == 0:
        
        # Save proportion in BEER file.
        if len(counts) > 0:
            proportion = counts[0]
        else:
            proportion = 0
        beer_path = "{}.beer.hdf5".format(file_out_suff)
        beer_file = h5py.File(beer_path, 'a', libver='latest')
        beer_key = 'proportion'
        if beer_key in beer_file.keys():
            beer_file.pop(beer_key)
        beer_file.create_dataset(beer_key, data=proportion)
        beer_file.close()
        
        if make_plots not in ['None', ''] and len(counts) > 0:
            plot_filename = "beer-proportion.{}".format(make_plots)
            path = os.path.join(plot_path, plot_filename)
            import pylab
            fig = pylab.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_position((0.1, 0.15, 0.8, 0.75))
            ax.step(matches, counts, color='black', where='post')
            ax.add_patch(unknown_zone)
            ax.grid(True)
            ax.set_xlim(0.0, numpy.amax(matches))
            ax.set_ylim(0.0, 100.0)
            ax.set_title("Proportion of juxta spike times near extra spike times")
            ax.set_xlabel("extra threshold")
            ax.set_ylabel("proportion (%)")
            fig.text(0.02, 0.02, "matching jitter: {} ms".format(matching_jitter))
            fig.text(0.42, 0.02, "juxta threshold: {}".format(juxta_thresh))
            tmp_indices = numpy.where(matches <= extra_thresh)[0]
            if 0 < len(tmp_indices):
                tmp_index = tmp_indices[-1]
            else:
                tmp_index = 0
            fig.text(0.72, 0.02, "[{} -> {:.2f}%]".format(extra_thresh, counts[tmp_index]))
            pylab.savefig(path)
            pylab.close()
    
    
    
    ##### GROUND TRUTH CELL'S SAMPLES ##########################################
    
    if comm.rank == 0:
        print_and_log(["Collecting ground truth cell's samples..."], level='debug', logger=logger)
    
    
    # Retrieve the spike times of the "ground truth cell".
    tresh = int(float(params.rate) * matching_jitter * 1.0e-3) # "matching threshold"
    matched_spike_times_juxta = numpy.zeros_like(spike_times_juxta, dtype='bool')
    matched_spike_times_extra = numpy.zeros_like(spike_times_extra, dtype='bool')
    mismatched_spike_times_extra = numpy.zeros_like(spike_times_extra, dtype='bool')
    for i, spike_time_juxta in enumerate(spike_times_juxta):
        diff = abs(spike_times_extra - spike_time_juxta)
        idx = numpy.where(diff <= thresh)[0]
        if 0 < len(idx):
            idx_ = numpy.argmin(diff[idx])
            matched_spike_times_juxta[i] = True
            matched_spike_times_extra[idx[idx_]] = True
            mismatched_spike_times_extra[idx] = True
            mismatched_spike_times_extra[idx[idx_]] = False
        else:
            pass
    
    ##### TODO: clean working zone
    
    threshold_false_negatives = matched_spike_times_juxta.size - numpy.count_nonzero(matched_spike_times_juxta)
    
    if comm.rank == 0:
        
        # Save number of false negatives due to threshold in BEER file.
        beer_path = "{}.beer.hdf5".format(file_out_suff)
        beer_file = h5py.File(beer_path, 'a', libver='latest')
        beer_key = 'thresh_fn'
        if beer_key in beer_file.keys():
            beer_file.pop(beer_key)
        beer_file.create_dataset(beer_key, data=threshold_false_negatives)
        beer_file.close()
        
    
    ##### TODO: clean temporary zone
    # Ground truth spike times defined from the juxtacellular traces.
    mask_gt = matched_spike_times_juxta # keep all the 'ground truth' spike times
    spike_times_gt = spike_times_juxta[mask_gt]
    # # or
    # # Ground truth spike times defined from the extracellular traces.
    # mask_gt = matched_spike_times_extra
    # spike_times_gt = spike_times_extra[mask_gt]
    ##### end temporary zone
    spike_times_gt = numpy.sort(spike_times_gt)
    
    ##### end working zone
    
    
    if comm.rank == 0:
        if verbose:
            msg = [
                "spike_times_gt.size: {}".format(spike_times_gt.size),
            ]
            print_and_log(msg, level='default', logger=logger)
    
    
    ##### TODO: clean working zone
    # TODO: share memory for these data structures.
    
    labels_gt = numpy.zeros(spike_times_gt.size)
    ##### TODO: clean test zone
    if SHARED_MEMORY:
        spikes_gt = get_stas_memshared(params, spike_times_gt, labels_gt, chan, chans, nodes=nodes, auto_align=False).T
    else:
        spikes_gt = get_stas(params, spike_times_gt, labels_gt, chan, chans, nodes=nodes, auto_align=False).T
    ##### end test zone
    
    # Reshape data.
    N_t = spikes_gt.shape[0]
    N_e = spikes_gt.shape[1]
    N_gt = spikes_gt.shape[2]
    ##### TODO: remove temporary zone
    # if comm.rank == 0:
    #     print(spikes_gt.flags)
    # spikes_gt.shape = (N_t, N_e * N_gt)
    ##### end temporary zone
    spikes_gt = spikes_gt.reshape(N_t, N_e * N_gt)
    spikes_gt = spikes_gt.T
    # Compute the PCA coordinates of each spike of the "ground truth cell".
    X_gt = numpy.dot(spikes_gt, basis_proj)
    X_gt = X_gt.T
    # Reshape data.
    X_gt = X_gt.reshape(N_p * N_e, N_gt)
    X_gt = X_gt.T
    
    # Define the outputs (i.e. 0 for ground truth samples).
    y_gt = numpy.zeros((N_gt, 1), dtype=numpy.float32)
    
    
    if comm.rank == 0:
        if verbose:
            msg = [
                "X_gt.shape: {}".format(X_gt.shape),
                "y_gt.shape: {}".format(y_gt.shape),
            ]
            print_and_log(msg, level='default', logger=logger)
    
    ##### end working zone
    
    
    
    ##### NON GROUND TRUTH CELL'S SAMPLES ######################################
    
    if comm.rank == 0:
        print_and_log(["Collecting non ground truth cells' samples..."], level='debug', logger=logger)
    
    
    mask_ngt = numpy.logical_or(matched_spike_times_extra, mismatched_spike_times_extra)
    mask_ngt = numpy.logical_not(mask_ngt)
    spike_times_ngt = spike_times_extra[mask_ngt]
    ##### TODO: clean temporary zone
    # Select a subset of the spike times if they are too many.
    max_spike_times_ngt = 10000
    if max_spike_times_ngt <= spike_times_ngt.size:
        if comm.Get_rank() == 0 and verbose:
            msg = [
                "Number of 'non ground truth' spike times too high (i.e. {}), limitation to {}.".format(spike_times_ngt.size, max_spike_times_ngt),
            ]
            print_and_log(msg, level='default', logger=logger)
        spike_times_ngt = numpy.random.choice(spike_times_ngt, size=max_spike_times_ngt, replace=False)
    ##### end temporary zone
    spike_times_ngt = numpy.sort(spike_times_ngt)
    
    if comm.rank == 0:
        if verbose:
            msg = [
                "spike_times_ngt.size: {}".format(spike_times_ngt.size),
            ]
            print_and_log(msg, level='default', logger=logger)
    
    
    ##### TODO: clean working zone
    
    labels_ngt = numpy.zeros(spike_times_ngt.size)
    ##### TODO: clean temporary zone
    if SHARED_MEMORY:
        spikes_ngt = get_stas_memshared(params, spike_times_ngt, labels_ngt, chan, chans, nodes=nodes, auto_align=False).T
    else:
        spikes_ngt = get_stas(params, spike_times_ngt, labels_ngt, chan, chans, nodes=nodes, auto_align=False).T
    ##### TODO: end temporary zone
    
    # Reshape data.
    N_t = spikes_ngt.shape[0]
    N_e = spikes_ngt.shape[1]
    N_ngt = spikes_ngt.shape[2]
    ##### TODO: remove temporary zone
    # if comm.rank == 0:
    #     print(spikes_ngt.flags)
    # spikes_ngt.shape = (N_t, N_e * N_ngt)
    ##### end temporary zone
    spikes_ngt = spikes_ngt.reshape(N_t, N_e * N_ngt)
    spikes_ngt = spikes_ngt.T
    # Compute the PCA coordinates of each spike of the "non ground truth cells".
    X_ngt = numpy.dot(spikes_ngt, basis_proj)
    X_ngt = X_ngt.T
    # Reshape data.
    X_ngt = X_ngt.reshape(N_p * N_e, N_ngt)
    X_ngt = X_ngt.T
    
    # Define the outputs (i.e. 1 for non ground truth samples).
    y_ngt = numpy.ones((N_ngt, 1))
    
    
    if comm.rank == 0:
        if verbose:
            msg = [
                "X_ngt.shape: {}".format(X_ngt.shape),
                "y_ngt.shape: {}".format(y_ngt.shape),
            ]
            print_and_log(msg, level='default', logger=logger)
    
    # import time
    # secs = 10.0 # s
    # if comm.rank == 0:
    #     print("Start to sleep...")
    # time.sleep(secs)
    # sys.exit(1)
    
    ##### end working zone
    
    
    
    # NORMALIZE DATASETS #######################################################
    
    if comm.rank == 0:
        print_and_log(["Normalizing datasets..."], level='debug', logger=logger)
    
    
    X_raw = numpy.vstack((X_gt, X_ngt))
    norm_scale = numpy.mean(numpy.linalg.norm(X_raw, axis=1))
    X_gt  /= norm_scale
    X_ngt /= norm_scale
    
    
    
    ##### SAMPLES ##############################################################
    
    if comm.rank == 0:
        print_and_log(["Samples..."], level='debug', logger=logger)
    
    
    # Option to include the pairwise product of feature vector elements.
    pairwise = True
    
    # Create the datasets to train the neural network.
    ## Create the input dataset.
    N = X_gt.shape[1]
    
    X_raw = numpy.vstack((X_gt, X_ngt))
    
    if pairwise:
        # With pairwise product of feature vector elements.
        M = N + N * (N + 1) // 2
        shape = (N_gt + N_ngt, M)
    else:
        # Without pairwise product of feature vector elments.
        M = N
        shape = (N_gt + N_ngt, M)
    X = numpy.zeros(shape, dtype=numpy.float32)
    X[:, :N] = X_raw
    
    if pairwise:
        # Add the pairwise product of feature vector elements.
        k = 0
        for i in xrange(0, N):
            for j in xrange(i, N):
                X[:, N + k] = numpy.multiply(X[:, i], X[:, j])
                k = k + 1
        
        if comm.rank == 0:
            if verbose:
                msg = [
                    "X.shape (with pairwise product of feature vector element): {}".format(X.shape),
                ]
                print_and_log(msg, level='default', logger=logger)
    
    ## Create the output dataset.
    y_raw = numpy.vstack((y_gt, y_ngt))
    y_raw = y_raw.ravel()
    y = y_raw
    
    if comm.rank == 0:
        if verbose:
            msg = [
                "X_raw.shape: {}".format(X_raw.shape),
                "y_raw.shape: {}".format(y_raw.shape),
                "X.shape: {}".format(X.shape),
                "y.shape: {}".format(y.shape),
            ]
            print_and_log(msg, level='default', logger=logger)
    
    
    
    ##### SANITY PLOT ##########################################################
    
    if comm.rank == 0:
        
        #print_and_log(["Sanity plot..."], level='info', logger=params)
        
        
        if make_plots not in ['None', '']:
            plot_filename = "beer-datasets.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            xs = [X_gt, X_ngt]
            ys = [y_gt, y_ngt]
            colors = ['r', 'b']
            labels = ["GT", "Non GT"]
            plot.view_datasets(params, xs, ys, [spike_times_gt, spike_times_ngt], colors=colors, labels=labels, save=path)
    
    
    
    ##### INITIAL PARAMETER ####################################################
    
    if comm.rank == 0:
        print_and_log(["Initializing parameters for the non-linear classifier..."], level='default', logger=logger)
    
    
    mu = numpy.mean(X_gt.T, axis=1)
    sigma = numpy.cov(X_gt.T)
    k = 1.0
    
    sigma_inv = numpy.linalg.inv(sigma)
    
    A_init = sigma_inv
    b_init = - 2.0 * numpy.dot(mu, sigma_inv)
    c_init = numpy.dot(mu, numpy.dot(sigma_inv, mu)) - k * k
    
    coefs_init = ellipsoid_matrix_to_coefs(A_init, b_init, c_init)
    
    if comm.rank == 0:
        if verbose:
            msg = [
                "coefs_init: {}".format(coefs_init),
            ]
            print_and_log(msg, level='default', logger=logger)
    
    
    # Compute false positive rate and true positive rate for various cutoffs.
    num = 300
    
    Mhlnb_gt = squared_Mahalanobis_distance(A_init, mu, X_gt)
    Mhlnb_ngt = squared_Mahalanobis_distance(A_init, mu, X_ngt)
    
    Mhlnb = numpy.concatenate((Mhlnb_gt, Mhlnb_ngt))
    Mhlnb = numpy.unique(Mhlnb)
    Mhlnb = numpy.sort(Mhlnb)
    indices = numpy.linspace(0, Mhlnb.size - 1, num=num)
    indices = indices.astype('int')
    cutoffs = numpy.zeros(num)
    fprs = numpy.zeros(num)
    tprs = numpy.zeros(num)
    for (i, index) in enumerate(indices):
        cutoffs[i] = Mhlnb[index]
        cutoff = cutoffs[i]
        fp = float(numpy.count_nonzero(Mhlnb_ngt < cutoff))
        n = float(Mhlnb_ngt.size)
        fprs[i] = fp / n
        tp = float(numpy.count_nonzero(Mhlnb_gt < cutoff))
        p = float(Mhlnb_gt.size)
        tprs[i] = tp / p
    
    if comm.rank == 0:
        if verbose:
            # msg = [
            #     "cutoffs: {}".format(cutoffs),
            #     "fprs: {}".format(fprs),
            #     "tprs: {}".format(tprs),
            # ]
            # print_and_log(msg, level='default', logger=params)
            pass
    
    # Compute mean acccuracy for various cutoffs.
    accs = numpy.zeros(num)
    for (i, index) in enumerate(indices):
        cutoff = Mhlnb[index]
        tp = float(numpy.count_nonzero(Mhlnb_gt <= cutoff))
        p = float(Mhlnb_gt.size)
        tn = float(numpy.count_nonzero(cutoff < Mhlnb_ngt))
        n = float(Mhlnb_ngt.size)
        accs[i] = (tp + tn) / (p + n)
    
    # Find the optimal cutoff.
    i_opt = numpy.argmax(accs)
    cutoff_opt_acc = cutoffs[i_opt]
    
    if comm.rank == 0:
        if verbose:
            msg = [
                "cutoff_opt_acc: {}".format(cutoff_opt_acc),
                "acc_opt: {}".format(accs[i_opt]),
            ]
            print_and_log(msg, level='default', logger=logger)
        
        #if make_plots not in ['None', '']:
        #    # Plot accuracy curve.
        #    title = "Accuracy curve for the initial parameter"
        #    plot_filename = "beer-accuracy-plot.%s" %make_plots
        #    path = os.path.join(plot_path, plot_filename)
        #    plot.view_accuracy(Mhlnb[indices], accs, Mhlnb[indices[i_opt]],
        #                       accs[i_opt], title=title, save=path)
    
    # Compute the normalized accuracy for various cutoffs.
    tprs = numpy.zeros(num)
    tnrs = numpy.zeros(num)
    norm_accs = numpy.zeros(num)
    for (i, index) in enumerate(indices):
        cutoff = Mhlnb[index]
        tp = float(numpy.count_nonzero(Mhlnb_gt <= cutoff))
        p = float(Mhlnb_gt.size)
        tpr = tp / p
        tn = float(numpy.count_nonzero(cutoff < Mhlnb_ngt))
        n = float(Mhlnb_ngt.size)
        tnr = tn / n
        tprs[i] = tpr
        tnrs[i] = tnr
        norm_accs[i] = 0.5 * (tpr + tnr)
    
    # Find the optimal cutoff.
    i_opt = numpy.argmax(norm_accs)
    cutoff_opt_norm_acc = cutoffs[i_opt]
    
    if comm.rank == 0:
        if verbose:
            msg = [
                "cutoff_opt_norm_acc: {}".format(cutoff_opt_norm_acc),
                "norm_acc_opt: {}".format(norm_accs[i_opt]),
            ]
            print_and_log(msg, level='default', logger=logger)
        
        if make_plots not in ['None', '']:
            # Plot normalized accuracy curve.
            title = "Normalized accuracy curve for the initial parameter"
            plot_filename = "beer-accuray.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)

            data1 = Mhlnb[indices], accs, Mhlnb[indices[i_opt]], accs[i_opt]
            data2 = Mhlnb[indices], tprs, tnrs, norm_accs, Mhlnb[indices[i_opt]], norm_accs[i_opt]

            plot.view_accuracy(data1, data2, title=title, save=path)
    
    # Set cutoff equal to the optimal cutoff.
    # cutoff = cutoff_opt_acc
    cutoff = cutoff_opt_norm_acc
    
    # Compute false positive rate and true positive rate for the chosen cutoff.
    fp = float(numpy.count_nonzero(Mhlnb_ngt < cutoff))
    n = float(Mhlnb_ngt.size)
    fpr = fp / n
    tp = float(numpy.count_nonzero(Mhlnb_gt < cutoff))
    p = float(Mhlnb_gt.size)
    tpr = tp / p
    
    if comm.rank == 0:
        if verbose:
            msg = [
                "cutoff: {}".format(cutoff),
                "fpr: {}".format(fpr),
                "tpr: {}".format(tpr),
            ]
            print_and_log(msg, level='default', logger=logger)
        
        #if make_plots not in ['None', '']:
            # Plot ROC curve.
        #    title = "ROC curve for the inital parameter"
        #    plot_filename = "beer-roc-curve-initial.%s" %make_plots
        #    path = os.path.join(plot_path, plot_filename)
        #    plot.view_roc_curve(fprs, tprs, fpr, tpr, title=title, save=path)
    
    # Scale the ellipse according to the chosen cutoff.
    A_init = (1.0 / cutoff) * A_init
    b_init = (1.0 / cutoff) * b_init
    c_init = (1.0 / cutoff) * (c_init + 1.0) - 1.0
    
    
    
    # SANITY PLOT (CLASSIFIER PROJECTION) ######################################
    
    if comm.rank == 0:
        print_and_log(["Sanity plot (classifier projection)..."],
                         level='debug', logger=logger)
        
        
        if make_plots not in ['None', '']:
            # Plot initial classifier (ellipsoid).
            #title = "Initial classifier (ellipsoid)"
            #plot_filename = "beer-classifier-projection-init.%s" %make_plots
            #path = os.path.join(plot_path, plot_filename)
            
            data_class_1 = [X_gt, X_ngt], [y_gt, y_ngt], A_init, b_init, c_init
    
    
    
    # MAHALANOBIS DISTRIBUTIONS ################################################
    
    if comm.rank == 0:
        
        print_and_log(["Intialising Mahalanobis distributions..."],
                         level='debug', logger=logger)
        
        
        # Compute mahalanobis distributions.
        mu = numpy.mean(X_gt, axis=0)
        Mhlnb_gt = squared_Mahalanobis_distance(A_init, mu, X_gt)
        Mhlnb_ngt = squared_Mahalanobis_distance(A_init, mu, X_ngt)
        
        data_mal1 = Mhlnb_gt, Mhlnb_ngt
    
    
    
    ##### LEARNING #############################################################
    
    # mode = 'decision'
    mode = 'prediction'
    
    model = 'sgd'
    
    def split_train_test(X, y, test_size=0.5, seed=0):
        size = X.shape[0]
        indices = numpy.random.permutation(size)
        thresh = int((1.0 - test_size) * float(size))
        indices_train = indices[:thresh]
        indices_test = indices[thresh:]
        return indices_train, indices_test
    
    indices_train, indices_test = split_train_test(X, y, test_size=0.3)
    X_train = X[indices_train, :]
    X_test = X[indices_test, :]
    y_train = y[indices_train]
    y_test = y[indices_test]
    
    if comm.rank == 0:
        print_and_log(["Start learning..."], level='debug', logger=logger)
    
    
    if comm.rank == 0:
        if verbose:
            msg = [
                "X_train.shape: {}".format(X_train.shape),
                "X_test.shape: {}".format(X_test.shape),
            ]
            print_and_log(msg, level='default', logger=logger)
    
    # Declare model.
    if model == 'mlp':
        clf = MLPClassifier(hidden_layer_sizes=(),
                            activation='logistic',
                            algorithm='sgd',
                            alpha=1.0e-12,
                            tol=1.0e-8,
                            learning_rate='adaptive',
                            random_state=0,
                            momentum=0.05,
                            nesterovs_momentum=False)
    elif model == 'perceptron':
        clf = Perceptron(penalty='l2',
                         alpha=1.0e-12,
                         fit_intercept=True,
                         random_state=0)
    elif model == 'sgd':
        ##### TODO: clean temporary zone
        _, _, class_weights = get_class_weights(y_gt, y_ngt, n=1)
        # _, _, class_weights = get_class_weights_bis(n_class_0, n_class_1, n=1)
        ##### end temporary zone
        clf = SGDClassifier(loss='log',
                            fit_intercept=True,
                            random_state=2,
                            learning_rate='optimal',
                            eta0=sys.float_info.epsilon,
                            class_weight=class_weights[0])
    
    # Initialize model (i.e. fake launch, weights initialization).
    if model == 'mlp':
        clf.set_params(max_iter=1)
        clf.set_params(learning_rate_init=sys.float_info.epsilon)
        clf.set_params(warm_start=False)
    elif model == 'perceptron' or model == 'sgd':
        clf.set_params(n_iter=1)
        clf.set_params(eta0=sys.float_info.epsilon)
        clf.set_params(warm_start=False)
    clf.fit(X_train, y_train)
    
    # if comm.rank == 0:
    #     if make_plots not in ['None', '']:
    #        # Plot prediction.
    #        title = "Initial prediction (random)"
    #        plot_filename = "beer-prediction-init-random.%s" %make_plots
    #        path = os.path.join(plot_path, plot_filename)
    #        plot.view_classification(clf, X, X_raw, y_raw, mode='predict',
    #                                 title=title, save=path)
    #        # Plot decision function.
    #        title = "Initial decision function (random)"
    #        plot_filename = "beer-decision-function-init-random.%s" %make_plots
    #        path = os.path.join(plot_path, plot_filename)
    #        plot.view_classification(clf, X, X_raw, y_raw, mode='decision_function',
    #                                 title=title, save=path)
    
    if comm.rank == 0:
        if verbose:
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred, class_weights=class_weights[0])
            msg = [
                # # Print the current loss.
                # "clf.loss_: {}".format(clf.loss_),
                # # Print the loss curve.
                # "clf.loss_curve_: {}".format(clf.loss_curve_),
                # # Print the number of iterations the algorithm has ran.
                # "clf.n_iter_: {}".format(clf.n_iter_),
                # Print the score on the test set.
                "accuracy_score(X_test, y_test): {} ({})".format(score, 1.0 - score),
            ]
            print_and_log(msg, level='default', logger=logger)
    
    coefs_init = ellipsoid_matrix_to_coefs(A_init, b_init, c_init)
    if model == 'mlp':
        clf.coefs_ = [coefs_init[1:, :]]
        clf.intercepts_ = [coefs_init[:1, :]]
    elif model == 'perceptron' or model == 'sgd':
        clf.coef_ = coefs_init[1:, :].reshape(1, -1)
        clf.intercept_ = coefs_init[:1, :].ravel()
    
    if comm.rank == 0:
        if make_plots not in ['None', '']:
            # Plot prediction.
            #title = "Initial prediction (ellipsoid)"
            #plot_filename = "beer-prediction-init-ellipsoid.%s" %make_plots
            #path = os.path.join(plot_path, plot_filename)
            #plot.view_classification(clf, X, X_raw, y_raw, mode='predict',
            #                         title=title, save=path)
            
            pred_1 = clf.predict(X), clf.decision_function(X), X, X_raw, y_raw
            # Plot decision function.
            #title = "Initial decision function (ellipsoid)"
            #plot_filename = "beer-decision-function-init-ellipsoid.%s" %make_plots
            #path = os.path.join(plot_path, plot_filename)
            #plot.view_classification(clf, X, X_raw, y_raw, mode='decision_function',
            #                         title=title, save=path)
    
    if comm.rank == 0:
        if verbose:
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred, class_weights=class_weights[0])
            msg = [
                # # Print the current loss.
                # "clf.loss_: {}".format(clf.loss_),
                # # Print the loss curve.
                # "clf.loss_curve_: {}".format(clf.loss_curve_),
                # # Print the number of iterations the algorithm has ran.
                # "clf.n_iter_: {}".format(clf.n_iter_),
                # Print the score on the test set.
                "accuracy_score(X_test, y_test): {} ({})".format(score, 1.0 - score),
            ]
            print_and_log(msg, level='default', logger=logger)
    
    # Train model.
    if model == 'mlp':
        clf.set_params(max_iter=max_iter)
        clf.set_params(learning_rate_init=learning_rate_init)
        clf.set_params(warm_start=True)
    elif model == 'perceptron' or model == 'sgd':
        n_iter = min(max_iter, 1000000 // N_max)
        clf.set_params(n_iter=n_iter)
        clf.set_params(eta0=learning_rate_init)
        clf.set_params(warm_start=True)
    clf.fit(X_train, y_train)
    
    if comm.rank == 0:
        if make_plots not in ['None', '']:
            # Plot final prediction.
            #title = "Final prediction "
            #plot_filename = "beer-prediction-final.%s" %make_plots
            #path = os.path.join(plot_path, plot_filename)
            #plot.view_classification(clf, X, X_raw, y_raw, mode='predict',
            #                         title=title, save=path)
            # Plot final decision function.
            #title = "Final decision function"
            plot_filename = "beer-decision.%s" %make_plots
            pred_2 = clf.predict(X), clf.decision_function(X), X, X_raw, y_raw
            
            path = os.path.join(plot_path, plot_filename)
            plot.view_classification(pred_1, pred_2, save=path)
    
    if comm.rank == 0:
        if verbose:
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred, class_weights=class_weights[0])
            msg = [
                # # Print the current loss computed with the loss function.
                # "clf.loss_: {}".format(clf.loss_),
                # # Print the loss curve.
                # "clf.loss_curve_: {}".format(clf.loss_curve_),
                # # Print the number of iterations the algorithm has ran.
                # "clf.n_iter_: {}".format(clf.n_iter_),
                # Print the score on the test set.
                "accuracy_score(X_test, y_test): {} ({})".format(score, 1.0 - score),
            ]
            print_and_log(msg, level='default', logger=logger)
    
    
    # # TODO: uncomment (i.e. compute loss curve for perceptron and sgd)
    # #       should find a replacement since perceptron and sgd do not give
    # #       access to the loss values.
    # if make_plots:
    #     # Plot the loss curve.
    #     plot_filename = "loss-curve.png"
    #     path = os.path.join(plot_path, plot_filename)
    #     plot.view_loss_curve(clf.loss_curve_, save=path)
    
    
    # Retrieve the coefficients of the ellipsoid.
    if model == 'mlp':
        bias = clf.intercepts_[0].flatten()
        weights = clf.coefs_[0].flatten()
    elif model == 'perceptron' or model == 'sgd':
        bias = clf.intercept_.flatten()
        weights = clf.coef_.flatten()
    # Concatenate the coefficients.
    coefs = numpy.concatenate((bias, weights))
    coefs = coefs.reshape(-1, 1)
    
    
    A, b, c = ellipsoid_coefs_to_matrix(coefs)
    
    
    
    # SANITY PLOT (CLASSIFIER PROJECTION) ######################################
    
    if comm.rank == 0:
        
        print_and_log(["Sanity plot (classifier projection)..."],
                         level='debug', logger=logger)
        
        
        if make_plots not in ['None', '']:
            # Plot final classifier.
            title = "Final classifier"
            plot_filename = "beer-classifier-projection.%s" %make_plots
            
            data_class_2 = [X_gt, X_ngt], [y_gt, y_ngt], A, b, c
            
            path = os.path.join(plot_path, plot_filename)
            plot.view_classifier(params, data_class_1, data_class_2, save=path, verbose=verbose)
    
    
    
    # MAHALANOBIS DISTRIBUTIONS ################################################
    
    if comm.rank == 0:
        
        print_and_log(["Computing final Mahalanobis distributions..."],
                         level='debug', logger=logger)
        
        
        # Compute the Mahalanobis distributions.
        mu = numpy.mean(X_gt, axis=0)
        Mhlnb_gt = squared_Mahalanobis_distance(A, mu, X_gt)
        Mhlnb_ngt = squared_Mahalanobis_distance(A, mu, X_ngt)
        
        data_mal2 = (Mhlnb_gt, Mhlnb_ngt)
        
        if verbose:
            msg = [
                "# Mhlnb_gt: {}".format(Mhlnb_gt),
            ]
            print_and_log(msg, level='default', logger=logger)
        
        if make_plots not in ['None', '']:
            # Plot Mahalanobis distributions.
            title = "Final Mahalanobis distributions"
            plot_filename = "beer-mahalanobis.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            plot.view_mahalanobis_distribution(data_mal1, data_mal2, save=path)
    
    
    
    # Synchronize CPUs before weighted learning.
    comm.Barrier()
    
    
    
    ##### WEIGHTED LEARNING ####################################################
    
    if comm.rank == 0:
        print_and_log(["Estimating the ROC curve..."], level='default', logger=logger)
    
    
    _, _, class_weights = get_class_weights(y_gt, y_ngt, n=roc_sampling)
    
    # Distribute weights over the CPUs.
    loc_indices = numpy.arange(comm.rank, roc_sampling, comm.size)
    loc_class_weights = [class_weights[loc_index] for loc_index in loc_indices]
    loc_nb_class_weights = len(loc_class_weights)
    
    # Preallocation to collect results.
    confusion_matrices = loc_nb_class_weights * [None]
    y_decfs = loc_nb_class_weights * [None]
    y_preds = loc_nb_class_weights * [None]
    
    to_explore = loc_class_weights

    if comm.rank == 0:
        to_explore = get_tqdm_progressbar(to_explore)
    
    if model == 'sgd':
        for (count, class_weight) in enumerate(to_explore):
            # Declare classifier.
            wclf = SGDClassifier(loss='log',
                                 # penalty='l2',
                                 # alpha=1.0e-12,
                                 fit_intercept=True,
                                 random_state=0,
                                 # learning_rate='constant',
                                 learning_rate='optimal',
                                 # eta0=sys.float_info.epsilon,
                                 class_weight=class_weight)
            # Initialize classifier (i.e. fake launch, weights initialization).
            wclf.set_params(n_iter=1)
            # wclf.set_params(eta0=sys.float_info.epsilon)
            wclf.set_params(warm_start=False)
            wclf.fit(X_train, y_train)
            # Initialize classifier (i.e. ellipsoid weights).
            coefs_init = ellipsoid_matrix_to_coefs(A_init, b_init, c_init)
            wclf.coef_ = coefs_init[1:, :].reshape(1, -1)
            wclf.intercept_ = coefs_init[:1, :].ravel()
            # Train classifier.
            n_iter = min(max_iter, 1000000 // N_max)
            wclf.set_params(n_iter=n_iter)
            # wclf.set_params(eta0=learning_rate_init)
            wclf.set_params(warm_start=True)
            wclf.fit(X_train, y_train)
            
            
            ##### TODO: fix depreciated zone
            
            # Compute the prediction on the test set.
            y_pred = wclf.predict(X_test)
            y_decf = wclf.decision_function(X_test)
            # Compute true positive, false negative, true negatives and
            # false positives.
            p = (y_test == 0.0)
            tp = float(numpy.count_nonzero(y_pred[p] == y_test[p]))
            fn = float(numpy.count_nonzero(y_pred[p] != y_test[p]))
            n = (y_test == 1.0)
            tn = float(numpy.count_nonzero(y_pred[n] == y_test[n]))
            fp = float(numpy.count_nonzero(y_pred[n] != y_test[n]))
            # Construct the confusion matrix.
            confusion_matrix = numpy.array([[tp, fn], [fp, tn]])
            # Save results.
            y_preds[count] = y_pred
            y_decfs[count] = y_decf
            confusion_matrices[count] = confusion_matrix
            
            
    else:
        raise Exception("Unsupported classifier: model={}".format(model))
        
    comm.Barrier()
    
    
    # Gather results on the root CPU.
    indices = comm.gather(loc_indices, root=0)
    y_preds_tmp = comm.gather(y_preds, root=0)
    y_decfs_tmp = comm.gather(y_decfs, root=0)
    if test_method == 'full':
        time_preds_tmp = comm.gather(time_preds, root=0)
    
    if comm.rank == 0:
        
        beer_filename = "{}.beer.hdf5".format(file_out_suff)
        beer_file = h5py.File(beer_filename, 'a', libver='latest')
        group_name = "beer_spiketimes"
        if group_name in beer_file.keys():
            beer_file.pop(group_name)
        beer_file.create_group(group_name)
        if test_method == 'full':
            for indices_, loc_time_preds, loc_y_preds, loc_y_decfs in zip(indices, time_preds_tmp, y_preds_tmp, y_decfs_tmp):
                for index, time_pred, y_pred, y_decf in zip(indices_, loc_time_preds, loc_y_preds, loc_y_decfs):
                    filename = "{}/time_pred_{}".format(group_name, index)
                    #print(filename)
                    beer_file.create_dataset(filename, data=time_pred)
                    filename = "{}/y_pred_{}".format(group_name, index)
                    #print(filename)
                    beer_file.create_dataset(filename, data=y_pred)
                    filename = "{}/y_decf_{}".format(group_name, index)
                    #print(filename)
                    beer_file.create_dataset(filename, data=y_decf)
                    filename = "{}/temp_{}".format(group_name, index)
                    #print(filename)
                    mask = (y_pred == 0.0)
                    temp = time_pred[mask]
                    beer_file.create_dataset(filename, data=temp)
        elif test_method == 'downsampled':
            for indices_, loc_y_preds, loc_y_decfs in zip(indices, y_preds_tmp, y_decfs_tmp):
                for index, y_pred, y_decf in zip(indices_, loc_y_preds, loc_y_decfs):
                    filename = "{}/y_pred_{}".format(group_name, index)
                    #print(filename)
                    beer_file.create_dataset(filename, data=y_pred)
                    filename = "{}/y_decf_{}".format(group_name, index)
                    #print(filename)
                    beer_file.create_dataset(filename, data=y_decf)
        beer_file.close()
    
    ##### end temporary zone
    
    
    # Gather results on the root CPU.
    indices = comm.gather(loc_indices, root=0)
    confusion_matrices_tmp = comm.gather(confusion_matrices, root=0)
    
    if comm.Get_rank() == 0:
        
        # Reorder confusion matrices properly.
        confusion_matrices = roc_sampling * [None]
        for (loc_indices, loc_confusion_matrices_tmp) in zip(indices, confusion_matrices_tmp):
            for (loc_index, loc_confusion_matrix) in zip(loc_indices, loc_confusion_matrices_tmp):
                confusion_matrices[loc_index] = loc_confusion_matrix
        # Save confusion matrices to BEER file.
        filename = "{}.beer.hdf5".format(file_out_suff)
        beer_file = h5py.File(filename, 'a', libver='latest')
        ## Save class weights.
        class_weights_ = numpy.array([[cw[0], cw[1]] for cw in class_weights])
        class_weights_key = "class_weights"
        if class_weights_key in beer_file.keys():
            beer_file.pop(class_weights_key)
        beer_file.create_dataset(class_weights_key, data=class_weights_)
        ## Save confusion matrices.
        confusion_matrices_ = numpy.array(confusion_matrices)
        confusion_matrices_key = "confusion_matrices"
        if confusion_matrices_key in beer_file.keys():
            beer_file.pop(confusion_matrices_key)
        beer_file.create_dataset(confusion_matrices_key, data=confusion_matrices_)
        beer_file.close()
        # Compute false positive rates and true positive rates.
        fprs = [M[1, 0] / (M[1, 0] + M[1, 1]) for M in confusion_matrices]
        tprs = [M[0, 0] / (M[0, 0] + M[0, 1]) for M in confusion_matrices]
        # Add false positive rates and true positive rates endpoints.
        fprs = [1.0] + fprs + [0.0]
        tprs = [1.0] + tprs + [0.0]
    
    
    if comm.rank == 0:
        
        ##### TODO: clean temporary zone
        
        MODE = 'custom'
        # MODE = 'harris'
        
        if MODE == 'custom':
            
            # Define the "matching threshold".
            thresh = int(float(params.rate) * matching_jitter * 1.0e-3)
            
            # Retrieve the SpyKING CIRCUS spiketimes.
            result = io.load_data(params, "results")
            data   = result['spiketimes']
            
            # Retrieve the templates.
            templates = io.load_data(params, 'templates')
            
            n_temp = len(data)
            res = numpy.zeros((n_temp, 2))
            sc_contingency_matrices = numpy.zeros((n_temp, 2, 2), dtype=numpy.int)
            
            # First pass to detect what are the scores.
            for i in xrange(n_temp):
                # Retrieve the spike times for the i-th detected template.
                spike_times = data['temp_' + str(i)]
                # Count the true positives (among actual posititves).
                for spike_time_gt in spike_times_gt:
                    idx = numpy.where(abs(spike_times - spike_time_gt) <= thresh)[0]
                    if 0 < len(idx):
                        # There is at least one spike for this template next to this ground truth spike.
                        res[i, 0] += 1.0
                        sc_contingency_matrices[i, 0, 0] += 1
                # Count the false negatives.
                sc_contingency_matrices[i, 0, 1] = spike_times_gt.size - sc_contingency_matrices[i, 0, 0]
                # Compute the true positive rate (for actual positives).
                if 0 < spike_times_gt.size:
                    res[i, 0] /= float(spike_times_gt.size)
                # Count the true positives (among expected positives).
                for k, spike_time in enumerate(spike_times):
                    idx = numpy.where(abs(spike_times_gt - spike_time) <= thresh)[0]
                    if 0 < len(idx):
                        res[i, 1] += 1.0
                        sc_contingency_matrices[i, 1, 0] += 1
                # Count the false positives.
                matched_spike_times = numpy.zeros_like(spike_times, dtype='bool')
                for k, spike_time in enumerate(spike_times):
                    idx = numpy.where(abs(spike_times_gt - spike_time) <= thresh)[0]
                    if 0 < len(idx):
                        matched_spike_times[k] = True
                    else:
                        idx = numpy.where(abs(spike_times_ngt - spike_time) <= thresh)[0]
                        if 0 < len(idx):
                            matched_spike_times[k] = True
                matched_spike_times = spike_times[matched_spike_times]
                sc_contingency_matrices[i, 1, 1] = matched_spike_times.size - sc_contingency_matrices[i, 1, 0]
                # Compute the true positive rate (for expected positives).
                if 0 < matched_spike_times.size:
                    res[i, 1] /= float(matched_spike_times.size)
            
            idx = numpy.argmax(numpy.mean(res, 1))
            selection = [idx]
            error = res[idx]
            sc_contingency_matrix = sc_contingency_matrices[idx, :, :]
            find_next = True
            source_temp = templates[:, idx].toarray().flatten()
            temp_match = []
            dmax = 0.1
            for i in xrange(templates.shape[1]/2):
                d = numpy.corrcoef(templates[:, i].toarray().flatten(), source_temp)[0, 1]
                if d > dmax and i not in selection:
                    temp_match += [i]
            
            ## Second pass to reach the best score with greedy aggregations
            if 0 < len(temp_match):
                
                while (find_next == True):
                    
                    temp_match = [i for i in temp_match if i not in selection]

                    nb_temps = len(temp_match)
                    local_errors = numpy.zeros((nb_temps, 2))
                    local_sc_contingency_matrices = numpy.zeros((nb_temps, 2, 2), dtype=numpy.int)
                    
                    for mcount, tmp in enumerate(temp_match):
                        
                        # Gather selected spikes.
                        spike_times = []
                        for xtmp in selection + [tmp]:
                            spike_times += data['temp_' + str(xtmp)].tolist()
                        spike_times = numpy.array(spike_times, dtype=numpy.int32)
                        
                        # Count true positive (among actual positives).
                        count = 0
                        for spike_time_gt in spike_times_gt:
                            idx = numpy.where(numpy.abs(spike_times - spike_time_gt) < thresh)[0]
                            if 0 < len(idx):
                                count += 1
                                local_sc_contingency_matrices[mcount, 0, 0] += 1
                        # Count false negatives.
                        local_sc_contingency_matrices[mcount, 0, 1] = spike_times_gt.size - local_sc_contingency_matrices[mcount, 0, 0]
                        # Compute true positive rate (for actual positives).
                        if 0 < spike_times_gt.size:
                            local_errors[mcount, 0] = float(count) / float(spike_times_gt.size)
                        
                        # Count true positives (among expected positives).
                        count = 0
                        for k, spike_time in enumerate(spike_times):
                            idx = numpy.where(numpy.abs(spike_times_gt - spike_time) < thresh)[0]
                            if 0 < len(idx):
                                count += 1
                                local_sc_contingency_matrices[mcount, 1, 0] += 1
                        # Count false positives.
                        matched_spike_times = numpy.zeros_like(spike_times)
                        for k, spike_time in enumerate(spike_times):
                            idx = numpy.where(numpy.abs(spike_times_gt - spike_time) < thresh)[0]
                            if 0 < len(idx):
                                matched_spike_times[k] = True
                            else:
                                idx = numpy.where(numpy.abs(spike_times_ngt - spike_time) < thresh)[0]
                                if 0 < len(idx):
                                    matched_spike_times[k] = True
                        matched_spike_times = spike_times[matched_spike_times]
                        local_sc_contingency_matrices[mcount, 1, 1] = matched_spike_times.size - local_sc_contingency_matrices[mcount, 1, 0]
                        # Compute true positive rate (for expected positives).
                        if 0 < matched_spike_times.size:
                            local_errors[mcount, 1]  = float(count) / float(matched_spike_times.size)
                    
                    errors = numpy.mean(local_errors, 1)
                    if 0 < errors.size and numpy.max(errors) > numpy.mean(error):
                        idx = numpy.argmax(errors)
                        selection += [temp_match[idx]]
                        error = local_errors[idx]
                        sc_contingency_matrix = local_sc_contingency_matrices[idx, :, :]
                    else:
                        find_next = False
            
            error = 100.0 * (1.0 - error)
            res = 100.0 * (1.0 - res)
            
            scerror = dict()
            scerror['res'] = res
            scerror['selection'] = selection
            scerror['error'] = error
            scerror['contingency_matrices'] = sc_contingency_matrices
            scerror['contingency_matrix'] = sc_contingency_matrix
        
        elif MODE == 'harris':
            
            spike_times_gt = spike_times_gt
            
            # Define the "matching threshold".
            thresh = int(float(params.rate) * matching_jitter * 1.0e-3)
            
            # Retrieve the SpyKING CIRCUS spiketimes.
            result = io.load_data(params, "results")
            data   = result['spiketimes']
            
            # Retrieve the templates.
            templates = io.load_data(params, 'templates')
            
            n_temp = len(data)
            res = numpy.zeros((n_temp, 2))
            
            # First pass to detect what are the scores.
            for i in xrange(n_temp):
                spikes = data['temp_' + str(i)]
                # Compute the false positive rate.
                for spike in spike_times_gt:
                    idx = numpy.where(abs(spikes - spike) <= thresh)[0]
                    if 0 < len(idx):
                        res[i, 0] += 1.0
                if 0 < spike_times_gt.size:
                    res[i, 0] /= float(spike_times_gt.size)
                # Compute the positive predictive value.
                for spike in spikes:
                    idx = numpy.where(abs(spike_times_gt - spike) <= thresh)[0]
                    if 0 < len(idx):
                        res[i, 1] += 1.0
                if 0 < spikes.size:
                    res[i, 1] /= float(spikes.size)
            
            idx = numpy.argmax(numpy.mean(res, 1))
            selection = [idx]
            error = res[idx]
            find_next = True
            source_temp = templates[:, idx].toarray().flatten()
            temp_match = []
            dmax = 0.1
            for i in xrange(templates.shape[1]/2):
                d = numpy.corrcoef(templates[:, i].toarray().flatten(), source_temp)[0, 1]
                if d > dmax and i not in selection:
                    temp_match += [i]
            
            ## Second pass to reach the best score with greedy aggregations
            if 0 < len(temp_match):
                
                while (find_next == True):
                    
                    temp_match = [i for i in temp_match if i not in selection]
                    
                    local_errors = numpy.zeros((len(temp_match), 2))
                    
                    for mcount, tmp in enumerate(temp_match):
                        
                        # Gather selected spikes.
                        spikes = []
                        for xtmp in selection + [tmp]:
                            spikes += data['temp_' + str(xtmp)].tolist()
                        spikes = numpy.array(spikes, dtype=numpy.int32)
                        
                        # Compute true positive rate.
                        count = 0.0
                        for spike in spike_times_gt:
                            idx = numpy.where(numpy.abs(spikes - spike) < thresh)[0]
                            if 0 < len(idx):
                                count += 1.0
                        if 0 < spike_times_gt.size:
                            local_errors[mcount, 0] = count / float(spike_times_gt.size)
                        
                        # Compute positive predictive value
                        count = 0.0
                        for spike in spikes:
                            idx = numpy.where(numpy.abs(spike_times_gt - spike) < thresh)[0]
                            if 0 < len(idx):
                                count += 1.0
                        if 0 < spikes.size:
                            local_errors[mcount, 1]  = count / float(spikes.size)
                    
                    errors = numpy.mean(local_errors, 1)
                    if 0 < errors.size and numpy.max(errors) > numpy.mean(error):
                        idx        = numpy.argmax(errors)
                        selection += [temp_match[idx]]
                        error      = local_errors[idx]
                    else:
                        find_next = False
            
            error = 100 * (1 - error)
            res = 100 * (1 - res)
            
            scerror = dict()
            scerror['res'] = res
            scerror['selection'] = selection
            scerror['error'] = error
        
        ##### end temporary zone
    
    
    if comm.Get_rank() == 0:
        
        ##### TODO: clean working zone
        
        # Prepare data to be saved.
        selection = numpy.array(scerror['selection'], dtype=numpy.int)
        sc_contingency_matrices = scerror['contingency_matrices']
        sc_contingency_matrix = scerror['contingency_matrix']
        # Save data to BEER file.
        filename = "{}.beer.hdf5".format(file_out_suff)
        beer_file = h5py.File(filename, 'a', libver='latest')
        ## Save selection.
        beer_key = 'selection'
        if beer_key in beer_file.keys():
            beer_file.pop(beer_key)
        beer_file.create_dataset(beer_key, data=selection)
        ## Save contingency matrices of templates.
        beer_key = 'sc_contingency_matrices'
        if beer_key in beer_file.keys():
            beer_file.pop(beer_key)
        beer_file.create_dataset(beer_key, data=sc_contingency_matrices)
        ## Save contingency matrix of best template.
        beer_key = 'sc_contingency_matrix'
        if beer_key in beer_file.keys():
            beer_file.pop(beer_key)
        beer_file.create_dataset(beer_key, data=sc_contingency_matrix)
        beer_file.close()
        
        ##### end working zone
        
        if verbose:
            msg = [
                "# class_weights: {}".format(class_weights),
                "# false positive rates: {}".format(fprs),
                "# true positive rates: {}".format(tprs),
            ]
            print_and_log(msg, level='default', logger=logger)
        
        if make_plots not in ['None', '']:
            # Plot the ROC curve of the BEER estimate.
            title = "ROC curve of the BEER estimate"
            plot_filename = "beer-roc-curve.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            # plot.view_roc_curve(fprs, tprs, None, None, title=title, save=path,
            #                     xlim=[0.0, 0.25], ylim=[0.75, 1.0])
            # plot.view_roc_curve(fprs, tprs, None, None, title=title, save=path,
            #                     xlim=[0.0, 0.5], ylim=[0.5, 1.0])
            # error = plot.view_roc_curve(params, fprs, tprs, None, None, save=path)
            ##### TODO: clean swap zone
            # error = plot.view_roc_curve(params, fprs, tprs, None, None, scerror=scerror, save=path)
            error = plot.view_roc_curve_(params, save=path)
            ##### end swap zone
            filename = "{}.beer.hdf5".format(file_out_suff)
            beer_file = h5py.File(filename, 'r+', libver='latest')
            if 'circus_error' in beer_file.keys():
                beer_file.pop('circus_error')
            beer_file.create_dataset('circus_error', data=numpy.array(error, dtype=numpy.float32))
            beer_file.close()  
        
    
    ############################################################################
    
    if comm.rank == 0:
        print_and_log(["Validation done."], level='debug', logger=logger)

    return
