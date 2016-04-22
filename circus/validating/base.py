import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, h5py

from ..shared.utils import *
from ..shared.files import get_stas
from ..shared import plot
from datetime import datetime

try:
    import sklearn
except Exception:
    if comm.rank == 0:
        print "Sklearn is not installed! Install spyking-circus with the beer extension (see documentation)"
    sys.exit(0)

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
# TODO: remove following line (i.e. remove warning).
# from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from .utils import *



def main(filename, params, nb_cpu, nb_gpu, use_gpu):
    
    #if comm.rank == 0:
    #    io.print_and_log(["Start validation..."], level='default', logger=params)
    
    
    # RETRIEVE PARAMETERS FOR VALIDATING #######################################
    
    data_file = params.get('data', 'data_file')
    data_offset = params.getint('data', 'data_offset')
    data_dtype = params.get('data', 'data_dtype')
    N_total = params.getint('data', 'N_total')
    sampling_rate = params.getint('data', 'sampling_rate')
    N_e = params.getint('data', 'N_e')
    template_shift = params.getint('data', 'template_shift')
    file_out_suff = params.get('data', 'file_out_suff')
    nb_repeats = params.getint('clustering', 'nb_repeats')
    max_iter = params.getint('validating', 'max_iter')
    learning_rate_init = params.getfloat('validating', 'learning_rate')
    make_plots = params.get('validating', 'make_plots')
    roc_sampling = params.getint('validating', 'roc_sampling')
    plot_path = os.path.join(params.get('data', 'data_file_noext'), 'plots')
    test_size = params.getfloat('validating', 'test_size')
    
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
    alpha_gt = 1.0
    alpha_ngt = 5.0
    alpha_noi = 5.0
    
    if test_method == 'full':
        # Cut data into two halves.
        train_size = 1.0 - test_size
        data_block = numpy.memmap(data_file, offset=data_offset, dtype=data_dtype, mode='r')
        N = len(data_block)
        data_len = N // N_total
        time_min = template_shift
        time_max = int(train_size * float(data_len - 1)) - template_shift
        time_min_test = int(train_size * float(data_len - 1)) + template_shift
        time_max_test = (data_len - 1) - template_shift
        
        print("Time min test: {}".format(time_min_test))
        print("Time max test: {}".format(time_max_test))
    elif test_method == 'downsampled':
        data_block = numpy.memmap(data_file, offset=data_offset, dtype=data_dtype, mode='r')
        N = len(data_block)
        data_len = N // N_total
        time_min = template_shift
        time_max = (data_len - 1) - template_shift
    else:
        raise Exception("Unknown test method: {}".format(test_method))
    
    
    ############################################################################
    
    # TODO: move this section to the right location.
    
    # if comm.rank == 0:
    #     w = 1
        
    #     spike_times_gt = io.load_data(params, 'juxta-triggers')
    #     n_positives = spike_times_gt.size
    #     spike_times_gt = numpy.sort(spike_times_gt)
    #     if 0 < w:
    #         spike_times_gt = (1 + 2 * w) * [spike_times_gt]
    #         for k in xrange(-w, w+1):
    #             spike_times_gt[k] = spike_times_gt[k] + k
    #         spike_times_gt = numpy.hstack(spike_times_gt)
    #         spike_times_gt = numpy.unique(spike_times_gt)
        
    #     results = io.load_data(params, 'results')
    #     spike_times_sp = results['spiketimes']
    #     N_cluster = len(spike_times_sp.keys())
        
    #     scores = numpy.zeros(N_cluster)
        
    #     for i in xrange(0, N_cluster):
    #         key = "temp_{}".format(i)
    #         spike_times_clst = spike_times_sp[key]
            
    #         spike_times_clst = numpy.sort(spike_times_clst)
            
    #         false_positives = numpy.setdiff1d(spike_times_clst, spike_times_gt)
    #         n_false_positives = false_positives.size
    #         n_true_positives = spike_times_clst.size - n_false_positives
            
    #         true_positive_rate = float(n_true_positives) / float(n_positives)
            
    #         scores[i] = true_positive_rate
        
    #     indices = numpy.arange(0, N_cluster)
    #     bar_width = 1.0
    #     index = numpy.argmax(scores)
    #     xtick = str(index)
        
    #     fig = plt.figure()
    #     ax = fig.gca()
    #     ax.bar(indices, scores, bar_width)
    #     ax.set_xlim(0, N_cluster)
    #     ax.set_ylim(0.0, 1.0)
    #     ax.set_xlabel("Cluster")
    #     ax.set_ylabel("score")
    #     plt.xticks([index + bar_width / 2], [xtick])
    #     plt.savefig("/tmp/scores.png")
    #     plt.close(fig)
        
    # sys.exit(0)
    
    
    # Initialize the random seed.
    _ = numpy.random.seed(0)
    
    
    
    ###### JUXTACELLULAR SPIKE DETECTION #######################################
    
    # Detect the spikes times of the juxtacellular trace.
    if comm.rank == 0:
        extract_juxta_spikes(filename, params)
    comm.Barrier()
    
    # Retrieve the spike times of the juxtacellular trace.
    spike_times_juxta = io.load_data(params, 'juxta-triggers')
    
    ############################################################################
    
    
    # Retrieve PCA basis.
    basis_proj, basis_rec = io.load_data(params, 'basis')
    N_p = basis_proj.shape[1]
    
    # Select only the neighboring channels of the best channel.
    chan = params.getint('validating', 'nearest_elec')
    if chan == -1:
        # Set best channel as the channel with the highest change in amplitude.
        nodes, chans = get_neighbors(params, chan=None)
        #juxta_spikes      = load_chunk(params, spike_times_juxta, chans=None)
        juxta_spikes      = get_stas(params, spike_times_juxta, numpy.zeros(len(spike_times_juxta)), 0, chans, nodes=nodes, auto_align=False).T
        mean_juxta_spikes = numpy.mean(juxta_spikes, axis=2)
        max_juxta_spikes  = numpy.amax(mean_juxta_spikes, axis=0)
        min_juxta_spikes  = numpy.amin(mean_juxta_spikes, axis=0)
        dif_juxta_spikes  = max_juxta_spikes - min_juxta_spikes
        chan = numpy.argmax(dif_juxta_spikes)
        if comm.rank == 0:
            msg = ["Ground truth neuron is close to channel {}".format(chan)]
            io.print_and_log(msg, level='default', logger=params)
    else:
        pass            

    nodes, chans = get_neighbors(params, chan=chan)
    
    if make_plots not in ['None', '']:
        plot_filename = "beer-trigger-times.%s" %make_plots
        path = os.path.join(plot_path, plot_filename)
        plot.view_trigger_times(params, spike_times_juxta, juxta_spikes[:, chan, :], save=path)


    ##### TODO: clean temporary zone
    if test_method == 'full':
        
        # Compute the weights of the classes.
        mask_min = time_min_test <= spike_times_juxta
        mask_max = spike_times_juxta <= time_max_test
        mask = numpy.logical_and(mask_min, mask_max)
        n_class_0 = numpy.count_nonzero(mask)
        n_class_1 = time_max_test - time_min_test + 1 - n_class_0
        
        if comm.rank == 0:
            msg = [
                "n_class_0: {}".format(n_class_0),
                "n_class_1: {}".format(n_class_1),
            ]
            io.print_and_log(msg, level='debug', logger=params)
        
        # sys.exit(0)
        
        # _, _, class_weights = get_class_weights_bis(n_class_0, n_class_1, n=7)
        # if comm.rank == 0:
        #     print("")
        #     for item in class_weights:
        #         print(item)
        # 
        # n_class_0 = 1881
        # n_class_1 = 9600
        # _, _, class_weights = get_class_weights_bis(n_class_0, n_class_1, n=7)
        # if comm.rank == 0:
        #     print("")
        #     for item in class_weights:
        #         print(item)
    ##### end temporary zone
    
    

    ##### SAMPLES PROPORTIONS ##################################################
    
    alpha = alpha_gt + alpha_ngt + alpha_noi
    N_gt_max = int((alpha_gt / alpha) * float(N_max))
    N_ngt_max = int((alpha_ngt / alpha) * float(N_max))
    N_noi_max = int((alpha_noi / alpha) * float(N_max))
    
    ##### TODO: remove debug zone
    if comm.rank == 0:
        print("alpha_gt: {}".format(alpha_gt))
        print("alpha_ngt: {}".format(alpha_ngt))
        print("alpha_noi: {}".format(alpha_noi))
        print("alpha: {}".format(alpha))
        print("N_max: {}".format(N_max))
        print("N_gt_max: {}".format(N_gt_max))
        print("N_ngt_max: {}".format(N_ngt_max))
        print("N_noi_max: {}".format(N_noi_max))
        N_total = N_gt_max + N_ngt_max + N_noi_max
        print("N_total: {}".format(N_total))
    # sys.exit(0)
    ##### end debug zone
    
    
    
    ##### GROUND TRUTH CELL'S SAMPLES ##########################################
    
    if comm.rank == 0:
        io.print_and_log(["Collecting ground truth cell's samples..."], level='debug', logger=params)
    
    # Retrieve the spike times of the "ground truth cell".
    spike_times_gt = spike_times_juxta
    
    ##### TODO: remove debug zone
    if comm.rank == 0:
        N_gt = spike_times_gt.size
        print("N_gt: {}".format(N_gt))
    # sys.exit(0)
    ##### end debug zone
    
    ##### TODO: clean temporary zone
    # Filter out the end of the data (~30%).
    if test_method == 'full':
        spike_times_gt_test = spike_times_gt[time_min_test <= spike_times_gt]
        spike_times_gt = spike_times_gt[spike_times_gt <= time_max]
    elif test_method == 'downsampled':
        pass
    else:
        raise Exception("Unknown test method: {}".format(test_method))
    ##### end temporary zone
    
    idx = numpy.sort(numpy.random.permutation(numpy.arange(len(spike_times_gt)))[:N_gt_max])
    spike_times_gt = spike_times_gt[idx]
    spikes_gt = get_stas(params, spike_times_gt, numpy.zeros(len(idx)), chan, chans, nodes=nodes, auto_align=False).T
    
    #if comm.rank == 0:
    #    if make_plots not in ['None', '']:
            #plot_filename = "beer-trigger-times-gt.%s" %make_plots
            #path = os.path.join(plot_path, plot_filename)
            #plot.view_trigger_times(filename, spike_times_gt, color='green', save=path)
    #        if make_plots_snippets:
    #            directory = "beer-trigger-snippets-gt"
    #            path = os.path.join(plot_path, directory)
    #            plot.view_trigger_snippets(spikes_gt, chans, save=path)
    
    # Reshape data.
    N_t = spikes_gt.shape[0]
    N_e = spikes_gt.shape[1]
    N_gt = spikes_gt.shape[2]
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
                "# X_gt.shape: {}".format(X_gt.shape),
                "# y_gt.shape: {}".format(y_gt.shape),
            ]
            io.print_and_log(msg, level='default', logger=params)
        
        #if make_plots not in ['None', '']:
        #    plot_filename = "beer-dataset-gt.%s" %make_plots
        #    path = os.path.join(plot_path, plot_filename)
        #    plot.view_dataset(X_gt, color='green', title="Ground-truth dataset", save=path)
    
   
    
    ############################################################################

    ##### TODO: remove quarantine zone
    # print("N_max = {}".format(N_max))
    # print("N_gt = {}".format(N_gt))
    # print("N_max / N_gt = {}".format(N_max / N_gt))
    # print("alpha_ngt = {}".format(alpha_ngt))
    # print("alpha_noi = {}".format(alpha_noi))
    
    # # Compute the amount of 'ngt' and 'noi' to load.
    # c = (float(N_max) / float(N_gt) - 1.0) / float(alpha_ngt + alpha_noi)
    # if c < 1.0:
    #     # TODO: subsample.
    #     raise Exception("TODO: c = {}".format(c))
    # else:
    #     alpha_ngt = c * alpha_ngt
    #     alpha_noi = c * alpha_noi
    ##### end quarantine zone
    
    # Compute the forbidden spike times.
    max_time_shift = 0.25 # ms
    max_time_shift = int(float(sampling_rate) * max_time_shift * 1.0e-3)
    spike_times_fbd = (2 * max_time_shift + 1) * [None]
    for i, time_shift in enumerate(xrange(-max_time_shift, max_time_shift+1)):
        spike_times_fbd[i] = spike_times_gt + time_shift
    spike_times_fbd = numpy.concatenate(spike_times_fbd)
    spike_times_fbd = numpy.unique(spike_times_fbd)
    
    
    
    ##### NON GROUND TRUTH CELL'S SAMPLES ######################################
    
    if comm.rank == 0:
        io.print_and_log(["Non ground truth cells' samples..."], level='debug', logger=params)
    
    # Detect the spikes times of the "non ground truth cell".
    extract_extra_spikes(filename, params)
    
    # Retrieve the spike times of the "non ground truth cell".
    spike_times_ngt_tmp = io.load_data(params, 'extra-triggers')

    ##### TODO: clean temporary zone
    if test_method == 'full':
        # Filter out the end of the data (~30%).
        spike_times_ngt_tmp = [t[t <= time_max] for t in spike_times_ngt_tmp]
    elif test_method == 'downsampled':
        pass
    else:
        raise Exception("Unknown test method: {}".format(test_method))
    ##### end temporary zone
    
    # Filter the spike times of the "non ground truth cell".
    ## Restrict to spikes which happened in the vicinity.
    spike_times_ngt_tmp = [spike_times_ngt_tmp[chan] for chan in chans]
    spike_times_ngt_tmp = numpy.concatenate(spike_times_ngt_tmp)
    spike_times_ngt_tmp = numpy.unique(spike_times_ngt_tmp)
    ## Restrict to spikes which are far from ground truth spikes.
    spike_times_ngt_tmp = numpy.setdiff1d(spike_times_ngt_tmp, spike_times_fbd)
    
    idx = numpy.sort(numpy.random.permutation(numpy.arange(len(spike_times_ngt_tmp)))[:N_ngt_max])
    spike_times_ngt = spike_times_ngt_tmp[idx]
    spikes_ngt = get_stas(params, spike_times_ngt, numpy.zeros(len(idx)), chan, chans, nodes=nodes, auto_align=False).T

    #if comm.rank == 0:
    #    if make_plots not in ['None', '']:
        #    plot_filename = "beer-trigger-times-ngt.%s" %make_plots
        #    path = os.path.join(plot_path, plot_filename)
        #    plot.view_trigger_times(filename, spike_times_ngt, color='blue', save=path)
    #        if make_plots_snippets:
    #            directory = "beer-trigger-snippets-ngt"
    #            path = os.path.join(plot_path, directory)
    #            plot.view_trigger_snippets(spikes_ngt, chans, save=path)
    
    # Reshape data.
    N_t = spikes_ngt.shape[0]
    N_e = spikes_ngt.shape[1]
    N_ngt = spikes_ngt.shape[2]
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
                "# X_ngt.shape: {}".format(X_ngt.shape),
                "# y_ngt.shape: {}".format(y_ngt.shape),
            ]
            io.print_and_log(msg, level='default', logger=params)
        
        #if make_plots not in ['None', '']:
        #    plot_filename = "beer-dataset-ngt.%s" %make_plots
        #    path = os.path.join(plot_path, plot_filename)
        #    plot.view_dataset(X_ngt, color='blue', title="Non ground-truth dataset", save=path)
    
    
    
    ##### NOISE SAMPLES ########################################################
    
    if comm.rank == 0:
        io.print_and_log(["Collecting noise samples..."], level='debug', logger=params)
    
    # Extract the noise times.
    ## Draw times uniformly.
    size = spike_times_ngt_tmp.size
    spike_times_noi = numpy.random.random_integers(time_min, time_max, size)
    spike_times_noi = numpy.unique(spike_times_noi)
    ## Restrict to spikes which are far from ground truth spikes.
    spike_times_noi = numpy.setdiff1d(spike_times_noi, spike_times_fbd)
    ## Downsample to get the wanted number of spikes.
    
    idx = numpy.sort(numpy.random.permutation(numpy.arange(len(spike_times_noi)))[:N_noi_max])
    spikes_noi = get_stas(params, spike_times_noi[idx], numpy.zeros(len(idx)), chan, chans, nodes=nodes, auto_align=False).T

    #if comm.rank == 0:
    #    if make_plots not in ['None', '']:
    #        plot_filename = "beer-trigger-times.%s" %make_plots
    #        path = os.path.join(plot_path, plot_filename)
    #        plot.view_trigger_times(filename, [spike_times_gt, spike_times_ngt, spike_times_noi], color=['g', 'b', 'r'], save=path)
            #if make_plots_snippets:
            #    directory = "beer-trigger-snippets-noi"
            #    path = os.path.join(plot_path, directory)
            #    plot.view_trigger_snippets(spikes_noi, chans, save=path)
    
    # Reshape data.
    N_t = spikes_noi.shape[0]
    N_e = spikes_noi.shape[1]
    N_noi = spikes_noi.shape[2]
    spikes_noi = spikes_noi.reshape(N_t, N_e * N_noi)
    spikes_noi = spikes_noi.T
    # Compute the PCA coordinates of each "non-spike" sample.
    X_noi = numpy.dot(spikes_noi, basis_proj)
    X_noi = X_noi.T
    # Reshape data.
    X_noi = X_noi.reshape(N_p * N_e, N_noi)
    X_noi = X_noi.T
    
    # Define outputs (i.e. 1 for non ground truth samples).
    y_noi = numpy.ones((N_noi, 1))
    
    if comm.rank == 0:
        if verbose:
            msg = [
                "# X_noi.shape: {}".format(X_noi.shape),
                "# y_noi.shape: {}".format(y_noi.shape),
            ]
            io.print_and_log(msg, level='default', logger=params)
        
        #if make_plots not in ['None', '']:
        #    plot_filename = "beer-dataset-noi.%s" %make_plots
        #    path = os.path.join(plot_path, plot_filename)
        #    plot.view_dataset(X_noi, color='red', title="Noise dataset", save=path)
    
    
    
    # NORMALIZE DATASETS #######################################################
    
    if comm.rank == 0:
        io.print_and_log(["Normalizing datasets..."], level='debug', logger=params)
    
    
    X_raw = numpy.vstack((X_gt, X_ngt, X_noi))
    norm_scale = numpy.mean(numpy.linalg.norm(X_raw, axis=1))
    X_gt  /= norm_scale
    X_ngt /= norm_scale
    X_noi /= norm_scale
    
    
    
    ##### SAMPLES ##############################################################
    
    if comm.rank == 0:
        io.print_and_log(["Samples..."], level='debug', logger=params)
    
    ##### TODO: remove debug zone
    if comm.rank == 0:
        print("N_gt: {}".format(y_gt.size))
        print("N_ngt: {}".format(y_ngt.size))
        print("N_noi: {}".format(y_noi.size))
    # sys.exit(0)
    ##### end debug zone
    
    # Option to include the pairwise product of feature vector elements.
    pairwise = True
    
    # Create the datasets to train the neural network.
    ## Create the input dataset.
    N = X_gt.shape[1]
    
    X_raw = numpy.vstack((X_gt, X_ngt, X_noi))
    
    if pairwise:
        # With pairwise product of feature vector elements.
        M = N + N * (N + 1) // 2
        shape = (N_gt + N_ngt + N_noi, M)
    else:
        # Without pairwise product of feature vector elments.
        M = N
        shape = (N_gt + N_ngt + N_noi, M)
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
                    "# X.shape (with pairwise product of feature vector element): {}".format(X.shape),
                ]
                io.print_and_log(msg, level='default', logger=params)
    
    ## Create the output dataset.
    y_raw = numpy.vstack((y_gt, y_ngt, y_noi))
    y_raw = y_raw.ravel()
    y = y_raw
    
    if comm.rank == 0:
        if verbose:
            msg = [
                "# X_raw.shape: {}".format(X_raw.shape),
                "# y_raw.shape: {}".format(y_raw.shape),
                "# X.shape: {}".format(X.shape),
                "# y.shape: {}".format(y.shape),
            ]
            io.print_and_log(msg, level='default', logger=params)
    
    
    ##### DATASET PROBABILITIES ################################################
    
    # Original probabilities without match window.
    b_origin_ = float(spike_times_juxta.size) / float(time_max - time_min + 1)
    
    # Original probabilities with match window.
    thresh = int(float(sampling_rate) * 2.0 * 1.0e-3)
    tmp = numpy.zeros(time_max - time_min + 1, dtype='bool')
    indices = numpy.repeat(spike_times_juxta, 2 * thresh + 1)
    indices = numpy.reshape(indices, (spike_times_juxta.size, 2 * thresh + 1))
    indices = indices + numpy.arange(- thresh, thresh + 1)
    indices = numpy.reshape(indices, (-1,))
    indices = indices[time_min <= indices]
    indices = indices[indices <= time_max]
    indices = indices - time_min
    tmp[indices] = True
    b_origin = float(numpy.count_nonzero(tmp)) / float(time_max - time_min + 1)
    
    if comm.rank == 0:
        
        # print("")
        # print("spike_times_juxta.size: {}".format(spike_times_juxta.size))
        # print("spike_times_juxta[:10]: {}".format(spike_times_juxta[:10]))
        # print("time_min: {}".format(time_min))
        # print("time_max: {}".format(time_max))
        # print("thresh: {}".format(thresh))
        # print("tmp.size: {}".format(tmp.size))
        # print("tmp: {}".format(tmp))
        # print("numpy.count_nonzero(tmp): {}".format(numpy.count_nonzero(tmp)))
        
        print("")
        print("    b_origin_: {}".format(b_origin_))
        print("1 - b_origin_: {}".format(1.0 - b_origin_))
        print("ratio: 1:{}".format(int((1.0 - b_origin_) / b_origin_)))
        print("    b_origin: {}".format(b_origin))
        print("1 - b_origin: {}".format(1.0 - b_origin))
        print("ratio: 1:{}".format(int((1.0 - b_origin) / b_origin)))
        print("b_origin / b_origin_: {}".format(b_origin / b_origin_))
    
    
    # Undersampled probabilities without match window.
    b_under_ = float(spike_times_gt.size) / float(spike_times_gt.size + spike_times_ngt.size + spike_times_noi.size)
    
    # Training probabilities with math window.
    thresh = int(float(sampling_rate) * 2.0 * 1.0e-3)
    tmp_0 = numpy.zeros(time_max - time_min + 1, dtype='bool')
    tmp_1 = numpy.zeros(time_max - time_min + 1, dtype='bool')
    indices_0 = numpy.repeat(spike_times_gt, 2 * thresh + 1)
    indices_0 = numpy.reshape(indices_0, (spike_times_gt.size, 2 * thresh + 1))
    indices_0 = indices_0 + numpy.arange(- thresh, thresh + 1)
    indices_0 = numpy.reshape(indices_0, (-1,))
    indices_0 = indices_0[time_min <= indices_0]
    indices_0 = indices_0[indices_0 <= time_max]
    indices_0 = indices_0 - time_min
    tmp_0[indices_0] = True
    spike_times = numpy.concatenate((spike_times_gt, spike_times_ngt, spike_times_noi))
    indices_1 = numpy.repeat(spike_times, 2 * thresh + 1)
    indices_1 = numpy.reshape(indices_1, (spike_times.size, 2 * thresh + 1))
    indices_1 = indices_1 + numpy.arange(- thresh, thresh + 1)
    indices_1 = numpy.reshape(indices_1, (-1,))
    indices_1 = indices_1[time_min <= indices_1]
    indices_1 = indices_1[indices_1 <= time_max]
    indices_1 = indices_1 - time_min
    tmp_1[indices_1] = True
    b_under = float(numpy.count_nonzero(tmp_0)) / float(numpy.count_nonzero(tmp_1))
    
    alpha_under = (b_origin * (1.0 - b_under)) / (b_under * (1.0 - b_origin))
    
    if comm.rank == 0:
        
        # print("")
        # print("spike_times_gt.size: {}".format(spike_times_gt.size))
        # print("spike_times_gt[:10]: {}".format(spike_times_gt[:10]))
        # print("spike_times_ngt.size: {}".format(spike_times_ngt.size))
        # print("spike_times_ngt[:10]: {}".format(spike_times_ngt[:10]))
        # print("spike_times_noi.size: {}".format(spike_times_noi.size))
        # print("spike_times_noi[:10]: {}".format(spike_times_noi[:10]))
        # print("numpy.count_nonzero(tmp_0): {}".format(numpy.count_nonzero(tmp_0)))
        # print("numpy.count_nonzero(tmp_1): {}".format(numpy.count_nonzero(tmp_1)))
        
        print("")
        print("      b_under_: {}".format(b_under_))
        print("1.0 - b_under_: {}".format(1.0 - b_under_))
        print("ratio: 1:{}".format(int((1.0 - b_under_) / b_under_)))
        print("      b_under: {}".format(b_under))
        print("1.0 - b_under: {}".format(1.0 - b_under))
        print("ratio: 1:{}".format(int((1.0 - b_under) / b_under)))
        print("alpha_under: {}".format(alpha_under))
    
    
    
    ##### SANITY PLOT ##########################################################
    
    if comm.rank == 0:
        
        #io.print_and_log(["Sanity plot..."], level='info', logger=params)
        
        
        if make_plots not in ['None', '']:
            plot_filename = "beer-datasets.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            xs = [X_gt, X_ngt, X_noi]
            ys = [y_gt, y_ngt, y_noi]
            colors = ['r', 'b', 'k']
            labels = ["GT", "Non GT", "Noise"]
            plot.view_datasets(params, xs, ys, [spike_times_gt, spike_times_ngt, spike_times_noi], colors=colors, labels=labels, save=path)
    
    
    
    ##### INITIAL PARAMETER ####################################################
    
    if comm.rank == 0:
        io.print_and_log(["Initializing parameters for the non-linear classifier..."], level='default', logger=params)
    
    
    method = 'covariance'
    #method = 'geometric'
    
    if method is 'covariance':
        
        mu = numpy.mean(X_gt.T, axis=1)
        sigma = numpy.cov(X_gt.T)
        k = 1.0
        
        sigma_inv = numpy.linalg.inv(sigma)
        
        A_init = sigma_inv
        b_init = - 2.0 * numpy.dot(mu, sigma_inv)
        c_init = numpy.dot(mu, numpy.dot(sigma_inv, mu)) - k * k
        
        coefs_init = ellipsoid_matrix_to_coefs(A_init, b_init, c_init)
        
    elif method is 'geometric':
        
        t = numpy.mean(X_gt.T, axis=1)
        c = numpy.cov(X_gt.T)
        
        
        if make_plots not in ['None', '']:
            fig = plt.figure()
            ax = fig.gca()
            cax = ax.imshow(c, interpolation='nearest')
            ax.set_title("Covariance matrix of ground truth")
            fig.colorbar(cax)
            plot_filename = "covariance-matrix-gt.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            fig.savefig(path)
            plt.close(fig)
        
        
        s, O = numpy.linalg.eigh(c)
        coefs_init = ellipsoid_standard_to_general(t, s, O)
        
    else:
        
        raise(Exception)
    
    if comm.rank == 0:
        if verbose:
            msg = [
                "# coefs_init: {}".format(coefs_init),
            ]
            io.print_and_log(msg, level='default', logger=params)
    
    
    # Compute false positive rate and true positive rate for various cutoffs.
    num = 300
    
    Mhlnb_gt = squared_Mahalanobis_distance(A_init, mu, X_gt)
    Mhlnb_ngt = squared_Mahalanobis_distance(A_init, mu, X_ngt)
    Mhlnb_noi = squared_Mahalanobis_distance(A_init, mu, X_noi)
    
    Mhlnb = numpy.concatenate((Mhlnb_gt, Mhlnb_ngt, Mhlnb_noi))
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
        fp = float(numpy.count_nonzero(Mhlnb_ngt < cutoff)
                   + numpy.count_nonzero(Mhlnb_noi < cutoff))
        n = float(Mhlnb_ngt.size + Mhlnb_noi.size)
        fprs[i] = fp / n
        tp = float(numpy.count_nonzero(Mhlnb_gt < cutoff))
        p = float(Mhlnb_gt.size)
        tprs[i] = tp / p
    
    if comm.rank == 0:
        if verbose:
            # msg = [
            #     "# cutoffs: {}".format(cutoffs),
            #     "# fprs: {}".format(fprs),
            #     "# tprs: {}".format(tprs),
            # ]
            # io.print_and_log(msg, level='default', logger=params)
            pass
    
    # Compute mean acccuracy for various cutoffs.
    accs = numpy.zeros(num)
    for (i, index) in enumerate(indices):
        cutoff = Mhlnb[index]
        tp = float(numpy.count_nonzero(Mhlnb_gt <= cutoff))
        p = float(Mhlnb_gt.size)
        tn = float(numpy.count_nonzero(cutoff < Mhlnb_ngt)
                   + numpy.count_nonzero(cutoff < Mhlnb_noi))
        n = float(Mhlnb_ngt.size + Mhlnb_noi.size)
        accs[i] = (tp + tn) / (p + n)
    
    # Find the optimal cutoff.
    i_opt = numpy.argmax(accs)
    cutoff_opt_acc = cutoffs[i_opt]
    
    if comm.rank == 0:
        if verbose:
            msg = [
                "# cutoff_opt_acc: {}".format(cutoff_opt_acc),
                "# acc_opt: {}".format(accs[i_opt]),
            ]
            io.print_and_log(msg, level='default', logger=params)
        
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
        tn = float(numpy.count_nonzero(cutoff < Mhlnb_ngt)
                   + numpy.count_nonzero(cutoff < Mhlnb_noi))
        n = float(Mhlnb_ngt.size + Mhlnb_noi.size)
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
                "# cutoff_opt_norm_acc: {}".format(cutoff_opt_norm_acc),
                "# norm_acc_opt: {}".format(norm_accs[i_opt]),
            ]
            io.print_and_log(msg, level='default', logger=params)
        
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
    fp = float(numpy.count_nonzero(Mhlnb_ngt < cutoff)
               + numpy.count_nonzero(Mhlnb_noi < cutoff))
    n = float(Mhlnb_ngt.size + Mhlnb_noi.size)
    fpr = fp / n
    tp = float(numpy.count_nonzero(Mhlnb_gt < cutoff))
    p = float(Mhlnb_gt.size)
    tpr = tp / p
    
    if comm.rank == 0:
        if verbose:
            msg = [
                "# cutoff: {}".format(cutoff),
                "# fpr: {}".format(fpr),
                "# tpr: {}".format(tpr),
            ]
            io.print_and_log(msg, level='default', logger=params)
        
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
        io.print_and_log(["Sanity plot (classifier projection)..."],
                         level='debug', logger=params)
        
        
        if make_plots not in ['None', '']:
            # Plot initial classifier (ellipsoid).
            #title = "Initial classifier (ellipsoid)"
            #plot_filename = "beer-classifier-projection-init.%s" %make_plots
            #path = os.path.join(plot_path, plot_filename)

            data_class_1 = [X_gt, X_ngt, X_noi], [y_gt, y_ngt, y_noi], A_init, b_init, c_init

    
    
    
    # MAHALANOBIS DISTRIBUTIONS ################################################
    
    if comm.rank == 0:
        
        io.print_and_log(["Intialising Mahalanobis distributions..."],
                         level='debug', logger=params)
        
        
        # Compute mahalanobis distributions.
        mu = numpy.mean(X_gt, axis=0)
        Mhlnb_gt = squared_Mahalanobis_distance(A_init, mu, X_gt)
        Mhlnb_ngt = squared_Mahalanobis_distance(A_init, mu, X_ngt)
        Mhlnb_noi = squared_Mahalanobis_distance(A_init, mu, X_noi)
        
        data_mal1 = Mhlnb_gt, Mhlnb_ngt, Mhlnb_noi
    
    
    
    ##### LEARNING #############################################################
    
    # mode = 'decision'
    mode = 'prediction'
    
    model = 'sgd'
    
    ##### TODO: clean temporary zone
    def split_train_test(X, y, test_size=0.5, seed=0):
        size = X.shape[0]
        indices = numpy.random.permutation(size)
        thresh = int((1.0 - test_size) * float(size))
        indices_train = indices[:thresh]
        indices_test = indices[thresh:]
        return indices_train, indices_test
    ##### end temporary zone
    
    # Preprocess dataset.
    if not skip_demo:
        ##### TODO: clean temporary zone
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        indices_train, indices_test = split_train_test(X, y, test_size=0.3)
        X_train = X[indices_train, :]
        X_test = X[indices_test, :]
        y_train = y[indices_train]
        y_test = y[indices_test]
        ##### end temporary zone
    else:
        X_train = X
        X_test = None
        y_train = y
        y_test = None
    
    ##### TODO: remove temporary zone
    if test_method == 'full':
        X_train, X_test, y_train, y_test = X, None, y, None
    elif test_method == 'downsampled':
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        indices_train, indices_test = split_train_test(X, y, test_size=0.3)
        X_train = X[indices_train, :]
        X_test = X[indices_test, :]
        y_train = y[indices_train]
        y_test = y[indices_test]
    else:
        raise Exception("Unknown test mode: {}".format(test_method))
    ##### end temporary zone
    
    
    ##### TODO: remove temporary zone
    # TODO: compute the probabilities of the training set and the test set.
    
    spike_times = numpy.concatenate((spike_times_gt, spike_times_ngt, spike_times_noi))
    
    spike_times_gt_train = spike_times[indices_train[indices_train < spike_times_gt.size]]
    spike_times_gt_test = spike_times[indices_test[indices_test < spike_times_gt.size]]
    
    spike_times_train = spike_times[indices_train]
    spike_times_test = spike_times[indices_test]
    
    # Training probabilities without match window.
    b_train_ = float(spike_times_gt_train.size) / float(spike_times_train.size)
    
    # Training probabilities with math window.
    thresh = int(float(sampling_rate) * 2.0 * 1.0e-3)
    tmp_0 = numpy.zeros(time_max - time_min + 1, dtype='bool')
    indices_0 = numpy.repeat(spike_times_gt_train, 2 * thresh + 1)
    indices_0 = numpy.reshape(indices_0, (spike_times_gt_train.size, 2 * thresh + 1))
    indices_0 = indices_0 + numpy.arange(- thresh, thresh + 1)
    indices_0 = numpy.reshape(indices_0, (-1,))
    indices_0 = indices_0[time_min <= indices_0]
    indices_0 = indices_0[indices_0 <= time_max]
    indices_0 = indices_0 - time_min
    tmp_0[indices_0] = True
    tmp_1 = numpy.zeros(time_max - time_min + 1, dtype='bool')
    indices_1 = numpy.repeat(spike_times_train, 2 * thresh + 1)
    indices_1 = numpy.reshape(indices_1, (spike_times_train.size, 2 * thresh + 1))
    indices_1 = indices_1 + numpy.arange(- thresh, thresh + 1)
    indices_1 = numpy.reshape(indices_1, (-1,))
    indices_1 = indices_1[time_min <= indices_1]
    indices_1 = indices_1[indices_1 <= time_max]
    indices_1 = indices_1 - time_min
    tmp_1[indices_1] = True
    b_train = float(numpy.count_nonzero(tmp_0)) / float(numpy.count_nonzero(tmp_1))
    
    # Testing probabilities without match window.
    b_test_ = float(spike_times_gt_test.size) / float(spike_times_test.size)
    
    # Testing probabilities with match window.
    thresh = int(float((sampling_rate) * 2.0 * 1.0e-3))
    tmp_0 = numpy.zeros(time_max - time_min + 1, dtype='bool')
    indices_0 = numpy.repeat(spike_times_gt_test, 2 * thresh + 1)
    indices_0 = numpy.reshape(indices_0, (spike_times_gt_test.size, 2 * thresh + 1))
    indices_0 = indices_0 + numpy.arange(- thresh, thresh + 1)
    indices_0 = numpy.reshape(indices_0, (-1,))
    indices_0 = indices_0[time_min <= indices_0]
    indices_0 = indices_0[indices_0 <= time_max]
    indices_0 = indices_0 - time_min
    tmp_0[indices_0] = True
    tmp_1 = numpy.zeros(time_max - time_min + 1, dtype='bool')
    indices_1 = numpy.repeat(spike_times_test, 2 * thresh + 1)
    indices_1 = numpy.reshape(indices_1, (spike_times_test.size, 2 * thresh + 1))
    indices_1 = indices_1 + numpy.arange(- thresh, thresh + 1)
    indices_1 = numpy.reshape(indices_1, (-1,))
    indices_1 = indices_1[time_min <= indices_1]
    indices_1 = indices_1[indices_1 <= time_max]
    indices_1 = indices_1 - time_min
    tmp_1[indices_1] = True
    b_test = float(numpy.count_nonzero(tmp_0)) / float(numpy.count_nonzero(tmp_1))
    
    alpha_train = (b_origin * (1.0 - b_train)) / (b_train * (1.0 - b_origin))
    alpha_test = (b_origin * (1.0 - b_test)) / (b_test * (1.0 - b_origin))
    
    if comm.rank == 0:
        
        # print("")
        # print("spike_times_gt.size: {}".format(spike_times_gt.size))
        # print("spike_times_gt[:10]: {}".format(spike_times_gt[:10]))
        # print("spike_times_ngt.size: {}".format(spike_times_ngt.size))
        # print("spike_times_ngt[:10]: {}".format(spike_times_ngt[:10]))
        # print("spike_times_noi.size: {}".format(spike_times_noi.size))
        # print("spike_times_noi[:10]: {}".format(spike_times_noi[:10]))
        # print("numpy.count_nonzero(tmp_0): {}".format(numpy.count_nonzero(tmp_0)))
        # print("numpy.count_nonzero(tmp_1): {}".format(numpy.count_nonzero(tmp_1)))
        
        print("")
        print("      b_train_: {}".format(b_train_))
        print("1.0 - b_train_: {}".format(1.0 - b_train_))
        print("ratio: 1:{}".format(int((1.0 - b_train_) / b_train_)))
        print("      b_train: {}".format(b_train))
        print("1.0 - b_train: {}".format(1.0 - b_train))
        print("ratio: 1:{}".format(int((1.0 - b_train) / b_train)))
        print("alpha_train: {}".format(alpha_train))
        
        print("")
        print("      b_test_: {}".format(b_test_))
        print("1.0 - b_test_: {}".format(1.0 - b_test_))
        print("ratio: 1:{}".format(int((1.0 - b_test_) / b_test_)))
        print("      b_test: {}".format(b_test))
        print("1.0 - b_test: {}".format(1.0 - b_test))
        print("ratio: 1:{}".format(int((1.0 - b_test) / b_test)))
        print("alpha_test: {}".format(alpha_test))
    
    
    # sys.exit(0)
    
    ##### end temporary zone
    
    
    if not skip_demo:
        
        if comm.rank == 0:
            io.print_and_log(["Start learning..."], level='debug', logger=params)
        
        
        if comm.rank == 0:
            if verbose:
                msg = [
                    "# X_train.shape: {}".format(X_train.shape),
                    "# X_test.shape: {}".format(X_test.shape),
                ]
                io.print_and_log(msg, level='default', logger=params)
        
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
            _, _, class_weights = get_class_weights(y_gt, y_ngt, y_noi, n=1)
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
        
        #if comm.rank == 0:
            #if make_plots not in ['None', '']:
            #    # Plot prediction.
            #    title = "Initial prediction (random)"
            #    plot_filename = "beer-prediction-init-random.%s" %make_plots
            #    path = os.path.join(plot_path, plot_filename)
            #    plot.view_classification(clf, X, X_raw, y_raw, mode='predict',
            #                             title=title, save=path)
            #    # Plot decision function.
            #    title = "Initial decision function (random)"
            #    plot_filename = "beer-decision-function-init-random.%s" %make_plots
            #    path = os.path.join(plot_path, plot_filename)
            #    plot.view_classification(clf, X, X_raw, y_raw, mode='decision_function',
            #                             title=title, save=path)
        
        if comm.rank == 0:
            if verbose:
                y_pred = clf.predict(X_test)
                score = accuracy_score(y_test, y_pred, class_weights=class_weights[0])
                msg = [
                    # # Print the current loss.
                    # "# clf.loss_: {}".format(clf.loss_),
                    # # Print the loss curve.
                    # "# clf.loss_curve_: {}".format(clf.loss_curve_),
                    # # Print the number of iterations the algorithm has ran.
                    # "# clf.n_iter_: {}".format(clf.n_iter_),
                    # Print the score on the test set.
                    "# accuracy_score(X_test, y_test): {} ({})".format(score, 1.0 - score),
                ]
                io.print_and_log(msg, level='default', logger=params)
        
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
                    # "# clf.loss_: {}".format(clf.loss_),
                    # # Print the loss curve.
                    # "# clf.loss_curve_: {}".format(clf.loss_curve_),
                    # # Print the number of iterations the algorithm has ran.
                    # "# clf.n_iter_: {}".format(clf.n_iter_),
                    # Print the score on the test set.
                    "# accuracy_score(X_test, y_test): {} ({})".format(score, 1.0 - score),
                ]
                io.print_and_log(msg, level='default', logger=params)
        
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
                    # "# clf.loss_: {}".format(clf.loss_),
                    # # Print the loss curve.
                    # "# clf.loss_curve_: {}".format(clf.loss_curve_),
                    # # Print the number of iterations the algorithm has ran.
                    # "# clf.n_iter_: {}".format(clf.n_iter_),
                    # Print the score on the test set.
                    "# accuracy_score(X_test, y_test): {} ({})".format(score, 1.0 - score),
                ]
                io.print_and_log(msg, level='default', logger=params)
        
        
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
    
    if not skip_demo:
        
        if comm.rank == 0:
            
            io.print_and_log(["Sanity plot (classifier projection)..."],
                             level='debug', logger=params)
            
            
            if make_plots not in ['None', '']:
                # Plot final classifier.
                title = "Final classifier"
                plot_filename = "beer-classifier-projection.%s" %make_plots

                data_class_2 = [X_gt, X_ngt, X_noi], [y_gt, y_ngt, y_noi], A, b, c

                path = os.path.join(plot_path, plot_filename)
                plot.view_classifier(params, data_class_1, data_class_2, save=path, verbose=verbose)
    
    
    
    # MAHALANOBIS DISTRIBUTIONS ################################################
    
    if not skip_demo:
        
        if comm.rank == 0:
            
            io.print_and_log(["Computing final Mahalanobis distributions..."],
                             level='debug', logger=params)
            
            
            # Compute the Mahalanobis distributions.
            mu = numpy.mean(X_gt, axis=0)
            Mhlnb_gt = squared_Mahalanobis_distance(A, mu, X_gt)
            Mhlnb_ngt = squared_Mahalanobis_distance(A, mu, X_ngt)
            Mhlnb_noi = squared_Mahalanobis_distance(A, mu, X_noi)

            data_mal2 = Mhlnb_gt, Mhlnb_ngt, Mhlnb_noi
            
            if verbose:
                msg = [
                    "# Mhlnb_gt: {}".format(Mhlnb_gt),
                ]
                io.print_and_log(msg, level='default', logger=params)
            
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
        io.print_and_log(["Estimating the ROC curve..."], level='default', logger=params)
    
    
    ##### TODO: clean temporary zone
    _, _, class_weights = get_class_weights(y_gt, y_ngt, y_noi, n=roc_sampling)
    # _, _, class_weights = get_class_weights_bis(n_class_0, n_class_1, n=roc_sampling)
    ##### end temporary zone
    
    # Distribute weights over the CPUs.
    loc_indices = numpy.arange(comm.rank, roc_sampling, comm.size)
    loc_class_weights = [class_weights[loc_index] for loc_index in loc_indices]
    loc_nb_class_weights = len(loc_class_weights)
    
    # Preallocation to collect results.
    confusion_matrices = loc_nb_class_weights * [None]
    y_decfs = loc_nb_class_weights * [None]
    y_preds = loc_nb_class_weights * [None]
    if test_method == 'full':
        time_preds = loc_nb_class_weights * [None]
    
    if comm.rank == 0:
        pbar = get_progressbar(loc_nb_class_weights)
    
    if model == 'sgd':
        for (count, class_weight) in enumerate(loc_class_weights):
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
            
            if test_method == 'full':
                
                time_chunk_size = 5000
                nb_time_chunks = (time_max_test - time_min_test + 1) // time_chunk_size
                if 0 < (time_max_test - time_min_test + 1) % time_chunk_size:
                    nb_time_chunks = nb_time_chunks + 1
                y_pred = nb_time_chunks * [None]
                y_decf = nb_time_chunks * [None]
                time_pred = nb_time_chunks * [None]
                for time_chunk in xrange(0, nb_time_chunks):
                    
                    #if comm.rank == 0:
                    #    print("##### New temporal chunk")
                    #    dt = datetime.now()
                    
                    ##### TODO: remove debug zone
                    # if comm.rank == 0:
                    #     print("{} / {}".format(time_chunk, nb_time_chunks))
                    ##### end debug zone
                    time_start = time_min_test + time_chunk * time_chunk_size
                    time_end = time_min_test + (time_chunk + 1) * time_chunk_size
                    time_end = min(time_end, time_max_test + 1)
                    spike_times_ = numpy.arange(time_start, time_end)
                    
                    #if comm.rank == 0:
                    #    dt_ = datetime.now()
                    #    print("Misc: {}s".format((dt_ - dt).total_seconds()))
                    #    dt = datetime.now()
                    
                    # Load the snippets
                    #spikes_ = load_chunk(params, spike_times_, chans=chans)
                    spikes_ = get_stas(params, spike_times_, numpy.zeros(len(spike_times_)), chan, chans, nodes=nodes, auto_align=False).T
                    
                    # Reshape data.
                    N_t = spikes_.shape[0]
                    N_e = spikes_.shape[1]
                    N_ = spikes_.shape[2]
                    spikes_ = spikes_.reshape(N_t, N_e * N_)
                    spikes_ = spikes_.T
                    # Compute the PCA coordinates.
                    X_ = numpy.dot(spikes_, basis_proj)
                    X_ = X_.T
                    # Reshape data.
                    X_ = X_.reshape(N_p * N_e, N_)
                    X_ = X_.T
                    
                    # if comm.rank == 0:
                    #    dt_ = datetime.now()
                    #    print("Load snippets: {}s".format((dt_ - dt).total_seconds()))
                    #    dt = datetime.now()
                    
                    # Normalize data.
                    X_ /= norm_scale
                    # Add quadratic features.
                    X_ = with_quadratic_feature(X_, pairwise=True)
                    
                    #if comm.rank == 0:
                    #    dt_ = datetime.now()
                    #    print("Add quadratic features: {}s".format((dt_ - dt).total_seconds()))
                    #    dt = datetime.now()
                    
                    time_pred[time_chunk] = spike_times_
                    # Compute the predictions.
                    y_pred[time_chunk] = wclf.predict(X_)
                    y_decf[time_chunk] = wclf.decision_function(X_)
                    
                    #if comm.rank == 0:
                    #    dt_ = datetime.now()
                    #    print("Classifier predictions: {}s".format((dt_ - dt).total_seconds()))
                    #    dt = datetime.now()
                    
                time_pred = numpy.concatenate(time_pred)
                y_pred = numpy.concatenate(y_pred)
                y_decf = numpy.concatenate(y_decf)
                
                ##### TODO: remove temporary zone
                # mask = (y_pred == 0)
                # time_pred = time_pred[mask]
                # print("CPU {}: {} {}".format(comm.rank, time_pred.shape, time_pred))
                # filename = "{}.decf-{}-{}.npy".format(file_out_suff, comm.rank, count)
                # numpy.save(filename, y_decf)
                ##### end temporary zone
                
                # TODO: filter y_pred with a time window (i.e. at least ? time slot between detections).
                
                # Retrieve the ground truth labeling.
                # TODO: find suitable dtype.
                y_test = numpy.ones(time_max_test - time_min_test + 1)
                indices = spike_times_gt_test - time_min_test
                y_test[indices] = 0
                
                # TODO: consider matches in a time window instead of exact matches.
                # Compute true positives, false negatives, true negatives and false
                # positives.
                p = (y_test == 0.0)
                tp = float(numpy.count_nonzero(y_pred[p] == y_test[p]))
                fn = float(numpy.count_nonzero(y_pred[p] != y_test[p]))
                n = (y_test == 1.0)
                tn = float(numpy.count_nonzero(y_pred[n] == y_test[n]))
                fp = float(numpy.count_nonzero(y_pred[n] != y_test[n]))
                # Construct and save the confusion matrix
                confusion_matrix = numpy.array([[tp, fn], [fp, tn]])
                ##### TODO: remove test zone
                
                print("")
                print(confusion_matrix)
                
                # TODO: take the base rate change into account.
                pp = (tp + fn) / (tp + fn + fp + tn)
                pn = (fp + tn) / (tp + fn + fp + tn)
                
                print("pp: {}".format(pp))
                print("pn: {}".format(pn))
                print("pp + pn: {}".format(pp + pn))
                
                print("b_origin: {}".format(b_origin))
                print("b_test: {}".format(b_test))
                
                pp_ = b_origin * (pp * (1.0 - b_test)) / (b_origin * pp + b_test * (1.0 - pp - b_origin))
                pn_ = (1.0 - b_origin) * pn * b_test / (b_origin * (1.0 - pn - b_test) + pn * b_test)
                
                print("pp_: {}".format(pp_))
                print("pn_: {}".format(pn_))
                print("pp_ + pn_: {}".format(pp_ + pn_))
                
                # TODO: uncomment to take the base rate change into account.
                # confusion_matrix[0, 0] = pp_ * (time_max - time_min) * tp / (tp + fn)
                # confusion_matrix[0, 1] = pp_ * (time_max - time_min) * fn / (tp + fn)
                # confusion_matrix[1, 0] = pn_ * (time_max - time_min) * fp / (fp + tn)
                # confusion_matrix[1, 1] = pn_ * (time_max - time_min) * tn / (fp + tn)
                
                print(confusion_matrix)
                
                ##### end test zone
                confusion_matrices[count] = confusion_matrix
                
                ##### TODO: remove temporary zone
                time_preds[count] = time_pred
                y_preds[count] = y_pred
                y_decfs[count] = y_decf
                ##### end temporary zone
                
                ##### end depreciated zone
                
            elif test_method == 'downsampled':
                
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
                ##### TODO: remove test zone
                
                print("")
                print(confusion_matrix)
                
                # TODO: take the base rate change into account.
                pp = (tp + fn) / (tp + fn + fp + tn)
                pn = (fp + tn) / (tp + fn + fp + tn)
                
                print("pp: {}".format(pp))
                print("pn: {}".format(pn))
                print("pp + pn: {}".format(pp + pn))
                
                print("b_origin: {}".format(b_origin))
                print("b_test: {}".format(b_test))
                
                pp_ = b_origin * (pp * (1.0 - b_test)) / (b_origin * pp + b_test * (1.0 - pp - b_origin))
                pn_ = (1.0 - b_origin) * pn * b_test / (b_origin * (1.0 - pn - b_test) + pn * b_test)
                
                print("pp_: {}".format(pp_))
                print("pn_: {}".format(pn_))
                print("pp_ + pn_: {}".format(pp_ + pn_))
                
                # TODO: uncomment to take the base rate change into account.
                # confusion_matrix[0, 0] = pp_ * (time_max - time_min) * tp / (tp + fn)
                # confusion_matrix[0, 1] = pp_ * (time_max - time_min) * fn / (tp + fn)
                # confusion_matrix[1, 0] = pn_ * (time_max - time_min) * fp / (fp + tn)
                # confusion_matrix[1, 1] = pn_ * (time_max - time_min) * tn / (fp + tn)
                
                print(confusion_matrix)
                
                ##### end test zone
                # Save results.
                y_preds[count] = y_pred
                y_decfs[count] = y_decf
                confusion_matrices[count] = confusion_matrix
                
            else:
                
                raise Exception("Unknown test method: {}".format(test_method))
            
            if comm.rank == 0:
                pbar.update(count)
            
    else:
        raise Exception("Unsupported classifier: model={}".format(model))
    
    if comm.rank == 0:
        pbar.finish()
    
    comm.Barrier()
    
    
    ##### TODO: remove temporary zone
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
        
        ##### TODO: remove depreciated zone.
        
        # # Reorder things properly.
        # time_preds = roc_sampling * [None]
        # y_preds = roc_sampling * [None]
        # y_decfs = roc_sampling * [None]
        # for (loc_indices, loc_time_preds_tmp) in zip(indices, time_preds_tmp):
        #     for (loc_index, loc_time_pred) in zip(loc_indices, loc_time_preds_tmp):
        #         time_preds[loc_index] = loc_time_pred
        # for (loc_indices, loc_y_preds_tmp) in zip(indices, y_preds_tmp):
        #     for (loc_index, loc_y_pred) in zip(loc_indices, loc_y_preds_tmp):
        #         y_preds[loc_index] = loc_y_pred
        # for (loc_indices, loc_y_decfs_tmp) in zip(indices, y_decfs_tmp):
        #     for (loc_index, loc_y_decf) in zip(loc_indices, loc_y_decfs_tmp):
        #         y_decfs[loc_index] = loc_y_decf
        # # Save things.
        # beer_filename = "{}.beer.hdf5".format(file_out_suff)
        # beer_file = h5py.File(beer_filename, 'a', libver='latest')
        # group_name = "beer_spiketimes"
        # if group_name in beer_file.keys():
        #     del beer_file[group_name]
        # beer_file.create_group(group_name)
        # for loc_index, time_pred, y_pred, y_decf in zip(loc_indices, time_preds, y_preds, y_decfs):
        #     filename = "{}/time_pred_{}".format(group_name, loc_index)
        #     print(filename)
        #     beer_file.create_dataset(filename, data=time_pred)
        #     filename = "{}/y_pred_{}".format(group_name, loc_index)
        #     print(filename)
        #     beer_file.create_dataset(filename, data=y_pred)
        #     filename = "{}/y_decf_{}".format(group_name, loc_index)
        #     print(filename)
        #     beer_file.create_dataset(filename, data=y_decfs)
        #     filename = "{}/temp_{}".format(group_name, loc_index)
        #     print(filename)
        #     mask = (y_pred == 1.0)
        #     temp = time_pred[mask]
        #     beer_file.create_dataset(filename, data=temp)
        # beer_file.close()
        
        ##### end depreciated zone.
    
    ##### end temorary zone
    
    
    # Gather results on the root CPU.
    indices = comm.gather(loc_indices, root=0)
    confusion_matrices_tmp = comm.gather(confusion_matrices, root=0)
    
    if comm.rank == 0:
        
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
        
        if verbose:
            msg = [
                "# class_weights: {}".format(class_weights),
                "# false positive rates: {}".format(fprs),
                "# true positive rates: {}".format(tprs),
            ]
            io.print_and_log(msg, level='default', logger=params)
        
        if make_plots not in ['None', '']:
            # Plot the ROC curve of the BEER estimate.
            title = "ROC curve of the BEER estimate"
            plot_filename = "beer-roc-curve.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            # plot.view_roc_curve(fprs, tprs, None, None, title=title, save=path,
            #                     xlim=[0.0, 0.25], ylim=[0.75, 1.0])
            # plot.view_roc_curve(fprs, tprs, None, None, title=title, save=path,
            #                     xlim=[0.0, 0.5], ylim=[0.5, 1.0])
            error = plot.view_roc_curve(params, fprs, tprs, None, None, save=path)
            filename = "{}.beer.hdf5".format(file_out_suff)
            beer_file = h5py.File(filename, 'r+', libver='latest')
            if 'circus_error' in beer_file.keys():
                beer_file.pop('circus_error')
            beer_file.create_dataset('circus_error', data=numpy.array(error, dtype=numpy.float32))
            beer_file.close()  
        
    
    ############################################################################
    
    if comm.rank == 0:
        io.print_and_log(["Validation done."], level='debug', logger=params)
    
    return
