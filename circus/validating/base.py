import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
# TODO: remove following line (i.e. remove warning).
# from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

import h5py

from ..shared.utils import *
from ..shared import plot
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
    nb_repeats = params.getint('clustering', 'nb_repeats')
    max_iter = params.getint('validating', 'max_iter')
    learning_rate_init = params.getfloat('validating', 'learning_rate')
    verbose = params.getboolean('validating', 'verbose')
    make_plots = params.get('validating', 'make_plots')
    roc_sampling = params.getint('validating', 'roc_sampling')
    plot_path = os.path.join(params.get('data', 'data_file_noext'), 'plots')
    test_size = params.getfloat('validating', 'test_size')
    file_out_suff = params.get('data', 'file_out_suff')
    
    # TODO: remove following lines.
    make_plots_snippets = False
    # N_max = 1000000
    N_max = 12000
    alpha_gt = 1.0
    alpha_ngt = 2.0
    alpha_noi = 2.0
    
    # Cut data into two halves.
    train_size = 0.9
    data_block = numpy.memmap(data_file, offset=data_offset, dtype=data_dtype, mode='r')
    N = len(data_block)
    data_len = N / N_total
    time_min = template_shift
    time_max = int(train_size * float(data_len - 1)) - template_shift
    time_min_test = int(train_size * float(data_len - 1)) + template_shift
    time_max_test = (data_len - 1) - template_shift
    
    
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
    
    ############################################################################
    
    
    
    # Initialize the random seed.
    _ = numpy.random.seed(0)
    
    # Retrieve PCA basis.
    basis_proj, basis_rec = io.load_data(params, 'basis')
    N_p = basis_proj.shape[1]
    
    # Select only the neighboring channels of the best channel.
    chan = params.getint('validating', 'val_chan')
    if chan == -1:
        # Automatic selection of the validation channel.
        # TODO: select the channel with the highest changes in amplitudes
        #       instead of an arbitrary selection.
        # TODO: remove default implementation which select a random channel.
        chan = numpy.random.randint(0, N_e)
    chans = get_neighbors(params, chan=chan)
    chan_index = numpy.argwhere(chans == chan)[0]
    
    

    ##### SAMPLES PROPORTIONS ##################################################
    
    alpha = alpha_gt + alpha_ngt + alpha_noi
    N_gt_max = int((alpha_gt / alpha) * float(N_max))
    N_ngt_max = int((alpha_ngt / alpha) * float(N_max))
    N_noi_max = int((alpha_noi / alpha) * float(N_max))
    
    
    
    ##### GROUND TRUTH CELL'S SAMPLES ##########################################
    
    if comm.rank == 0:
        io.print_and_log(["Collecting ground truth cell's samples..."], level='default', logger=params)
    
    # Detect the spikes times of the "ground truth cell".
    if comm.rank == 0:
        extract_juxta_spikes(filename, params)
    comm.Barrier()
    
    # Retrieve the spike times of the "ground truth cell".
    spike_times_gt = io.load_data(params, 'juxta-triggers')

    ##### TODO: clean temporary zone
    # Filter out the end of the data (~30%).
    spike_times_gt_test = spike_times_gt[time_min_test <= spike_times_gt]
    spike_times_gt = spike_times_gt[spike_times_gt <= time_max]
    ##### end temporary zone
    
    # Load the spikes of all the "ground truth cells".
    spikes_gt = load_chunk(params, spike_times_gt, chans=chans)
    
    # Downsample to get the wanted number of spikes.
    N_gt = spikes_gt.shape[2]
    if N_gt_max < N_gt:
        idx_gt = numpy.random.choice(N_gt, size=N_gt_max, replace=False)
        spikes_gt = spikes_gt[:, :, idx_gt]
    
    if comm.rank == 0:
        if make_plots not in ['None', '']:
            plot_filename = "beer-trigger-times-gt.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            plot.view_trigger_times(filename, spike_times_gt, color='green', save=path)
            if make_plots_snippets:
                directory = "beer-trigger-snippets-gt"
                path = os.path.join(plot_path, directory)
                plot.view_trigger_snippets(spikes_gt, chans, save=path)
    
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
    y_gt = numpy.zeros((N_gt, 1))
    
    if comm.rank == 0:
        if verbose:
            msg = [
                "# X_gt.shape: {}".format(X_gt.shape),
                "# y_gt.shape: {}".format(y_gt.shape),
            ]
            io.print_and_log(msg, level='default', logger=params)
        
        if make_plots not in ['None', '']:
            plot_filename = "beer-dataset-gt.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            plot.view_dataset(X_gt, color='green', title="Ground-truth dataset", save=path)
    
   
    
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
    # Filter out the end of the data (~30%).
    spike_times_ngt_tmp = [t[t <= time_max] for t in spike_times_ngt_tmp]
    ##### end temporary zone
    
    # Filter the spike times of the "non ground truth cell".
    ## Restrict to spikes which happened in the vicinity.
    spike_times_ngt_tmp = [spike_times_ngt_tmp[chan] for chan in chans]
    spike_times_ngt_tmp = numpy.concatenate(spike_times_ngt_tmp)
    spike_times_ngt_tmp = numpy.unique(spike_times_ngt_tmp)
    ## Restrict to spikes which are far from ground truth spikes.
    spike_times_ngt_tmp = numpy.setdiff1d(spike_times_ngt_tmp, spike_times_fbd)
    ## Downsample to get the wanted number of spikes.
    N_ngt = spike_times_ngt_tmp.shape[0]
    if N_ngt_max < N_ngt:
        idxs_ngt = numpy.random.choice(N_ngt, size=N_ngt_max, replace=False)
        idxs_ngt = numpy.sort(idxs_ngt)
        spike_times_ngt = spike_times_ngt_tmp[idxs_ngt]
    
    # Load the spikes of all the "non ground truth cells".
    spikes_ngt = load_chunk(params, spike_times_ngt, chans=chans)
    
    if comm.rank == 0:
        if make_plots not in ['None', '']:
            plot_filename = "beer-trigger-times-ngt.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            plot.view_trigger_times(filename, spike_times_ngt, color='blue', save=path)
            if make_plots_snippets:
                directory = "beer-trigger-snippets-ngt"
                path = os.path.join(plot_path, directory)
                plot.view_trigger_snippets(spikes_ngt, chans, save=path)
    
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
        
        if make_plots not in ['None', '']:
            plot_filename = "beer-dataset-ngt.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            plot.view_dataset(X_ngt, color='blue', title="Non ground-truth dataset", save=path)
    
    
    
    ##### NOISE SAMPLES ########################################################
    
    if comm.rank == 0:
        io.print_and_log(["Collecting noise samples..."], level='default', logger=params)
    
    # Extract the noise times.
    ## Draw times uniformly.
    size = spike_times_ngt_tmp.size
    spike_times_noi = numpy.random.random_integers(time_min, time_max, size)
    spike_times_noi = numpy.unique(spike_times_noi)
    ## Restrict to spikes which are far from ground truth spikes.
    spike_times_noi = numpy.setdiff1d(spike_times_noi, spike_times_fbd)
    ## Downsample to get the wanted number of spikes.
    N_noi = spike_times_noi.shape[0]
    if N_noi_max < N_noi:
        idxs_noi = numpy.random.choice(N_noi, size=N_noi_max, replace=False)
        idxs_noi = numpy.sort(idxs_noi)
        spike_times_noi = spike_times_noi[idxs_noi]
    
    # Load the chunks for noise.
    spikes_noi = load_chunk(params, spike_times_noi, chans=chans)
    
    if comm.rank == 0:
        if make_plots not in ['None', '']:
            plot_filename = "beer-trigger-times-noi.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            plot.view_trigger_times(filename, spike_times_noi, color='red', save=path)
            if make_plots_snippets:
                directory = "beer-trigger-snippets-noi"
                path = os.path.join(plot_path, directory)
                plot.view_trigger_snippets(spikes_noi, chans, save=path)
    
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
        
        if make_plots not in ['None', '']:
            plot_filename = "beer-dataset-noi.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            plot.view_dataset(X_noi, color='red', title="Noise dataset", save=path)
    
    
    
    # NORMALIZE DATASETS #######################################################
    
    if comm.rank == 0:
        io.print_and_log(["Normalizing datasets..."], level='default', logger=params)
    
    
    X_raw = numpy.vstack((X_gt, X_ngt, X_noi))
    norm_scale = numpy.mean(numpy.linalg.norm(X_raw, axis=1))
    X_gt = X_gt / norm_scale
    X_ngt = X_ngt / norm_scale
    X_noi = X_noi / norm_scale
    
    
    
    ##### SAMPLES ##############################################################
    
    if comm.rank == 0:
        io.print_and_log(["Samples..."], level='debug', logger=params)
    
    # Option to include the pairwise product of feature vector elements.
    pairwise = True
    
    # Create the datasets to train the neural network.
    ## Create the input dataset.
    N = X_gt.shape[1]
    
    X_raw = numpy.vstack((X_gt, X_ngt, X_noi))
    
    if pairwise:
        # With pairwise product of feature vector elements.
        M = N + N * (N + 1) / 2
        shape = (N_gt + N_ngt + N_noi, M)
    else:
        # Without pairwise product of feature vector elments.
        M = N
        shape = (N_gt + N_ngt + N_noi, M)
    X = numpy.zeros(shape)
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
    y_raw = y_raw.flatten()
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
    
    
    
    ##### SANITY PLOT ##########################################################
    
    if comm.rank == 0:
        
        #io.print_and_log(["Sanity plot..."], level='info', logger=params)
        
        
        if make_plots not in ['None', '']:
            plot_filename = "beer-datasets.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            Xs = [X_ngt, X_noi, X_gt]
            ys = [y_ngt, y_noi, y_gt]
            colors = ['b', 'r', 'g']
            labels = ["non ground truth", "noise", "ground truth"]
            plot.view_datasets(Xs, ys, colors=colors, labels=labels, save=path)
    
    
    
    ##### INITIAL PARAMETER ####################################################
    
    if comm.rank == 0:
        io.print_and_log(["Initialising parameters..."], level='default', logger=params)
    
    
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
        
        if make_plots not in ['None', '']:
            # Plot accuracy curve.
            title = "Accuracy curve for the initial parameter"
            plot_filename = "beer-accuracy-plot.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            plot.view_accuracy(Mhlnb[indices], accs, Mhlnb[indices[i_opt]],
                               accs[i_opt], title=title, save=path)
    
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
            plot_filename = "beer-normalized-accuray-plot.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            plot.view_normalized_accuracy(Mhlnb[indices], tprs, tnrs, norm_accs,
                                          Mhlnb[indices[i_opt]], norm_accs[i_opt],
                                          title=title, save=path)
    
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
        
        if make_plots not in ['None', '']:
            # Plot ROC curve.
            title = "ROC curve for the inital parameter"
            plot_filename = "beer-roc-curve-initial.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            plot.view_roc_curve(fprs, tprs, fpr, tpr, title=title, save=path)
    
    # Scale the ellipse according to the chosen cutoff.
    A_init = (1.0 / cutoff) * A_init
    b_init = (1.0 / cutoff) * b_init
    c_init = (1.0 / cutoff) * (c_init + 1.0) - 1.0
    
    
    
    # SANITY PLOT (CLASSIFIER PROJECTION) ######################################
    
    if comm.rank == 0:
        io.print_and_log(["Sanity plot (classifier projection)..."],
                         level='info', logger=params)
        
        
        if make_plots not in ['None', '']:
            # Plot initial classifier (ellipsoid).
            title = "Initial classifier (ellipsoid)"
            plot_filename = "beer-classifier-projection-init.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            plot.view_classifier(filename, [X_gt, X_ngt, X_noi], [y_gt, y_ngt, y_noi],
                                 A_init, b_init, c_init,
                                 title=title, save=path, verbose=verbose)
    
    
    
    # MAHALANOBIS DISTRIBUTIONS ################################################
    
    if comm.rank == 0:
        
        io.print_and_log(["Compute intial Mahalanobis distributions..."],
                         level='info', logger=params)
        
        
        # Compute mahalanobis distributions.
        mu = numpy.mean(X_gt, axis=0)
        Mhlnb_gt = squared_Mahalanobis_distance(A_init, mu, X_gt)
        Mhlnb_ngt = squared_Mahalanobis_distance(A_init, mu, X_ngt)
        Mhlnb_noi = squared_Mahalanobis_distance(A_init, mu, X_noi)
        
        if make_plots not in ['None', '']:
            # Plot Mahalanobis distributions.
            title = "Mahalanobis distributions (ellipsoid)"
            plot_filename = "beer-mahalanobis-init.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            plot.view_mahalanobis_distribution(Mhlnb_gt, Mhlnb_ngt, Mhlnb_noi,
                                               title=title, save=path)
    
    
    
    ##### LEARNING #############################################################
    
    if comm.rank == 0:
        io.print_and_log(["Learning..."], level='info', logger=params)
    
    
    # mode = 'decision'
    mode = 'prediction'
    
    # Preprocess dataset.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    if comm.rank == 0:
        if verbose:
            msg = [
                "# X_train.shape: {}".format(X_train.shape),
                "# X_test.shape: {}".format(X_test.shape),
            ]
            io.print_and_log(msg, level='default', logger=params)
    
    model = 'sgd'
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
        _, _, class_weights = get_class_weights(y_gt, y_ngt, y_noi, n=1)
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
    
    if comm.rank == 0:
        if make_plots not in ['None', '']:
            # Plot prediction.
            title = "Initial prediction (random)"
            plot_filename = "beer-prediction-init-random.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            plot.view_classification(clf, X, X_raw, y_raw, mode='predict',
                                     title=title, save=path)
            # Plot decision function.
            title = "Initial decision function (random)"
            plot_filename = "beer-decision-function-init-random.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            plot.view_classification(clf, X, X_raw, y_raw, mode='decision_function',
                                     title=title, save=path)
    
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
            title = "Initial prediction (ellipsoid)"
            plot_filename = "beer-prediction-init-ellipsoid.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            plot.view_classification(clf, X, X_raw, y_raw, mode='predict',
                                     title=title, save=path)
            # Plot decision function.
            title = "Initial decision function (ellipsoid)"
            plot_filename = "beer-decision-function-init-ellipsoid.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            plot.view_classification(clf, X, X_raw, y_raw, mode='decision_function',
                                     title=title, save=path)
    
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
        n_iter = min(max_iter, 1000000 / N_max)
        clf.set_params(n_iter=n_iter)
        clf.set_params(eta0=learning_rate_init)
        clf.set_params(warm_start=True)
    clf.fit(X_train, y_train)
    
    if comm.rank == 0:
        if make_plots not in ['None', '']:
            # Plot final prediction.
            title = "Final prediction "
            plot_filename = "beer-prediction-final.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            plot.view_classification(clf, X, X_raw, y_raw, mode='predict',
                                     title=title, save=path)
            # Plot final decision function.
            title = "Final decision function"
            plot_filename = "beer-decision-function-final.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            plot.view_classification(clf, X, X_raw, y_raw, mode='decision_function',
                                     title=title, save=path)
    
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
    
    if comm.rank == 0:
        
        io.print_and_log(["Sanity plot (classifier projection)..."],
                         level='info', logger=params)
        
        
        if make_plots not in ['None', '']:
            # Plot final classifier.
            title = "Final classifier"
            plot_filename = "beer-classifier-projection-final.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            plot.view_classifier(filename, [X_gt, X_ngt, X_noi], [y_gt, y_ngt, y_noi],
                                 A, b, c, title=title, save=path, verbose=verbose)
    
    
    
    # MAHALANOBIS DISTRIBUTIONS ################################################
    
    if comm.rank == 0:
        
        io.print_and_log(["Compute final Mahalanobis distributions..."],
                         level='info', logger=params)
        
        
        # Compute the Mahalanobis distributions.
        mu = numpy.mean(X_gt, axis=0)
        Mhlnb_gt = squared_Mahalanobis_distance(A, mu, X_gt)
        Mhlnb_ngt = squared_Mahalanobis_distance(A, mu, X_ngt)
        Mhlnb_noi = squared_Mahalanobis_distance(A, mu, X_noi)
        
        if verbose:
            msg = [
                "# Mhlnb_gt: {}".format(Mhlnb_gt),
            ]
            io.print_and_log(msg, level='default', logger=params)
        
        if make_plots not in ['None', '']:
            # Plot Mahalanobis distributions.
            title = "Final Mahalanobis distributions"
            plot_filename = "beer-mahalanobis-final.%s" %make_plots
            path = os.path.join(plot_path, plot_filename)
            plot.view_mahalanobis_distribution(Mhlnb_gt, Mhlnb_ngt, Mhlnb_noi,
                                               title=title, save=path)
    
    
    
    # Synchronize CPUs before weighted learning.
    comm.Barrier()
    
    
    
    ##### WEIGHTED LEARNING ####################################################
    
    if comm.rank == 0:
        io.print_and_log(["Weighted learning..."], level='info', logger=params)
    
    
    _, _, class_weights = get_class_weights(y_gt, y_ngt, y_noi, n=roc_sampling)
    
    # Distribute weights over the CPUs.
    loc_indices = numpy.arange(comm.rank, roc_sampling, comm.size)
    loc_class_weights = [class_weights[loc_index] for loc_index in loc_indices]
    loc_nb_class_weights = len(loc_class_weights)
    
    # Preallocation to collect results.
    confusion_matrices = loc_nb_class_weights * [None]
    
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
            n_iter = min(max_iter, 1000000 / N_max)
            wclf.set_params(n_iter=n_iter)
            # wclf.set_params(eta0=learning_rate_init)
            wclf.set_params(warm_start=True)
            wclf.fit(X_train, y_train)
            
            
            ##### TODO: fix depreciated zone
            
            time_chunk_size = 5000
            nb_time_chunks = (time_max_test - time_min_test + 1) / time_chunk_size
            if 0 < (time_max_test - time_min_test + 1) % time_chunk_size:
                nb_time_chunks = nb_time_chunks + 1
            ##### TODO: remove temporary zone
            time_pred = nb_time_chunks * [None]
            ##### end temporary zone
            y_pred = nb_time_chunks * [None]
            for time_chunk in xrange(0, nb_time_chunks):
                ##### TODO: remove temporary zone
                if comm.rank == 0:
                    print("{} / {}".format(time_chunk, nb_time_chunks))
                ##### end temporary zone
                time_start = time_min_test + time_chunk * time_chunk_size
                time_end = time_min_test + (time_chunk + 1) * time_chunk_size
                time_end = min(time_end, time_max_test + 1)
                spike_times_ = numpy.arange(time_start, time_end)
                # Load the snippets
                spikes_ = load_chunk(params, spike_times_, chans=chans)
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
                # Normalize data.
                X_ = X_ / norm_scale
                # Add quadratic features.
                X_ = with_quadratic_feature(X_, pairwise=True)
                ##### TODO: remove temporary zone
                time_pred[time_chunk] = spike_times_
                ##### end temporary zone
                # Compute the predictions.
                y_pred[time_chunk] = wclf.predict(X_)
                # y_pred[time_chunk] = wclf.decision_function(X_)
            ##### TODO: remove temporary zone
            time_pred = numpy.concatenate(time_pred)
            ##### end temporary zone
            y_pred = numpy.concatenate(y_pred)
            
            # TODO: filter y_pred with a time window (i.e. at least ? time slot between detections).
            
            # TODO: remove following lines
            # # Classifer prediction on train set.
            # y_pred = wclf.predict(X_test)

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
            confusion_matrices[count] = confusion_matrix
            
            ##### end depreciated zone
            
            
            if comm.rank == 0:
                pbar.update(count)
    else:
        raise Exception("Unsupported classifier: model={}".format(model))
    
    if comm.rank == 0:
        pbar.finish()
    
    comm.Barrier()
    
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
            del beer_file[class_weights_key]
        beer_file.create_dataset(class_weights_key, data=class_weights_)
        ## Save confusion matrices.
        confusion_matrices_ = numpy.array(confusion_matrices)
        confusion_matrices_key = "confusion_matrices"
        if confusion_matrices_key in beer_file.keys():
            del beer_file[confusion_matrices_key]
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
            plot.view_roc_curve(fprs, tprs, None, None, title=title, save=path,
                                xlim=[0.0, 1.0], ylim=[0.0, 1.0])
    
    
    
    ############################################################################
    
    if comm.rank == 0:
        io.print_and_log(["Validation done."], level='info', logger=params)
    
    return
