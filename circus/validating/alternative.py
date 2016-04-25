from sklearn.linear_model import SGDClassifier

from ..shared.utils import *
from ..shared.files import get_stas
from ..shared import plot
from .utils import *



def main_alternative(filename, params, nb_cpu, nb_gpu, us_gpu):
    
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
    
    verbose   = True
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
    data_block = numpy.memmap(data_file, offset=data_offset, dtype=data_dtype, mode='r')
    N = len(data_block)
    data_len = N // N_total
    time_min = template_shift
    time_max = (data_len - 1) - template_shift
    
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
    
    
    
    ##### GROUND TRUTH CELL'S SAMPLES ##########################################
    
    if comm.rank == 0:
        io.print_and_log(["Collecting ground truth cell's samples..."], level='debug', logger=params)
    
    # Retrieve the spike times of the "ground truth cell".
    spike_times_gt = spike_times_juxta
    
    if comm.rank == 0:
        if verbose:
            msg = [
                "spike_times_gt.size: {}".format(spike_times_gt.size),
            ]
            io.print_and_log(msg, level='default', logger=params)
    
    labels = numpy.zeros(spike_times_gt.size)
    spikes_gt = get_stas(params, spike_times_gt, labels, chan, chans, nodes=nodes, auto_align=False).T
    
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
                "X_gt.shape: {}".format(X_gt.shape),
                "y_gt.shape: {}".format(y_gt.shape),
            ]
            io.print_and_log(msg, level='default', logger=params)
    
    
    
    ############################################################################
    
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
    
    # Filter the spike times of the "non ground truth cell".
    ## Restrict to spikes which happened in the vicinity.
    spike_times_ngt_tmp = [spike_times_ngt_tmp[chan] for chan in chans]
    spike_times_ngt_tmp = numpy.concatenate(spike_times_ngt_tmp)
    spike_times_ngt_tmp = numpy.unique(spike_times_ngt_tmp)
    ## Restrict to spikes which are far from ground truth spikes.
    spike_times_ngt_tmp = numpy.setdiff1d(spike_times_ngt_tmp, spike_times_fbd)
    
    spike_times_ngt = spike_times_ngt_tmp
    
    if comm.rank == 0:
        if verbose:
            msg = [
                "spike_times_ngt.size: {}".format(spike_times_ngt.size),
            ]
            io.print_and_log(msg, level='default', logger=params)
    
    labels = numpy.zeros(spike_times_ngt.size)
    spikes_ngt = get_stas(params, spike_times_ngt, labels, chan, chans, nodes=nodes, auto_align=False).T
    
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
                "X_ngt.shape: {}".format(X_ngt.shape),
                "y_ngt.shape: {}".format(y_ngt.shape),
            ]
            io.print_and_log(msg, level='default', logger=params)
    
    
    
    # NORMALIZE DATASETS #######################################################
    
    if comm.rank == 0:
        io.print_and_log(["Normalizing datasets..."], level='debug', logger=params)
    
    
    X_raw = numpy.vstack((X_gt, X_ngt))
    norm_scale = numpy.mean(numpy.linalg.norm(X_raw, axis=1))
    X_gt  /= norm_scale
    X_ngt /= norm_scale
    
    
    
    ##### SAMPLES ##############################################################
    
    if comm.rank == 0:
        io.print_and_log(["Samples..."], level='debug', logger=params)
    
    
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
                io.print_and_log(msg, level='default', logger=params)
    
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
            io.print_and_log(msg, level='default', logger=params)
    
    
    
    ##### SANITY PLOT ##########################################################
    
    if comm.rank == 0:
        
        #io.print_and_log(["Sanity plot..."], level='info', logger=params)
        
        
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
        io.print_and_log(["Initializing parameters for the non-linear classifier..."], level='default', logger=params)
    
    
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
            io.print_and_log(msg, level='default', logger=params)
    
    
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
            # io.print_and_log(msg, level='default', logger=params)
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
            
            data_class_1 = [X_gt, X_ngt], [y_gt, y_ngt], A_init, b_init, c_init
    
    
    
    # MAHALANOBIS DISTRIBUTIONS ################################################
    
    if comm.rank == 0:
        
        io.print_and_log(["Intialising Mahalanobis distributions..."],
                         level='debug', logger=params)
        
        
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
        io.print_and_log(["Start learning..."], level='debug', logger=params)
    
    
    if comm.rank == 0:
        if verbose:
            msg = [
                "X_train.shape: {}".format(X_train.shape),
                "X_test.shape: {}".format(X_test.shape),
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
                # "clf.loss_: {}".format(clf.loss_),
                # # Print the loss curve.
                # "clf.loss_curve_: {}".format(clf.loss_curve_),
                # # Print the number of iterations the algorithm has ran.
                # "clf.n_iter_: {}".format(clf.n_iter_),
                # Print the score on the test set.
                "accuracy_score(X_test, y_test): {} ({})".format(score, 1.0 - score),
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
                # "clf.loss_: {}".format(clf.loss_),
                # # Print the loss curve.
                # "clf.loss_curve_: {}".format(clf.loss_curve_),
                # # Print the number of iterations the algorithm has ran.
                # "clf.n_iter_: {}".format(clf.n_iter_),
                # Print the score on the test set.
                "accuracy_score(X_test, y_test): {} ({})".format(score, 1.0 - score),
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
                         level='debug', logger=params)
        
        
        if make_plots not in ['None', '']:
            # Plot final classifier.
            title = "Final classifier"
            plot_filename = "beer-classifier-projection.%s" %make_plots
            
            data_class_2 = [X_gt, X_ngt], [y_gt, y_ngt], A, b, c
            
            path = os.path.join(plot_path, plot_filename)
            plot.view_classifier(params, data_class_1, data_class_2, save=path, verbose=verbose)
    
    
    
    # MAHALANOBIS DISTRIBUTIONS ################################################
    
    if comm.rank == 0:
        
        io.print_and_log(["Computing final Mahalanobis distributions..."],
                         level='debug', logger=params)
        
        
        # Compute the Mahalanobis distributions.
        mu = numpy.mean(X_gt, axis=0)
        Mhlnb_gt = squared_Mahalanobis_distance(A, mu, X_gt)
        Mhlnb_ngt = squared_Mahalanobis_distance(A, mu, X_ngt)
        
        data_mal2 = (Mhlnb_gt, Mhlnb_ngt)
        
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
    
    
    _, _, class_weights = get_class_weights(y_gt, y_ngt, n=roc_sampling)
    
    # Distribute weights over the CPUs.
    loc_indices = numpy.arange(comm.rank, roc_sampling, comm.size)
    loc_class_weights = [class_weights[loc_index] for loc_index in loc_indices]
    loc_nb_class_weights = len(loc_class_weights)
    
    # Preallocation to collect results.
    confusion_matrices = loc_nb_class_weights * [None]
    y_decfs = loc_nb_class_weights * [None]
    y_preds = loc_nb_class_weights * [None]
    
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
            
            if comm.rank == 0:
                pbar.update(count)
            
    else:
        raise Exception("Unsupported classifier: model={}".format(model))
    
    if comm.rank == 0:
        pbar.finish()
    
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
        
        
        ##### TODO: clean temporary zone
        # TODO: compute the error of SpyKING CIRCUS.
        
        # mode = 'perso'
        mode = 'harris'
        
        if mode == 'perso':
            
            # # Retrieve the juxtacellular spiketimes.
            # juxta_times = load_data(params, "juxta-triggers")
            # # Retrieve the extracellular spiketimes.
            # extra_times = load_data(params, "extra-triggers")
            # # Filter out the extracellular spiketimes not in the neighborhood.
            # extra_times = [extra_times[chan] for chan in chans]
            spike_times_gt = spike_times_gt
            spike_times_ngt = spike_times_ngt
            
            # Define the "matching threshold".
            thresh = int(params.getint('data', 'sampling_rate')*2*1e-3)
            # thresh = int(params.getint('data', 'sampling_rate')*0.25*1e-3)
            
            # Retrieve the SpyKING CIRCUS spiketimes.
            result = io.load_data(params, "results")
            data   = result['spiketimes']
            
            # Retrieve the templates.
            templates = io.load_data(params, 'templates')
            
            n_temp = len(data)
            res = numpy.zeros((n_temp, 2))
            tot = numpy.zeros((n_temp, 2))
            
            # First pass to detect what are the scores.
            for i in xrange(n_temp):
                spikes = data['temp_' + str(i)]
                # Compute the false positive rate.
                for spike in spike_times_gt:
                    idx = numpy.where(abs(spikes - spike) <= thresh)[0]
                    if 0 < len(idx):
                        res[i, 0] += 1.0
                        tot[i, 0] += 1.0
                for spike in spike_times_ngt:
                    idx = numpy.where(abs(spikes - spike) <= thresh)[0]
                    if 0 < len(idx):
                        tot[i, 0] += 1.0
                if 0.0 < tot[i, 0]:
                    res[i, 0] /= tot[i, 0]
                # Compute the positive predictive value.
                for spike in spikes:
                    idx = numpy.where(abs(spike_times_gt - spike) <= thresh)[0]
                    if 0 < len(idx):
                        res[i, 1] += 1.0
                        tot[i, 1] += 1.0
                for spike in spikes:
                    idx = numpy.where(abs(spike_times_ngt - spike) <= thresh)[0]
                    if 0 < len(idx):
                        tot[i, 1] += 1.0
                if 0 < tot[i, 1]:
                    res[i, 1] /= tot[i, 1]
            
            idx = numpy.argmax(numpy.mean(res, 1))
            selection = [idx]
            error = res[idx]
            find_next = True
            source_temp = templates[:, idx].toarray().flatten()
            temp_match = []
            dmax = 0.1
            for i in xrange(templates.shape[1]/2):
                d = numpy.corrcoef(templates[:, i].toarray().flatten(), source_temp)[0, 1]
                ##### TODO: remove debug zone
                # print("i, d: {}, {}".format(i, d))
                ##### end debug zone
                if d > dmax and i not in selection:
                    temp_match += [i]
            ##### TODO: remove debug zone
            # print("temp_match: {}".format(temp_match))
            ##### end debug zone
            
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
                        total = 0.0
                        for spike in spike_times_gt:
                            idx = numpy.where(numpy.abs(spikes - spike) < thresh)[0]
                            if 0 < len(idx):
                                count += 1.0
                                total += 1.0
                        for spike in spike_times_ngt:
                            idx = numpy.where(numpy.abs(spikes - spike) < thresh)[0]
                            if 0 < len(idx):
                                total += 1.0
                        if 0.0 < total:
                            local_errors[mcount, 0] = count / total
                        
                        # Compute positive predictive value
                        count = 0.0
                        total = 0.0
                        for spike in spikes:
                            idx = numpy.where(numpy.abs(spike_times_gt - spike) < thresh)[0]
                            if 0 < len(idx):
                                count += 1.0
                                total += 1.0
                        for spike in spikes:
                            idx = numpy.where(numpy.abs(spike_times_ngt - spike) < thresh)[0]
                            if 0 < len(idx):
                                total += 1.0
                        if 0.0 < total:
                            local_errors[mcount, 1]  = count / total
                    
                    errors = numpy.mean(local_errors, 1)
                    ##### TODO: remove debug zone
                    # print("numpy.max(errors): {}".format(numpy.max(errors)))
                    # print("numpy.mean(error): {}".format(numpy.mean(error)))
                    ##### end debug zone
                    if numpy.max(errors) > numpy.mean(error):
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
        
        elif mode == 'harris':
            
            spike_times_gt = spike_times_gt
            
            # Define the "matching threshold".
            thresh = int(params.getint('data', 'sampling_rate')*2*1e-3)
            # thresh = int(params.getint('data', 'sampling_rate')*0.25*1e-3)
            
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
                ##### TODO: remove debug zone
                # print("i, d: {}, {}".format(i, d))
                ##### end debug zone
                if d > dmax and i not in selection:
                    temp_match += [i]
            ##### TODO: remove debug zone
            # print("temp_match: {}".format(temp_match))
            ##### end debug zone
            
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
                    ##### TODO: remove debug zone
                    # print("numpy.max(errors): {}".format(numpy.max(errors)))
                    # print("numpy.mean(error): {}".format(numpy.mean(error)))
                    ##### end debug zone
                    if numpy.max(errors) > numpy.mean(error):
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
            # error = plot.view_roc_curve(params, fprs, tprs, None, None, save=path)
            error = plot.view_roc_curve(params, fprs, tprs, None, None, scerror=scerror, save=path)
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
