from .shared.utils import *


def main(filename, params, nb_cpu, nb_gpu, use_gpu):
    
    io.print_and_log(["Start validation..."], level='info', logger=params)
    
    
    # RETRIEVE PARAMETERS FOR VALIDATING #######################################
    
    nb_repeats = params.getint('clustering', 'nb_repeats')
    max_iter = params.getint('validating', 'max_iter')
    learning_rate_init = params.getfloat('validating', 'learning_rate')
    verbose = params.getboolean('validating', 'verbose')
    make_plots = params.getboolean('validating', 'make_plots')
    plot_path = os.path.join(params.get('data', 'data_file_noext'), 'plots')
    
    ############################################################################
    
    
    
    def get_neighbors(params, elec=43, radius=120):
        if radius is None:
            pass
        else:
            radius = 120 # um
            _ = params.set('data', 'radius', str(radius))
        N_total = params.getint('data', 'N_total')
        nodes, edges = io.get_nodes_and_edges(params)
        if elec is None:
            # Select all the electrodes.
            indices = nodes
        else:
            # Select only the neighboring electrodes of the best electrode.
            inv_nodes = numpy.zeros(N_total, dtype=numpy.int32)
            inv_nodes[nodes] = numpy.argsort(nodes)
            indices = inv_nodes[edges[nodes[elec]]]
        return indices
    
    # Define an auxiliary function to load spike data given spike times.
    def load_chunk(params, spike_times, elec=43):
        # Load the parameters of the spike data.
        data_file = params.get('data', 'data_file')
        data_offset = params.getint('data', 'data_offset')
        data_dtype = params.get('data', 'data_dtype')
        chunk_size = params.getint('data', 'chunk_size')
        N_total = params.getint('data', 'N_total')
        N_t = params.getint('data', 'N_t')
        dtype_offset = params.getint('data', 'dtype_offset')
        indices = get_neighbors(params)
        N_filt = indices.size
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
            local_chunk = local_chunk[:, indices]
            local_chunk = local_chunk.astype(numpy.float32)
            local_chunk -= dtype_offset
            # Save data.
            spikes[:, :, count] = local_chunk
        return spikes
    
    def squared_Mahalanobis_distance(A, mu, X):
        '''Compute the squared Mahalanobis distance.'''
        N = X.shape[0]
        d2 = numpy.zeros(N)
        for i in xrange(0, N):
            d2[i] = numpy.dot(X[i, :] - mu, numpy.dot(A, X[i, :] - mu))
        return d2

    
    
    # Initialize the random seed.
    numpy.random.seed(0)
    
    # Retrieve PCA basis.
    basis_proj, basis_rec = io.load_data(params, 'basis')
    N_p = basis_proj.shape[1]
    # Retrieve sampling rate.
    sampling_rate  = params.getint('data', 'sampling_rate')
    
    
    
    ##### GROUND TRUTH CELL'S SAMPLES ##########################################
    
    io.print_and_log(["# Ground truth cell's samples..."], level='info', logger=params)
    
    # Retrieve the spikes of the "ground truth cell".
    spike_times_gt, spikes_gt = io.load_data(params, 'triggers')
    
    # Select only the neighboring electrodes of the best electrode.
    indices = get_neighbors(params)
    spikes_gt = spikes_gt[:, indices, :]
    
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
    # Define the outputs.
    y_gt = numpy.zeros((N_gt, 1))
    
    
    if verbose:
        msg = [
            "# X_gt.shape",
            "%s" %(X_gt.shape,),
            "# y_gt.shape",
            "%s" %(y_gt.shape,),
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    
    ############################################################################
    
    # Compute the forbidden spike times.
    spike_times_fbd = []
    # TODO: check the validity of 'int'.
    max_time_shift = int(float(sampling_rate) * 0.25e-3)
    for time_shift in xrange(-max_time_shift, max_time_shift+1):
        spike_times_fbd.append(spike_times_gt + time_shift)
    spike_times_fbd = numpy.concatenate(spike_times_fbd)
    spike_times_fbd = numpy.unique(spike_times_fbd)
    
    
    
    ##### NON GROUND TRUTH CELL'S SAMPLES ######################################
    
    io.print_and_log(["# Non ground truth cell's samples..."], level='info', logger=params)
    
    # Retrieve the spikes of all the "non ground truth cells".
    clusters = io.load_data(params, 'clusters')
    ## Find all the spike times.
    spike_times_ngt_tmp = []
    keys = [key for key in clusters.keys() if "times_" in key]
    for key in keys:
        spike_times_ngt_tmp.append(clusters[key])
    spike_times_ngt_tmp = numpy.concatenate(spike_times_ngt_tmp)
    spike_times_ngt_tmp = numpy.unique(spike_times_ngt_tmp)
    spike_times_ngt_tmp = numpy.setdiff1d(spike_times_ngt_tmp, spike_times_fbd)
    alpha = 4
    idxs_ngt = numpy.random.choice(spike_times_ngt_tmp.size,
                                   size=alpha*spike_times_gt.shape[0],
                                   replace=False)
    idxs_ngt = numpy.unique(idxs_ngt)
    spike_times_ngt = spike_times_ngt_tmp[idxs_ngt]
    
    # Load the spikes of all the 'non ground truth cells".
    spikes_ngt = load_chunk(params, spike_times_ngt)
    
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
    # Define the outputs.
    y_ngt = numpy.ones((N_ngt, 1))
    
    
    if verbose:
        msg = [
            "# X_ngt.shape",
            "%s" %(X_ngt.shape,),
            "# y_ngt.shape",
            "%s" %(y_ngt.shape,),
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    
    ##### NOISE SAMPLES ########################################################
    
    io.print_and_log(["# Noise samples..."], level='info', logger=params)
    
    # Compute the PCA coordinates of each "non-spike" sample.
    # TODO: replace temporary solution for 'low' and 'high'.
    low = min(numpy.amin(spike_times_gt), numpy.amin(spike_times_ngt))
    high = max(numpy.amax(spike_times_gt), numpy.amin(spike_times_ngt))
    size = spike_times_ngt_tmp.size
    spike_times_noi = numpy.random.random_integers(low, high, size)
    spike_times_noi = numpy.unique(spike_times_noi)
    spike_times_noi = numpy.setdiff1d(spike_times_noi, spike_times_fbd)
    alpha = 4
    idxs_noi = numpy.random.choice(spike_times_noi.size, size=alpha*spike_times_gt.shape[0], replace=False)
    idxs_noi = numpy.unique(idxs_noi)
    spike_times_noi = spike_times_noi[idxs_noi]
    
    # TODO: filter ground truth spike times.
    
    # Load some "non-spike" samples.
    spikes_noi = load_chunk(params, spike_times_noi)
    
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
    # Define outputs.
    y_noi = numpy.ones((N_noi, 1))
    
    
    if verbose:
        msg = [
            "# X_noi.shape",
            "%s" %(X_noi.shape,),
            "# y_noi.shape",
            "%s" %(y_noi.shape,),
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    
    ##### SANITY PLOTS #########################################################
    
    io.print_and_log(["# Sanity plots..."], level='info', logger=params)
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    
    #max_component = X_gt.shape[1]
    max_component = 1 * 3
    
    if make_plots:
        fig = plt.figure()
        for x_component in xrange(0, max_component):
            for y_component in xrange(x_component + 1, max_component):
                fig.suptitle("Samples for validation")
                gs = gridspec.GridSpec(1, 1)
                ax = fig.add_subplot(gs[0])
                ax.scatter(X_noi[:, x_component], X_noi[:, y_component], c='r')
                ax.scatter(X_ngt[:, x_component], X_ngt[:, y_component], c='b')
                ax.scatter(X_gt[:, x_component], X_gt[:, y_component], c='g')
                ax.set_xlabel("%dst principal component" %(x_component + 1))
                ax.set_ylabel("%dnd principal component" %(y_component + 1))
                ax.grid()
                filename = "validation-samples-%d-%d.png" %(x_component, y_component)
                path = os.path.join(plot_path, filename)
                plt.savefig(path)
                fig.clear()
        plt.close(fig)
    
    
    
    ##### SAMPLES ##############################################################
    
    io.print_and_log(["# Samples..."], level='info', logger=params)
    
    # Option to include the pairwise product of feature vector elements.
    pairwise = True
    
    # Create the datasets to train the neural network.
    ## Create the input dataset.
    N = X_gt.shape[1]
    
    X_raw = numpy.vstack((X_gt, X_ngt, X_noi))
    
    if pairwise:
        # With pairwise product of feature vector elements.
        shape = (N_gt + N_ngt + N_noi, N + N * (N + 1) / 2)
    else:
        # Without pairwise product of feature vector elments.
        shape = (N_gt + N_ngt + N_noi, N)
    X = numpy.zeros(shape)
    X[:, :N] = X_raw
    
    if pairwise:
        # Add the pairwise product of feature vector elements.
        k = 0
        for i in xrange(0, N):
            for j in xrange(i, N):
                X[:, N + k] = numpy.multiply(X[:, i], X[:, j])
                k = k + 1
        
        if verbose:
            msg = [
                "# X.shape (with pairwise product of feature vector element",
                "%s" %(X.shape,),
            ]
            io.print_and_log(msg, level='default', logger=params)
    
    ## Create the output dataset.
    y_raw = numpy.vstack((y_gt, y_ngt, y_noi))
    y_raw = y_raw.flatten()
    y = y_raw
    
    
    if verbose:
        msg = [
            "# X_raw.shape",
            "%s" %(X_raw.shape,),
            "# y_raw.shape",
            "%s" %(y_raw.shape,),
            "# X.shape",
            "%s" %(X.shape,),
            "# y.shape",
            "%s" %(y.shape,),
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    
    ##### INITIAL PARAMETER ####################################################
    
    io.print_and_log(["# Initial parameter..."], level='info', logger=params)
    
    
    # Useful function to convert an ellispoid in standard form to an ellispoid
    # in general form.
    def ellipsoid_standard_to_general(t, s, O):
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
            io.print_and_log(msg, level='default', logger=params)
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
        
        
        if make_plots:
            fig = plt.figure()
            ax = fig.gca()
            cax = ax.imshow(c, interpolation='nearest')
            ax.set_title("Covariance matrix of ground truth")
            fig.colorbar(cax)
            filename = "covariance-matrix-gt.png"
            path = os.path.join(plot_path, filename)
            fig.savefig(path)
            plt.close(fig)
        
        
        s, O = numpy.linalg.eigh(c)
        coefs_init = ellipsoid_standard_to_general(t, s, O)
        
    else:
        
        raise(Exception)
    
    
    if verbose:
        msg = [
            "# coefs_init",
            "%s" %(coefs_init,),
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    # Compute false positive rate and true positive rate for various cutoffs.
    num = 100
    
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
    
    
    if verbose:
        # msg = [
        #     "# cutoffs",
        #     "%s" %(cutoffs,),
        #     "# fprs",
        #     "%s" %(fprs,),
        #     "# tprs",
        #     "%s" %(tprs,),
        # ]
        # io.print_and_log(msg, level='default', logger=params)
        pass
    
    
    # Compute false positive rate and true positive rate for the chosen cutoff.
    cutoff = 89.33 # 0.5 quantile of the chi^2 distribution with 90 degree of freedom
    fp = float(numpy.count_nonzero(Mhlnb_ngt < cutoff)
               + numpy.count_nonzero(Mhlnb_noi < cutoff))
    n = float(Mhlnb_ngt.size + Mhlnb_noi.size)
    fpr = fp / n
    tp = float(numpy.count_nonzero(Mhlnb_gt < cutoff))
    p = float(Mhlnb_gt.size)
    tpr = tp / p
    
    
    if verbose:
        msg = [
            "# cutoff",
            "%s" %(cutoff,),
            "# fpr",
            "%s" %(fpr,),
            "# tpr",
            "%s" %(tpr),
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    if make_plots:
        # Plot ROC curve.
        fig = plt.figure()
        ax = fig.gca()
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title("ROC curve for the inital parameter")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.plot([0.0, 1.0], [0.0, 1.0], 'k--')
        ax.plot(fprs, tprs, 'b-')
        ax.plot(fpr, tpr, 'bo')
        filename = "roc-curve.png"
        path = os.path.join(plot_path, filename)
        fig.savefig(path)
        plt.close(fig)
    
    
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
    
    
    if verbose:
        msg = [
            "# cutoff_opt_acc",
            "%s" %(cutoff_opt_acc,),
            "# acc_opt",
            "%s" %accs[i_opt],
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    if make_plots:
        # Plot accuracy plot.
        fig = plt.figure()
        ax = fig.gca()
        ax.set_title("Accuracy curve for the initial parameter")
        ax.set_xlabel("Cutoff")
        ax.set_ylabel("Accuracy")
        ax.set_xlim([numpy.amin(Mhlnb[indices]), numpy.amax(Mhlnb[indices])])
        ax.set_ylim([0.0, 1.0])
        ax.grid(True)
        ax.plot(Mhlnb[indices], accs, 'b-')
        ax.plot(Mhlnb[indices[i_opt]], accs[i_opt], 'bo')
        filename = "accuracy-plot.png"
        path = os.path.join(plot_path, filename)
        fig.savefig(path)
        plt.close(fig)
    
    
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
    
    
    if verbose:
        msg = [
            "# cutoff_opt_norm_acc",
            "%s" %(cutoff_opt_norm_acc,),
            "# norm_acc_opt",
            "%s" %(norm_accs[i_opt],),
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    if make_plots:
        # Plot normalized accuracy plot.
        fig = plt.figure()
        ax = fig.gca()
        ax.set_title("Normalized accuracy curve for the initial parameter")
        ax.set_xlabel("Cutoff")
        ax.set_ylabel("Normalized accuracy")
        ax.set_xlim([numpy.amin(Mhlnb[indices]), numpy.amax(Mhlnb[indices])])
        ax.set_ylim([0.0, 1.0])
        ax.grid(True)
        ax.plot(Mhlnb[indices], norm_accs, 'b-')
        ax.plot(Mhlnb[indices], tprs, 'g-')
        ax.plot(Mhlnb[indices], tnrs, 'r-')
        ax.plot(Mhlnb[indices[i_opt]], norm_accs[i_opt], 'bo')
        filename = "normalized-accuray-plot.png"
        path = os.path.join(plot_path, filename)
        fig.savefig(path)
        plt.close(fig)
    
    
    # Set cutoff equal to the optimal cutoff.
    cutoff = cutoff_opt_acc
    
    # Scale the ellipse according to the chosen cutoff.
    A_init = (1.0 / cutoff) * A_init
    b_init = (1.0 / cutoff) * b_init
    c_init = (1.0 / cutoff) * (c_init + 1.0) - 1.0
    
    
    
    # SANITY PLOT (ELLIPSE PROJECTION) #########################################
    
    io.print_and_log(["# Sanity plot (ellipse projection)..."], level='info', logger=params)
    
    from sklearn.decomposition import PCA
    
    
    # Compute PCA with two components.
    n_components = 2
    pca = PCA(n_components)
    _ = pca.fit(X_raw)
    
    # Data transformation.
    X_raw_ = pca.transform(X_raw)
    X_gt_ = pca.transform(X_gt)
    X_ngt_ = pca.transform(X_ngt)
    X_noi_ = pca.transform(X_noi)
    
    # Mean transformation.
    mu_gt = numpy.mean(X_gt, axis=0).reshape(1, -1)
    mu_gt_ = pca.transform(mu_gt)
    mu_ngt = numpy.mean(X_ngt, axis=0).reshape(1, -1)
    mu_ngt_ = pca.transform(mu_ngt)
    mu_noi = numpy.mean(X_noi, axis=0).reshape(1, -1)
    mu_noi_ = pca.transform(mu_noi)
    
    # Ellipse transformation.
    f = 0.25 * numpy.dot(numpy.dot(b_init, numpy.linalg.inv(A_init)), b_init) - c_init
    t = - 0.5 * numpy.dot(numpy.linalg.inv(A_init), b_init).reshape(1, -1)
    s, O = numpy.linalg.eigh(numpy.linalg.inv((1.0 / f) * A_init))
    s = numpy.sqrt(s)
    t_ = pca.transform(t)
    O_ = pca.transform(numpy.multiply(O, s).T + t)
    
    
    if verbose:
        msg = [
            "# s (i.e. demi-axes)",
            "%s" %(s,),
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    # Find plot limits.
    pad = 0.3
    x_dif = numpy.amax(X_raw_[:, 0]) - numpy.amin(X_raw_[:, 0])
    x_min = numpy.amin(X_raw_[:, 0]) - pad * x_dif
    x_max = numpy.amax(X_raw_[:, 0]) + pad * x_dif
    y_dif = numpy.amax(X_raw_[:, 1]) - numpy.amin(X_raw_[:, 1])
    y_min = numpy.amin(X_raw_[:, 1]) - pad * y_dif
    y_max = numpy.amax(X_raw_[:, 1]) + pad * y_dif
    
    if make_plots:
        # Plot.
        fig = plt.figure()
        ax = fig.gca()
        ax.set_aspect('equal')
        ax.grid()
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel("1st component")
        ax.set_ylabel("2nd component")
        # Plot datasets.
        ax.scatter(X_gt_[:, 0], X_gt_[:, 1], c='g', s=5, lw=0.1)
        ax.scatter(X_ngt_[:, 0], X_ngt_[:, 1], c='b', s=5, lw=0.1)
        ax.scatter(X_noi_[:, 0], X_noi_[:, 1], c='r', s=5, lw=0.1)
        # Plot ellipse transformation.
        for i in xrange(0, O_.shape[0]):
            ax.plot([t_[0, 0], O_[i, 0]], [t_[0, 1], O_[i, 1]], 'y', zorder=3)
        # Plot means of datasets.
        ax.scatter(mu_gt_[:, 0], mu_gt_[:, 1], c='y', s=30, lw=0.1, zorder=4)
        ax.scatter(mu_ngt_[:, 0], mu_ngt_[:, 1], c='y', s=30, lw=0.1, zorder=4)
        ax.scatter(mu_noi_[:, 0], mu_noi_[:, 1], c='y', s=30, lw=0.1, zorder=4)
        filename = "sanity-ellipse-projection-init.png"
        path = os.path.join(plot_path, filename)
        fig.savefig(path)
        plt.close(fig)
    
    
    
    ##### SANITY PLOT (QUADRIC APPARENT CONTOUR) ###############################
    
    io.print_and_log(["# Sanity plot (quadric apparent contour)..."], level='info', logger=params)
    
    
    v1 = pca.components_[0, :]
    v2 = pca.components_[1, :]
    
    
    if verbose:
        # msg = [
        #     "# norm(v1)",
        #     "%s" %(numpy.linalg.norm(v1),),
        #     "# norm(v2)",
        #     "%s" %(numpy.linalg.norm(v2),),
        # ]
        # io.print_and_log(msg, level='default', logger=params)
        pass
    
    
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
        io.print_and_log(msg, level='default', logger=params)
        pass
    
    
    R_ = R.T
    t_ = pca.mean_
    A_ = numpy.dot(numpy.dot(R_.T, A_init), R_)
    b_ = numpy.dot(R_.T, 2.0 * numpy.dot(A_init, t_) + b_init)
    c_ = numpy.dot(numpy.dot(A_init, t_) + b_init, t_) + c_init
    
    
    if verbose:
        msg = [
            "# t_",
            "%s" %(t_,),
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    xs = [numpy.array([0.0, 0.0]),
          numpy.array([1.0, 0.0]),
          numpy.array([0.0, 1.0])]
    
    # Solve the linear system 2 * A.T * y + b = 0 for fixed couples (y_1, y_2).
    ys = []
    for x in xs:
        c1 = 2.0 * A_[2:, 2:].T
        c2 = - (numpy.dot(2.0 * A_[:2, 2:].T, x) + b_[2:])
        yx = numpy.linalg.solve(c1, c2)
        ys.append(yx)
    
    
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
    
    A__ = numpy.dot(alpha.T, numpy.dot(A_, alpha))
    b__ = numpy.dot(alpha.T, 2.0 * numpy.dot(A_, beta) + b_)
    c__ = numpy.dot(numpy.dot(A_, beta) + b_, beta) + c_
    
    
    if verbose:
        msg = [
            "# A__",
            "%s" %(A__,),
            "# b__",
            "%s" %(b__,),
            "# c__",
            "%s" %(c__,),
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    if make_plots:
        # Plot conic section.
        n = 100
        # x_min = -3100.0
        # x_max = +3100.0
        # y_min = -3100.0
        # y_max = +3100.0
        x_r = numpy.linspace(x_min, x_max, n)
        y_r = numpy.linspace(y_min, y_max, n)
        xx, yy = numpy.meshgrid(x_r, y_r)
        zz = numpy.zeros(xx.shape)
        for i in xrange(0, xx.shape[0]):
            for j in xrange(0, xx.shape[1]):
                v = numpy.array([xx[i, j], yy[i, j]])
                zz[i, j] = numpy.dot(numpy.dot(v, A__), v) + numpy.dot(b__, v) + c__
        vv = numpy.array([0.0])
        # vv = numpy.arange(0.0, 1.0, 0.1)
        # vv = numpy.arange(0.0, 20.0)
        
        fig = plt.figure()
        ax = fig.gca()
        ax.set_xlabel("1st component")
        ax.set_ylabel("2nd component")
        ax.set_aspect('equal')
        ax.grid()
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.contour(xx, yy, zz, vv, colors='y', linewidths=1.0)
        # cs = ax.contour(xx, yy, zz, vv, colors='k', linewidths=1.0)
        # ax.clabel(cs, inline=1, fontsize=10)
        ax.scatter(pca.transform(X_gt)[:, 0], pca.transform(X_gt)[:, 1], c='g', s=5, lw=0.1)
        filename = "contour-init.png"
        path = os.path.join(plot_path, filename)
        fig.savefig(path)
        plt.close(fig)
    
    
    
    ############################################################################
    
    # Compute the Mahalanobis distance.
    def evaluate_ellipse(A, b, c, X):
        x2 = numpy.sum(numpy.multiply(X.T, numpy.dot(A, X.T)), axis=0)
        x1 = numpy.dot(b, X.T)
        x0 = c
        d2 = x2 + x1 + x0
        return d2
    
    def Mahalanobis_distance(A, mu, X):
        N = X.shape[0]
        d2 = numpy.zeros(N)
        for i in xrange(0, N):
            d2[i] = numpy.dot(X[i, :] - mu, numpy.dot(A, X[i, :] - mu))
        return d2
    
    mu = numpy.mean(X_gt, axis=0)
    
    Mhlnb_gt = Mahalanobis_distance(A_init, mu, X_gt)
    Mhlnb_ngt = Mahalanobis_distance(A_init, mu, X_ngt)
    Mhlnb_noi = Mahalanobis_distance(A_init, mu, X_noi)
    
    Ell_gt = evaluate_ellipse(A_init, b_init, c_init, X_gt)
    
    
    if make_plots:
        fig = plt.figure()
        ax = fig.gca()
        ax.grid()
        ax.hist(Ell_gt, bins=75, color='g')
        filename = "ellipse-values.png"
        path = os.path.join(plot_path, filename)
        fig.savefig(path)
        plt.close(fig)
    
    
    if make_plots:
        fig = plt.figure()
        ax = fig.gca()
        ax.grid()
        ax.hist(Mhlnb_noi, bins=50, color='r')
        ax.hist(Mhlnb_ngt, bins=50, color='b')
        ax.hist(Mhlnb_gt, bins=75, color='g')
        filename = "mahalanobis-init.png"
        path = os.path.join(plot_path, filename)
        fig.savefig(path)
        plt.close(fig)
    
    
    
    ##### LEARNING #############################################################
    
    io.print_and_log(["# Learning..."], level='info', logger=params)
    
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    
    
    # mode = 'decision'
    mode = 'prediction'
    
    x_component = 0
    y_component = 1
    
    # Standardize features by removing the mean and scaling to unit variance.
    # X = StandardScaler().fit_transform(X)
    
    pad = 0.5 # padding coefficient
    x_dif = numpy.amax(X[:, x_component]) - numpy.amin(X[:, x_component])
    x_min = numpy.amin(X[:, x_component]) - pad * x_dif
    x_max = numpy.amax(X[:, x_component]) + pad * x_dif
    y_dif = numpy.amax(X[:, y_component]) - numpy.amin(X[:, y_component])
    y_min = numpy.amin(X[:, y_component]) - pad * y_dif
    y_max = numpy.amax(X[:, y_component]) + pad * y_dif
    h = max(x_dif, y_dif) / 100.0 # step size in the mesh
    x_r = numpy.arange(x_min, x_max, h)
    y_r = numpy.arange(y_min, y_max, h)
    xx, yy = numpy.meshgrid(x_r, y_r)
        
    # Preprocess dataset.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    
    if verbose:
        msg = [
            "# X_train.shape",
            "%s" %(X_train.shape,),
            "# X_test.shape",
            "%s" %(X_test.shape,),
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    # Declare model.
    clf = MLPClassifier(hidden_layer_sizes=(),
                        activation='logistic',
                        algorithm='sgd',
                        alpha=1.0e-12,
                        tol=1.0e-8,
                        learning_rate='adaptive',
                        random_state=0,
                        momentum=0.05,
                        nesterovs_momentum=False)
    
    # Initialize model (i.e. fake launch + weights initialization).
    clf.set_params(max_iter=1)
    clf.set_params(learning_rate_init=sys.float_info.epsilon)
    clf.set_params(warm_start=False)
    clf.fit(X_train, y_train)
    
    
    if make_plots:
        fig = plt.figure()
        ax = fig.gca()
        ax.grid()
        c_raw = clf.predict(X) 
        # c_raw = clf.predict_proba(X)[:, 0]
        vmax = 1.0
        vmin = 0.0
        # c_raw = clf.decision_function(X)
        # vmax = max(abs(numpy.amin(c_raw)), abs(numpy.amax(c_raw)))
        # vmin = - vmax
        x_raw = pca.transform(X_raw)[:, 0]
        y_raw = pca.transform(X_raw)[:, 1]
        # cs = ax.scatter(x_raw, y_raw, c=c_raw, s=5, lw=0.1, cmap='bwr', vmin=vmin, vmax=vmax)
        # fig.colorbar(cs)
        ax.scatter(x_raw[0.5 < c_raw], y_raw[0.5 < c_raw], c='r', s=5, lw=0.1)
        ax.scatter(x_raw[c_raw < 0.5], y_raw[c_raw < 0.5], c='g', s=5, lw=0.1)
        filename = "temp-1.png"
        path = os.path.join(plot_path, filename)
        fig.savefig(path)
        plt.close(fig)
    
    
    if verbose:
        msg = [
            # Print the current loss.
            "# clf.loss_",
            "%s" %(clf.loss_,),
            # Print the loss curve.
            "# clf.loss_curve_",
            "%s" %(clf.loss_curve_,),
            # Print the number of iterations the algorithm has ran.
            "# clf.n_iter_",
            "%s" %(clf.n_iter_,),
            # Print the score on the test set.
            "# clf.score(X_test, y_test)",
            "%s" %(clf.score(X_test, y_test),),
            "%s" %(1.0 - clf.score(X_test, y_test),),
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    coefs_init = ellipsoid_matrix_to_coefs(A_init, b_init, c_init)
    clf.coefs_ = [coefs_init[1:, :]]
    clf.intercepts_ = [coefs_init[:1, :]]
    
    
    if make_plots:
        fig = plt.figure()
        ax = fig.gca()
        ax.grid()
        c_raw = clf.predict(X)
        # c_raw = clf.predict_proba(X)[:, 0]
        vmax = 1.0
        vmin = 0.0
        # c_raw = clf.decision_function(X)
        # vmax = max(abs(numpy.amin(c_raw)), abs(numpy.amax(c_raw)))
        # vmin = - vmax
        x_raw = pca.transform(X_raw)[:, 0]
        y_raw = pca.transform(X_raw)[:, 1]
        # cs = ax.scatter(x_raw, y_raw, c=c_raw, s=5, lw=0.1, cmap='bwr', vmin=vmin, vmax=vmax)
        # fig.colorbar(cs)
        ax.scatter(x_raw[0.5 < c_raw], y_raw[0.5 < c_raw], c='r', s=5, lw=0.1)
        ax.scatter(x_raw[c_raw < 0.5], y_raw[c_raw < 0.5], c='g', s=5, lw=0.1)
        filename = "temp-2.png"
        path = os.path.join(plot_path, filename)
        fig.savefig(path)
        plt.close(fig)
    
    
    if verbose:
        msg = [
            # # Print the current loss.
            # "# clf.loss_",
            # "%s" %(clf.loss_,),
            # Print the loss curve.
            "# clf.loss_curve_",
            "%s" %(clf.loss_curve_,),
            # # Print the number of iterations the algorithm has ran.
            # "# clf.n_iter_",
            # "%s" %(clf.n_iter_,),
            # Print the score on the test set.
            "# clf.score(X_test, y_test)",
            "%s" %(clf.score(X_test, y_test),),
            "%s" %(1.0 - clf.score(X_test, y_test),)
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    # Train model.
    clf.set_params(max_iter=max_iter)
    clf.set_params(learning_rate_init=learning_rate_init)
    clf.set_params(warm_start=True)
    clf.fit(X_train, y_train)
    
    
    if make_plots:
        fig = plt.figure()
        ax = fig.gca()
        ax.grid()
        c_raw = clf.predict(X)
        vmax = 1.0
        vmin = 0.0
        # c_raw = clf.predict_proba(X)[:, 0]
        # c_raw = clf.decision_function(X)
        # vmax = max(abs(numpy.amin(c_raw)), abs(numpy.amax(c_raw)))
        # vmin = - vmax
        x_raw = pca.transform(X_raw)[:, 0]
        y_raw = pca.transform(X_raw)[:, 1]
        # cs = ax.scatter(x_raw, y_raw, c=c_raw, s=5, lw=0.1, cmap='bwr', vmin=-vmax, vmax=vmax)
        # fig.colorbar(cs)
        ax.scatter(x_raw[0.5 < c_raw], y_raw[0.5 < c_raw], c='r', s=5, lw=0.1)
        ax.scatter(x_raw[c_raw < 0.5], y_raw[c_raw < 0.5], c='g', s=5, lw=0.1)
        filename = "temp-3.png"
        path = os.path.join(plot_path, filename)
        fig.savefig(path)
        plt.close(fig)
    
    
    if verbose:
        msg = [
            # Print the current loss computed with the loss function.
            "# clf.loss_",
            "%s" %(clf.loss_,),
            # Print the loss curve.
            "# clf.loss_curve_",
            "%s" %(clf.loss_curve_,),
            # Print the number of iterations the algorithm has ran.
            "# clf.n_iter_",
            "%s" %(clf.n_iter_,),
            # Print the score on the test set.
            "# clf.score(X_test, y_test)",
            "%s" %(clf.score(X_test, y_test),),
            "%s" %(1.0 - clf.score(X_test, y_test),),
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    if make_plots:
        # Plot the loss curve.
        fig = plt.figure()
        ax = fig.gca()
        ax.grid(True, which='both')
        ax.set_title("Loss curve")
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        x_min = 1
        x_max = len(clf.loss_curve_) - 1
        ax.set_xlim([x_min - 1, x_max + 1])
        # ax.set_ylim([0.0, 1.1 * numpy.amax(clf.loss_curve_[1:])])
        # ax.plot(range(x_min, x_max + 1), clf.loss_curve_[1:], 'b-')
        # ax.plot(range(x_min, x_max + 1), clf.loss_curve_[1:], 'bo')
        ax.semilogy(range(x_min, x_max + 1), clf.loss_curve_[1:], 'b-')
        # ax.semilogy(range(x_min, x_max + 1), clf.loss_curve_[1:], 'bo')
        filename = "loss-curve.png"
        path = os.path.join(plot_path, filename)
        fig.savefig(path)
        plt.close(fig)
    
    
    # Retrieve the coefficients of the ellipsoid.
    bias = clf.intercepts_[0].flatten() # bias vector
    weights = clf.coefs_[0].flatten() # weight vector
    # Concatenate the coefficients.
    coefs = numpy.concatenate((bias, weights))
    coefs = coefs.reshape(-1, 1)
    
    
    A, b, c = ellipsoid_coefs_to_matrix(coefs)
    
    
    if verbose:
        msg = [
            "# det(A)",
            "%s" %(numpy.linalg.det(A),),
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    
    # SANITY PLOT (ELLIPSE PROJECTION) #########################################
    
    io.print_and_log(["# Sanity plot (ellipse projection)..."], level='info', logger=params)
    
    from sklearn.decomposition import PCA
    
    
    # Compute PCA with two components.
    n_components = 2
    pca = PCA(n_components)
    _ = pca.fit(X_raw)
    
    # Data transformation.
    X_raw_ = pca.transform(X_raw)
    X_gt_ = pca.transform(X_gt)
    X_ngt_ = pca.transform(X_ngt)
    X_noi_ = pca.transform(X_noi)
    
    # Mean transformation.
    mu_gt = numpy.mean(X_gt, axis=0).reshape(1, -1)
    mu_gt_ = pca.transform(mu_gt)
    mu_ngt = numpy.mean(X_ngt, axis=0).reshape(1, -1)
    mu_ngt_ = pca.transform(mu_ngt)
    mu_noi = numpy.mean(X_noi, axis=0).reshape(1, -1)
    mu_noi_ = pca.transform(mu_noi)
    
    # Ellipse transformation.
    f = 0.25 * numpy.dot(numpy.dot(b, numpy.linalg.inv(A)), b) - c
    t = - 0.5 * numpy.dot(numpy.linalg.inv(A), b).reshape(1, -1)
    s, O = numpy.linalg.eigh(numpy.linalg.inv((1.0 / f) * A))
    ##### TODO: remove test zone.
    s = numpy.abs(s)
    ##### end test zone
    s = numpy.sqrt(s)
    t_ = pca.transform(t)
    O_ = pca.transform(numpy.multiply(s, O).T + t)
    
    
    if verbose:
        msg = [
            "# t",
            "%s" %(t,),
            "# s (i.e. demi-axes)",
            "%s" %(s,),
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    if make_plots:
        # Plot ellipse projection.
        ## Find plot limits.
        pad = 0.3
        x_dif = numpy.amax(X_raw_[:, 0]) - numpy.amin(X_raw_[:, 0])
        x_min = numpy.amin(X_raw_[:, 0]) - pad * x_dif
        x_max = numpy.amax(X_raw_[:, 0]) + pad * x_dif
        y_dif = numpy.amax(X_raw_[:, 1]) - numpy.amin(X_raw_[:, 1])
        y_min = numpy.amin(X_raw_[:, 1]) - pad * y_dif
        y_max = numpy.amax(X_raw_[:, 1]) + pad * y_dif
        ## Create plot.
        fig = plt.figure()
        ax = fig.gca()
        ax.set_xlabel("1st component")
        ax.set_ylabel("2nd component")
        ax.grid()
        ax.set_aspect('equal')
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ## Plot datasets.
        ax.scatter(X_gt_[:, 0], X_gt_[:, 1], c='g', s=5, lw=0.1)
        ax.scatter(X_ngt_[:, 0], X_ngt_[:, 1], c='b', s=5, lw=0.1)
        ax.scatter(X_noi_[:, 0], X_noi_[:, 1], c='r', s=5, lw=0.1)
        ## Plot ellipse transformation.
        for i in xrange(0, O_.shape[0]):
            ax.plot([t_[0, 0], O_[i, 0]], [t_[0, 1], O_[i, 1]], 'y', zorder=3)
        ## Plot means of datasets.
        ax.scatter(mu_gt_[:, 0], mu_gt_[:, 1], c='y', s=30, lw=0.1, zorder=4)
        ax.scatter(mu_ngt_[:, 0], mu_ngt_[:, 1], c='y', s=30, lw=0.1, zorder=4)
        ax.scatter(mu_noi_[:, 0], mu_noi_[:, 1], c='y', s=30, lw=0.1, zorder=4)
        ## Save plot.
        filename = "sanity-ellipse-projection.png"
        path = os.path.join(plot_path, filename)
        fig.savefig(path)
        plt.close(fig)
    
    
    
    ##### SANITY PLOT (QUADRIC APPARENT CONTOUR) ###############################
    
    io.print_and_log(["# Sanity plot (quadric apparent contour)..."], level='info', logger=params)
    
    
    v1 = pca.components_[0, :]
    v2 = pca.components_[1, :]
    
    
    if verbose:
        # msg = [
        #     "# norm(v1)",
        #     "%s" %(numpy.linalg.norm(v1),),
        #     "# norm(v2)",
        #     "%s" %(numpy.linalg.norm(v2),),
        # ]
        # io.print_and_log(msg, level='default', logger=params)
        pass
    
    
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
        u1 = numpy.dot(R, v1)
        u1[numpy.abs(u1) < 1.0e-10] = 0.0
        u2 = numpy.dot(R, v2)
        u2[numpy.abs(u2) < 1.0e-10] = 0.0
        msg = [
            "# R * v1",
            "%s" %(u1,),
            "# R * v2",
            "%s" %(u2,),
        ]
        io.print_and_log(msg, level='default', logger=params)
        pass
    
    
    R_ = R.T
    t_ = pca.mean_
    A_ = numpy.dot(numpy.dot(R_.T, A), R_)
    b_ = numpy.dot(R_.T, 2.0 * numpy.dot(A, t_) + b)
    c_ = numpy.dot(numpy.dot(A, t_) + b, t_) + c
    
    
    if verbose:
        msg = [
            "# t_",
            "%s" %(t_,),
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    xs = [numpy.array([0.0, 0.0]),
          numpy.array([1.0, 0.0]),
          numpy.array([0.0, 1.0])]
    
    # Solve the linear system 2 * A.T * y + b = 0 for fixed couples (y_1, y_2).
    ys = []
    for x in xs:
        c1 = 2.0 * A_[2:, 2:].T
        c2 = - (numpy.dot(2.0 * A_[:2, 2:].T, x) + b_[2:])
        yx = numpy.linalg.solve(c1, c2)
        ys.append(yx)
    
    
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
    
    A__ = numpy.dot(alpha.T, numpy.dot(A_, alpha))
    b__ = numpy.dot(alpha.T, 2.0 * numpy.dot(A_, beta) + b_)
    c__ = numpy.dot(numpy.dot(A_, beta) + b_, beta) + c_
    
    
    if verbose:
        msg = [
            "# A__",
            "%s" %(A__,),
            "# b__",
            "%s" %(b__,),
            "# c__",
            "%s" %(c__,),
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    if make_plots:
        # Plot conic section.
        n = 100
        # x_min = -300.0
        # x_max = +300.0
        # y_min = -300.0
        # y_max = +300.0
        x_r = numpy.linspace(x_min, x_max, n)
        y_r = numpy.linspace(y_min, y_max, n)
        xx, yy = numpy.meshgrid(x_r, y_r)
        zz = numpy.zeros(xx.shape)
        for i in xrange(0, xx.shape[0]):
            for j in xrange(0, xx.shape[1]):
                v = numpy.array([xx[i, j], yy[i, j]])
                zz[i, j] = numpy.dot(numpy.dot(v, A__), v) + numpy.dot(b__, v) + c__
        vv = numpy.array([0.0])
        # vv = numpy.arange(0.0, 1.0, 0.1)
        # vv = numpy.arange(0.0, 20.0)
        
        fig = plt.figure()
        ax = fig.gca()
        ax.set_xlabel("1st component")
        ax.set_ylabel("2nd component")
        ax.set_aspect('equal')
        ax.grid()
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.contour(xx, yy, zz, vv, colors='y', linewidths=1.0)
        # cs = ax.contour(xx, yy, zz, vv, colors='y', linewidths=1.0)
        # cs = ax.contour(xx, yy, zz, colors='y', linewidths=1.0)
        # ax.clabel(cs, inline=1, fontsize=10)
        # ax.scatter(pca.transform(X_gt)[:, 0], pca.transform(X_gt)[:, 1], c='g', s=5, lw=0.1)
        c_raw = clf.predict(X)[0:X_raw.shape[0]]
        ax.scatter(pca.transform(X_raw)[:, 0], pca.transform(X_raw)[:, 1], c=c_raw, s=5, lw=0.1)
        ## Save plot.
        filename = "contour.png"
        path = os.path.join(plot_path, filename)
        fig.savefig(path)
        plt.close(fig)
    
    
    
    ############################################################################
    
    # Compute the Mahalanobis distance.
    def squared_Mahalanobis_distance(A, mu, X):
        N = X.shape[0]
        d = numpy.zeros(N)
        for i in xrange(0, N):
            d[i] = numpy.dot(X[i, :] - mu, numpy.dot(A, X[i, :] - mu))
        return d
    
    mu = numpy.mean(X_gt, axis=0)
    Mhlnb_gt = squared_Mahalanobis_distance(A, mu, X_gt)
    Mhlnb_ngt = squared_Mahalanobis_distance(A, mu, X_ngt)
    Mhlnb_noi = squared_Mahalanobis_distance(A, mu, X_noi)
    
    
    if verbose:
        msg = [
            "# Mhlnb_gt",
            "%s" %(Mhlnb_gt,),
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    if make_plots:
        fig = plt.figure()
        ax = fig.gca()
        ax.grid()
        ax.hist(Mhlnb_ngt, bins=50, color='b')
        ax.hist(Mhlnb_noi, bins=50, color='r')
        ax.hist(Mhlnb_gt, bins=75, color='g')
        filename = "mahalanobis.png"
        path = os.path.join(plot_path, filename)
        fig.savefig(path)
        plt.close(fig)
    
    
    
    ############################################################################
    
    # Compute prediction on a grid of the input space for plotting.
    shape_pre = (xx.shape[0] * xx.shape[1], X_train.shape[1])
    print("# Coefficients ellipse")
    X_pre = numpy.zeros(shape_pre)
    print("# 1")
    print(coefs[0])
    X_pre[:, x_component] = xx.ravel()
    print("# x")
    print(coefs[1 + x_component])
    X_pre[:, y_component] = yy.ravel()
    print("# y")
    print(coefs[1 + y_component])
    if pairwise:
        k = 0
        for i in xrange(0, N):
            for j in xrange(i, N):
                if i == x_component and j == x_component:
                    X_pre[:, N + k] = numpy.multiply(xx.ravel(), xx.ravel())
                    print("# x * x")
                    print(coefs[1 + N + k])
                elif i == x_component and j == y_component:
                    X_pre[:, N + k] = numpy.multiply(xx.ravel(), yy.ravel())
                    print("# x * y")
                    print(coefs[1 + N + k])
                elif i == y_component and j == y_component:
                    X_pre[:, N + k] = numpy.multiply(yy.ravel(), yy.ravel())
                    print("# y * y")
                    print(coefs[1 + N + k])
                else:
                    pass
                k = k + 1
    if mode is 'decision':
        zz = clf.decision_function(X_pre)
    elif mode is 'prediction':
        zz = clf.predict(X_pre)
    else:
        raise(Exception)
    zz = zz.reshape(xx.shape)
    
    
    
    ##### SANITY PLOT ##########################################################
    
    io.print_and_log(["# Sanity plot..."], level='info', logger=params)
    
    
    if make_plots:
        fig = plt.figure()
        fig.suptitle("Dataset and decision boundaries")
        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0])
        ax.hold(True)
        if mode is 'decision':
            vlim = max(1.0, max(abs(numpy.amin(zz)), numpy.amax(zz)))
            vmin = -vlim
            vmax= vlim
        elif mode is 'prediction':
            vmin = 0.0
            vmax = 1.0
        else:
            raise(Exception)
        cs = ax.contourf(xx, yy, zz, 20, alpha=0.8, cmap='bwr', vmin=vmin, vmax=vmax)
##### TODO: remove test zone
        # Comment scatter to see the prediction boundary only.
        ax.scatter(X_test[:, x_component], X_test[:, y_component], c=y_test, cmap='bwr', alpha=0.6)
        ax.scatter(X_train[:, x_component], X_train[:, y_component], c=y_train, cmap='bwr')
##### end test zone
        ax.hold(False)
        fig.colorbar(cs)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel("%dst principal component" %(x_component + 1))
        ax.set_ylabel("%dnd principal component" %(y_component + 1))
        ax.grid()
        filename = "decision-boundaries-%d-%d.png" %(x_component, y_component)
        path = os.path.join(plot_path, filename)
        plt.savefig(path)
        fig.clear()
    
    
    ##### SANITY PLOT (PCA) ####################################################
    
    io.print_and_log(["# Sanity plot (PCA)..."], level='info', logger=params)
    
    from sklearn.decomposition import PCA
    
    
    # Compute PCA with two components.
    n_components = 2
    pca = PCA(n_components)
    _ = pca.fit(X_raw)
    X_raw_r = pca.transform(X_raw)
    
    # Find plot limits.
    pad = 0.1
    x_dif = numpy.amax(X_raw_r[:, 0]) - numpy.amin(X_raw_r[:, 0])
    x_min = numpy.amin(X_raw_r[:, 0]) - pad * x_dif
    x_max = numpy.amax(X_raw_r[:, 0]) + pad * x_dif
    y_dif = numpy.amax(X_raw_r[:, 1]) - numpy.amin(X_raw_r[:, 1])
    y_min = numpy.amin(X_raw_r[:, 1]) - pad * y_dif
    y_max = numpy.amax(X_raw_r[:, 1]) + pad * y_dif
    
##### TODO: remove temporary zone.
    # Set plot limits to zoom in.
    x_min = -800.0
    x_max = +600.0
    y_min = -750.0
    y_max = +750.0
##### end temporary zone
    

    # TODO: compute the projection of the ellipsoid on Vect(pca1, pca2).
    # Retrieve pca1 and pca2.
    vpca = pca.components_
    
    
    if verbose:
        msg = [
            "# Shapes of the components",
            "%s" %(vpca.shape,),
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    v = numpy.array([[1.0, 0.0], [0.0, 1.0]])
    vpca = pca.inverse_transform(v)
    
    if verbose:
        msg = [
            "# Shapes after inverse transform of v (i.e. vpca)",
            "%s" %(vpca.shape,),
            "# Shapes X_raw",
            "%s" %(X_raw.shape,),
            "# Norms of vpca0 and vpca1",
            "%s" %(numpy.linalg.norm(vpca, axis=1),),
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    # Retrieve the coefficients of the ellipsoid.
    weights = clf.coefs_[0].flatten() # weight vector
    bias = clf.intercepts_[0].flatten() # bias vector
    # Concatenate the coefficients.
    coefs = numpy.concatenate((bias, weights))
    
    
    if verbose:
        msg = [
            "# Weights",
            "%s" %(weights,),
            "%s" %(type(weights),),
            "%s" %(weights.shape,),
            "# Bias",
            "%s" %(bias,),
            "%s" %(type(bias),),
            "%s" %(bias.shape,),
            "# Coefs",
            "%s" %(coefs,),
            "%s" %(type(coefs),),
            "%s" %(coefs.shape,),
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    # Check if ellipsoid.
    # TODO: complete (i.e. check if det(A) > 0 which is the criterion for ellipse).
        
    # Useful function to convert an ellispoid in general form to an ellispoid in
    # standard form.
    def ellipsoid_general_to_standard(coefs):
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
            io.print_and_log(msg, level='default', logger=params)
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
            io.print_and_log(msg, level='default', logger=params)
        ##### end test zone
        ##### TODO: remove plot zone
        if make_plots:
            fig = plt.figure()
            ax = fig.gca()
            cax = ax.imshow(A, interpolation='nearest', cmap='jet')
            fig.colorbar(cax)
            filename = "ellipse.png"
            path = os.path.join(plot_path, filename)
            fig.savefig(path)
            plt.close(fig)
        ##### end plot zone
        
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
            io.print_and_log(msg, level='default', logger=params)
        ##### end print zone.
        
        # Semi-axes from reduced canonical equation.
        ##### TODO: remove test zone.
        # eaxis = numpy.sqrt(- c / evals)
        eaxis = numpy.sqrt(numpy.abs(-c / evals))
        ##### end test zone
        return center, eaxis, evecs
    
    
##### TODO: remove test zone (standard -> genral -> standard)
    # if verbose:
    #     print("")

    # # Test.
    # t = numpy.array([1.0, 2.0])
    # s = numpy.array([0.5, 0.2])
    # O = numpy.array([[1.0, 0.0], [0.0, 1.0]])
    
    # if verbose:
    #     print("# t")
    #     print(t)
    #     print("# s")
    #     print(s)
    #     print("# O")
    #     print(O)
    
    # coefs_bis = ellipsoid_standard_to_general(t, s, O)
    
    # if verbose:
    #     print("# coefs_bis")
    #     print(coefs_bis)
    
    # t_bis, s_bis, O_bis = ellipsoid_general_to_standard(coefs_bis)
    
    # if verbose:
    #     print("# t_bis")
    #     print(t_bis)
    #     print("# s_bis")
    #     print(s_bis)
    #     print("# O_bis")
    #     print(O_bis)
    
    # if verbose:
    #     print("")
    
    # import sys
    # sys.exit(0)
##### end test zone
    
    center, eaxis, evecs = ellipsoid_general_to_standard(coefs)
    
    
    if verbose:
        msg = [
            "# Conversion",
            "# Center",
            "%s" %(center,),
            "%s" %(center.shape,),
            "# Eigenaxis",
            "%s" %(eaxis,),
            "%s" %(eaxis.shape,),
            "# Eigenvectors",
            "%s" %(evecs,),
            "%s" %(evecs.shape,),
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    coefs_bis = ellipsoid_standard_to_general(center, eaxis, evecs)
    
    
    if verbose:
        msg = [
            "# Transform and untransfrom",
            "# coefs",
            "%s" %(coefs,),
            "# coefs_bis",
            "%s" %(coefs_bis,),
        ]
        io.print_and_log(msg, level='default', logger=params)
    
    
    # TODO: compute the projection of the eigenvectors on Vect(vpca[0, :], vpca[1, :]).
    # Projection of the center.
    shape = (1, 2)
    cprojs = pca.transform(center.reshape(1, -1))
    # Projection of the eigenvectors.
    shape = (evecs.shape[1], 2)
    eprojs = numpy.zeros(shape)
    for j in xrange(evecs.shape[1]):
        # eprojs[j, :] = pca.transform(eaxis[j] * evecs[:, j])
        vec = numpy.add(pca.mean_, 300.0 * evecs[:, j])
        vec = vec.reshape(1, -1)
        eprojs[j, :] = pca.transform(vec)
        #eprojs[j, :] = pca.transform(pca.mean_ + evecs[:, j])
        pass
        # import math
        # if math.isnan(eaxis[j]):
        #     print("Pass")
        #     vec = numpy.add(pca.mean_, evecs[:, j])
        #     eprojs[j, :] = pca.transform(vec)
        # else:
        #     # print("#####")
        #     # print(pca.mean_)
        #     # print(eaxis[j])
        #     # print(evecs[:, j])
        #     vec = numpy.add(pca.mean_, eaxis[j] * evecs[:, j])
        #     eprojs[j, :] = pca.transform(vec)
        
        # TODO: remove incorrect projection on the PCA space.
        # eprojs[j, 0] = numpy.dot(eaxis[j] * evecs[:, j], vpca[0, :])
        # eprojs[j, 1] = numpy.dot(eaxis[j] * evecs[:, j], vpca[1, :])
    
    
    if verbose:
        msg = [
            "# Center projection",
            "%s" %(cprojs,),
            "%s" %(cprojs.shape,),
            # "# Eigenprojections",
            # "%s" %(eprojs,),
            # "%s" %(eprojs.shape,),
        ]
        io.print_and_log(msg, level='default', logger=params)

    
    
    
##### TODO: remove plot zone.
    if make_plots:
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(vpca[0, :])
        filename = "plot0.png"
        path = os.path.join(plot_path, filename)
        fig.savefig(path)
        plt.close(fig)
    
    if make_plots:
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(vpca[1, :])
        filename = "plot1.png"
        path = os.path.join(plot_path, filename)
        fig.savefig(path)
        plt.close(fig)
    
    if make_plots:
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(vpca[1, :] - vpca[0, :])
        filename = "plot2.png"
        path = os.path.join(plot_path, filename)
        fig.savefig(path)
        plt.close(fig)
##### end plot zone

    
##### TODO: remove experimental zone.
    # h = 0.1 # step size in the mesh
    # xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h),
    #                         numpy.arange(y_min, y_max, h))
    # shape_pre = (xx.shape[0] * xx.shape[1], X_train.shape[1])
    # X_pre = numpy.zeros(shape_pre)
    # X_pre[:, x_component] = xx.ravel()
    # X_pre[:, y_component] = yy.ravel()
    # if pairwise:
    #     k = 0
    #     for i in xrange(0, N):
    #         for j in xrange(i, N):
    #             if i == x_component and j == y_component:
    #                 X_pre[:, N + k] = numpy.multiply(xx.ravel(), yy.ravel())
    #             else:
    #                 pass
    #             k = k + 1
    
    # if mode is 'decision':
    #     zz = clf.decision_function(X_pre)
    # elif mode is 'prediction':
    #     zz = clf.predict(X_pre)
    # else:
    #     raise(Exception)
    # zz = zz.reshape(xx.shape)
##### end experimental zone
    
    
    # SANITY PLOTS (REDUCED DATASETS) ##########################################
    
    io.print_and_log(["# Sanity plots (reduced datasets)..."], level='info', logger=params)
    
    
    if make_plots:
        # Plot reduced dataset.
        fig = plt.figure()
        ax = fig.gca()
        ax.grid()
        ax.scatter(X_raw_r[:, 0], X_raw_r[:, 1], c=y, cmap='bwr', zorder=1)
        # Plot the projection of the ellipsoid.
        ax.scatter(cprojs[0, 0], cprojs[0, 1], c='y', s=100, zorder=3)
        for j in xrange(0, eprojs.shape[0]):
            xp = cprojs[0, 0] + [0.0, eprojs[j, 0]]
            yp = cprojs[0, 1] + [0.0, eprojs[j, 1]]
            ax.plot(xp, yp, 'y-', zorder=2)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_title("Plot PCA-reduced dataset")
        ax.set_xlabel("1st component")
        ax.set_ylabel("2nd component")
        filename = "reduced-dataset.png"
        path = os.path.join(plot_path, filename)
        fig.savefig(path)
        plt.close(fig)
    
    if make_plots:
        # Plot reduced datset restricted to the ground truth cell.
        fig = plt.figure()
        ax = fig.gca()
        ax.grid()
        ax.scatter(X_raw_r[y == 1, 0], X_raw_r[y == 1, 1], c='r', zorder=1)
        # Plot the projection of the ellipsoid.
        ax.scatter(cprojs[0, 0], cprojs[0, 1], c='y', s=100, zorder=3)
        for j in xrange(0, eprojs.shape[0]):
            xp = cprojs[0, 0] + [0.0, eprojs[j, 0]]
            yp = cprojs[0, 1] + [0.0, eprojs[j, 1]]
            ax.plot(xp, yp, 'y-', zorder=2)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_title("Plot PCA-reduced dataset restricted to the ground truth cell")
        ax.set_xlabel("1st component")
        ax.set_ylabel("2nd component")
        filename = "reduced-dataset-true.png"
        path = os.path.join(plot_path, filename)
        fig.savefig(path)
        plt.close(fig)
    
    if make_plots:
        # Plot reduced datset restricted non ground truth cells and noise.
        fig = plt.figure()
        ax = fig.gca()
        ax.grid()
        ax.scatter(X_raw_r[y == 0, 0], X_raw_r[y == 0, 1], c='b', zorder=1)
        # Plot the projection of the ellipsoid.
        ax.scatter(cprojs[0, 0], cprojs[0, 1], c='y', s=100, zorder=3)
        for j in xrange(0, eprojs.shape[0]):
            xp = cprojs[0, 0] + [0.0, eprojs[j, 0]]
            yp = cprojs[0, 1] + [0.0, eprojs[j, 1]]
            ax.plot(xp, yp, 'y-', zorder=2)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_title("Plot PCA-reduced dataset restricted to the non ground truth cells and noise")
        ax.set_xlabel("1st component")
        ax.set_ylabel("2nd component")
        filename = "/tmp/reduced-dataset-false.png"
        path = os.path.join(plot_path, filename)
        fig.savefig(path)
        plt.close(fig)
    
    
    
    io.print_and_log(["Validation done."], level='info', logger=params)
    
    return
