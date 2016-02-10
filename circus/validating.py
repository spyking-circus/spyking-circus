import numpy

from .shared.utils import io


def main(filename, params, nb_cpu, nb_gpu, use_gpu):
    
    print("Validating...")
    
    
##### TODO: select a subset of electrodes if necessary. (how ?)
    
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
    
    # Initialize the random seed.
    numpy.random.seed(0)
    
    # Retrieve PCA basis.
    basis_proj, basis_rec = io.load_data(params, 'basis')
    # Retrieve sampling rate.
    sampling_rate  = params.getint('data', 'sampling_rate')
    
    
    ##### GROUND TRUTH CELL'S SAMPLES ##########################################
    
    # Retrieve the spikes of the "ground truth cell".
    spike_times_gt, spikes_gt = io.load_data(params, 'triggers')
    
    # Select only the neighboring electrodes of the best electrode.
    indices = get_neighbors(params)
    spikes_gt = spikes_gt[:, indices, :]
    
    # Reshape data.
    shape_gt = spikes_gt.shape
    spikes_gt = spikes_gt.reshape(shape_gt[0], shape_gt[1] * shape_gt[2])
    spikes_gt = spikes_gt.T
    
    # Compute the PCA coordinates of each spike of the "ground truth cell".
    X_gt = numpy.dot(spikes_gt, basis_proj)
    # Reshape data.
    X_gt = X_gt.reshape(X_gt.shape[0] / shape_gt[1], X_gt.shape[1] * shape_gt[1])
    # Define the outputs.
    y_gt = numpy.zeros((X_gt.shape[0], 1))
    
    print("# X_gt.shape")
    print(X_gt.shape)
    
    
    # Compute the forbidden spike times.
    spike_times_fbd = []
    # TODO: check the validity of 'int'.
    max_time_shift = int(float(sampling_rate) * 0.25e-3)
    for time_shift in xrange(-max_time_shift, max_time_shift+1):
        spike_times_fbd.append(spike_times_gt + time_shift)
    spike_times_fbd = numpy.concatenate(spike_times_fbd)
    spike_times_fbd = numpy.unique(spike_times_fbd)
    
    
    ##### NON GROUND TRUTH CELL'S SAMPLES ######################################
    
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
    alpha = 5
    print(spike_times_ngt_tmp.size)
    print(alpha * spike_times_gt.shape[0])
    idxs_ngt = numpy.random.choice(spike_times_ngt_tmp.size,
                                   size=alpha*spike_times_gt.shape[0],
                                   replace=False)
    idxs_ngt = numpy.unique(idxs_ngt)
    spike_times_ngt = spike_times_ngt_tmp[idxs_ngt]
    
    # Load the spikes of all the 'non ground truth cells".
    spikes_ngt = load_chunk(params, spike_times_ngt)
    
    # Reshape data.
    shape_ngt = spikes_ngt.shape
    spikes_ngt = spikes_ngt.reshape(shape_ngt[0], shape_ngt[1] * shape_ngt[2])
    spikes_ngt = spikes_ngt.T
    
    # Compute the PCA coordinates of each spike of the "non ground truth cells".
    X_ngt = numpy.dot(spikes_ngt, basis_proj)
    # Reshape data.
    X_ngt = X_ngt.reshape(X_ngt.shape[0] / shape_gt[1], X_ngt.shape[1] * shape_gt[1])
    # Define the outputs.
    y_ngt = numpy.ones((X_ngt.shape[0], 1))
    
    print("# X_ngt.shape")
    print(X_ngt.shape)
    
    
    ##### NOISE SAMPLES ########################################################
    
    # Compute the PCA coordinates of each "non-spike" sample.
    # TODO: replace temporary solution for 'low' and 'high'.
    low = min(numpy.amin(spike_times_gt), numpy.amin(spike_times_ngt))
    high = max(numpy.amax(spike_times_gt), numpy.amin(spike_times_ngt))
    size = spike_times_ngt_tmp.size
    spike_times_noi = numpy.random.random_integers(low, high, size)
    spike_times_noi = numpy.unique(spike_times_noi)
    spike_times_noi = numpy.setdiff1d(spike_times_noi, spike_times_fbd)
    alpha = 5
    print(spike_times_noi.size)
    print(alpha * spike_times_gt.shape[0])
    idxs_noi = numpy.random.choice(spike_times_noi.size, size=alpha*spike_times_gt.shape[0], replace=False)
    idxs_noi = numpy.unique(idxs_noi)
    spike_times_noi = spike_times_noi[idxs_noi]
    
    # TODO: filter ground truht spike times.
    
    # Load some "non-spike" samples.
    spikes_noi = load_chunk(params, spike_times_noi)
    
    # Reshape data.
    shape_noi = spikes_noi.shape
    spikes_noi = spikes_noi.reshape(shape_noi[0], shape_noi[1] * shape_noi[2])
    spikes_noi = spikes_noi.T
    
    # Compute the PCA coordinates of each "non-spike" sample.
    X_noi = numpy.dot(spikes_noi, basis_proj)
    # Reshape data.
    X_noi = X_noi.reshape(X_noi.shape[0] / shape_noi[1], X_noi.shape[1] * shape_noi[1])
    # Define outputs.
    y_noi = numpy.ones((X_noi.shape[0], 1))
    
    print("# X_noi.shape")
    print(X_noi.shape)
    
    
    ##### SANITY PLOTS #########################################################
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    #max_component = X_gt.shape[1]
    max_component = 1 * 3
    x_component = 0
    y_component = 1
    y_component = min(y_component, max_component - 1)
    
    print("# Sanity plots...")
    
    fig = plt.figure()
    for x_component in xrange(0, max_component):
        for y_component in xrange(x_component + 1, max_component):
            fig.suptitle("Samples for validation")
            gs = gridspec.GridSpec(1, 1)
            ax = fig.add_subplot(gs[0])
            ax.hold(True)
            ax.plot(X_noi[:, x_component], X_noi[:, y_component], 'r.')
            ax.plot(X_ngt[:, x_component], X_ngt[:, y_component], 'b.')
            ax.plot(X_gt[:, x_component], X_gt[:, y_component], 'g.')
            ax.hold(False)
            ax.set_xlabel("%dst principal component" %(x_component + 1))
            ax.set_ylabel("%dnd principal component" %(y_component + 1))
            ax.grid()
            filename = "/tmp/validation-samples-%d-%d.png" %(x_component, y_component)
            plt.savefig(filename)
            print("%s done." %filename)
            fig.clear()
    
    

    ##### SAMPLES ##############################################################
    
    # Option to include the pairwise product of feature vector elements.
    pairwise = True
    
    # Create the datasets to train the neural network.
    ## Create the input dataset.
    N = X_gt.shape[1]
    
    X_raw = numpy.vstack((X_gt, X_ngt, X_noi))
    
    if pairwise:
        # With pairwise product of feature vector elements.
        shape = (X_gt.shape[0] + X_ngt.shape[0] + X_noi.shape[0], N + N * (N + 1) / 2)
    else:
        # Without pairwise product of feature vector elments.
        shape = (X_gt.shape[0] + X_ngt.shape[0] + X_noi.shape[0], N)
    X = numpy.zeros(shape)
    X[:, :N] = X_raw
    
    print("# X.shape")
    print(X.shape)
    
    if pairwise:
        # Add the pairwise product of feature vector elements.
        k = 0
        for i in xrange(0, N):
            for j in xrange(i, N):
                X[:, N + k] = numpy.multiply(X[:, i], X[:, j])
                k = k + 1
    
        print("# X.shape (with pairwise product of feature vector element")
        print(X.shape)
    
    ## Create the output dataset.
    y_raw = numpy.vstack((y_gt, y_ngt, y_noi))
    y_raw = y_raw.flatten()
    y = y_raw
    
    
    
    ##### INITIAL PARAMETER ####################################################
    
    # TODO: compute the covariance matrix of 
    # TODO: compute coef of the ellipse
    
    # Useful function to convert an ellispoid in standard form to an ellispoid
    # in general form.
    def ellipsoid_standard_to_general(t, s, O):
        # Translation from standard matrix to general matrix.
        d = numpy.divide(1.0, numpy.power(s, 2.0))
        D = numpy.diag(d)
        A = O * D * O.T
        ##### TODO: remove test zone
        w, v = numpy.linalg.eigh(A)
        #print("# det(A)")
        #print(numpy.linalg.det(A))
        print("# Eigenvalues")
        print(w)
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
            A[i, i] = coefs[1 + N + k]
            k = k + 1
            for j in xrange(i + 1, N):
                A[i, j] = coefs[1 + N + k] / 2.0
                A[j, i] = coefs[1 + N + k] / 2.0
                k = k + 1
        # Retrieve b.
        b = coefs[1:1+N, 0]
        # Retrieve c.
        c = coefs[0]
        return A, b, c
    
    
    method = 'covariance'
    #method = 'geometric'
    
    print("")
    
    if method is 'covariance':
        
        mu = numpy.mean(X_gt.T, axis=1)
        sigma = numpy.cov(X_gt.T )
        k = 1.0
        
        sigma_inv = numpy.linalg.inv(sigma)
        
        A_init = sigma_inv
        b_init = - 2.0 * numpy.dot(mu, sigma_inv)
        c_init = numpy.dot(mu, numpy.dot(sigma_inv, mu)) - k * k
        
        coefs_init = ellipsoid_matrix_to_coefs(A_init, b_init, c_init)
        
    elif method is 'geometric':
        
        t = numpy.mean(X_gt.T, axis=1)
        c = numpy.cov(X_gt.T)
        
        #####
        fig = plt.figure()
        ax = fig.gca()
        cax = ax.imshow(c, interpolation='nearest')
        ax.set_title("Covariance matrix of ground truth")
        fig.colorbar(cax)
        fig.savefig("/tmp/covariance-matrix-gt.png")
        #####
        
        s, O = numpy.linalg.eigh(c)
        coefs_init = ellipsoid_standard_to_general(t, s, O)
        
    else:
        raise(Exception)
    
    print("# coefs_init")
    print(coefs_init)
    
    print("")
    
    
    #A_init, b_init, c_init = ellipsoid_coefs_to_matrix(coefs_init)
    
    
    
    # SANITY PLOT (ELLIPSE PROJECTION) #########################################
    
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
    mu_gt = numpy.mean(X_gt, axis=0)
    mu_gt_ = pca.transform(mu_gt)
    mu_ngt = numpy.mean(X_ngt, axis=0)
    mu_ngt_ = pca.transform(mu_ngt)
    mu_noi = numpy.mean(X_noi, axis=0)
    mu_noi_ = pca.transform(mu_noi)
    
    # Ellipse transformation.
    t = - 0.5 * numpy.dot(numpy.linalg.inv(A_init), b_init)
    f = 1.0 / (0.25 * numpy.dot(b_init, numpy.dot(numpy.linalg.inv(A_init), b_init)) - c_init)
    s, O = numpy.linalg.eigh(numpy.linalg.inv(f * A_init))
    s = numpy.sqrt(s)
    O_ = pca.transform(t + numpy.multiply(s, O).T)
    
    # Find plot limits.
    pad = 0.1
    x_dif = numpy.amax(X_raw_[:, 0]) - numpy.amin(X_raw_[:, 0])
    x_min = numpy.amin(X_raw_[:, 0]) - pad * x_dif
    x_max = numpy.amax(X_raw_[:, 0]) + pad * x_dif
    y_dif = numpy.amax(X_raw_[:, 1]) - numpy.amin(X_raw_[:, 1])
    y_min = numpy.amin(X_raw_[:, 1]) - pad * y_dif
    y_max = numpy.amax(X_raw_[:, 1]) + pad * y_dif
    
    # Plot.
    fig = plt.figure()
    ax = fig.gca()
    ax.grid()
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xlabel("1st component")
    ax.set_ylabel("2nd component")
    # Plot datasets.
    ax.scatter(X_gt_[:, 0], X_gt_[:, 1], c='g')
    ax.scatter(X_ngt_[:, 0], X_ngt_[:, 1], c='b')
    ax.scatter(X_noi_[:, 0], X_noi_[:, 1], c='r')
    # Plot ellipse transformation.
    ax.scatter(O_[:, 0], O_[:, 1], c='y', s=50)
    # Plot means of datasets.
    ax.scatter(mu_gt_[:, 0], mu_gt_[:, 1], c='y', s=100)
    ax.scatter(mu_ngt_[:, 0], mu_ngt_[:, 1], c='y', s=100)
    ax.scatter(mu_noi_[:, 0], mu_noi_[:, 1], c='y', s=100)
    fig.savefig("/tmp/sanity-ellipse-projection-init.png")
    
    ############################################################################
    
    # # Compute the Mahalanobis distance.
    # def evaluate_ellipse(A, b, c, X):
    #     x2 = numpy.sum(numpy.multiply(X.T, numpy.dot(A, X.T)), axis=0)
    #     x1 = numpy.dot(b, X.T)
    #     x0 = c
    #     return numpy.sqrt(x2 + x1 + x0)
    # def Mahalanobis_distance(A, mu, X):
    #     X_ = X - mu
    #     #X_ = X
    #     return numpy.sqrt(numpy.sum(numpy.multiply(X_.T, numpy.dot(A, X_.T)), axis=0))
    # mu = numpy.mean(X_gt, axis=0)
    # Mhlnb_gt = Mahalanobis_distance(A_init, mu, X_gt)
    # Mhlnb_ngt = Mahalanobis_distance(A_init, mu, X_ngt)
    # Mhlnb_noi = Mahalanobis_distance(A_init, mu, X_noi)
    # Ell_gt = evaluate_ellipse(A_init, b_init, c_init, X_gt)
    # #####
    # fig = plt.figure()
    # ax = fig.gca()
    # ax.grid()
    # ax.hist(Ell_gt, bins=500)
    # fig.savefig("/tmp/ellipse-values.png")
    # #####
    # #####
    # fig = plt.figure()
    # ax = fig.gca()
    # ax.grid()
    # ax.hist(Mhlnb_noi, bins=50, color='r')
    # ax.hist(Mhlnb_ngt, bins=50, color='b')
    # ax.hist(Mhlnb_gt, bins=75, color='g')
    # fig.savefig("/tmp/mahalanobis-init.png")
    # #####
    
    # def project_ellipse(A, b, c, v1, v2):
    #     pass
    #     return
    
    ############################################################################
    
    
    ##### TODO: compute the False Positive and False Negative for this initial
    #####       ellipse.
    
    
    ##### LEARNING #############################################################
    
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
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h),
                            numpy.arange(y_min, y_max, h))
    
    # Preprocess dataset.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    print("# X_train.shape")
    print(X_train.shape)
    print("# X_test.shape")
    print(X_test.shape)
    
    # Declare model.
    # clf = MLPClassifier(hidden_layer_sizes=(),
    #                     activation='logistic',
    #                     algorithm='l-bfgs',
    #                     alpha=1e-12, # L2 penalty parameter (regularization)
    #                     learning_rate='adaptive',
    #                     max_iter=2000,
    #                     random_state=1,
    #                     tol=1e-5,
    #                     warm_start=True)
    clf = MLPClassifier(hidden_layer_sizes=(),
                        activation='logistic',
                        algorithm='sgd',
                        learning_rate='constant',
                        momentum=0.9,
                        nesterovs_momentum=True,
                        learning_rate_init=0.2,
                        warm_start=True)
    
    # Initialize model (i.e. fake launch + weights initialization).
    clf.set_params(max_iter=5)
    clf.set_params(warm_start=False)
    clf.fit(X_train, y_train)
    clf.coefs_ = [coefs_init[1:, :]]
    clf.intercepts_ = [coefs_init[:1, :]]
    
    # Train model.
    clf.set_params(max_iter=400)
    clf.set_params(warm_start=True)
    clf.fit(X_train, y_train)
    
    
    # Print the current loss computed with the loss function.
    print("# clf._loss_")
    print(clf.loss_)
    # Print the number of iterations the algorithm has ran.
    print("# clf.n_iter_")
    print(clf.n_iter_)
    # Print the score on the test set.
    print("# clf.score(X_test, y_test)")
    print(clf.score(X_test, y_test))
    # Print prediction at (0, 0).
    print("# clf.predict([? * [0.0]])")
    if pairwise:
        print(clf.predict([(N + N * (N + 1) / 2) * [0.0]]))
    else:
        print(clf.predict([N * [0.0]]))
    
    
    # Retrieve the coefficients of the ellipsoid.
    weights = clf.coefs_[0].flatten() # weight vector
    bias = clf.intercepts_[0].flatten() # bias vector
    # Concatenate the coefficients.
    coefs = numpy.concatenate((bias, weights))
    coefs = coefs.reshape(-1, 1)
    
    
    A, b, c = ellipsoid_coefs_to_matrix(coefs)
    
    ############################################################################
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
    mu_gt = numpy.mean(X_gt, axis=0)
    mu_gt_ = pca.transform(mu_gt)
    mu_ngt = numpy.mean(X_ngt, axis=0)
    mu_ngt_ = pca.transform(mu_ngt)
    mu_noi = numpy.mean(X_noi, axis=0)
    mu_noi_ = pca.transform(mu_noi)
    
    # Ellipse transformation.
    t = - 0.5 * numpy.dot(numpy.linalg.inv(A), b)
    f = 1.0 / (0.25 * numpy.dot(b, numpy.dot(numpy.linalg.inv(A), b)) - c)
    print("# f")
    print(f)
    s, O = numpy.linalg.eigh(numpy.linalg.inv(f * A))
    print("# s")
    print(s)
    #s = numpy.sqrt(s)
    #O_ = pca.transform(t + numpy.multiply(s, O).T)
    
    # Find plot limits.
    pad = 0.1
    x_dif = numpy.amax(X_raw_[:, 0]) - numpy.amin(X_raw_[:, 0])
    x_min = numpy.amin(X_raw_[:, 0]) - pad * x_dif
    x_max = numpy.amax(X_raw_[:, 0]) + pad * x_dif
    y_dif = numpy.amax(X_raw_[:, 1]) - numpy.amin(X_raw_[:, 1])
    y_min = numpy.amin(X_raw_[:, 1]) - pad * y_dif
    y_max = numpy.amax(X_raw_[:, 1]) + pad * y_dif
    
    # Plot.
    fig = plt.figure()
    ax = fig.gca()
    ax.grid()
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xlabel("1st component")
    ax.set_ylabel("2nd component")
    # Plot datasets.
    ax.scatter(X_gt_[:, 0], X_gt_[:, 1], c='g')
    ax.scatter(X_ngt_[:, 0], X_ngt_[:, 1], c='b')
    ax.scatter(X_noi_[:, 0], X_noi_[:, 1], c='r')
    # # Plot ellipse transformation.
    # ax.scatter(O_[:, 0], O_[:, 1], c='y', s=50)
    # Plot means of datasets.
    ax.scatter(mu_gt_[:, 0], mu_gt_[:, 1], c='y', s=100)
    ax.scatter(mu_ngt_[:, 0], mu_ngt_[:, 1], c='y', s=100)
    ax.scatter(mu_noi_[:, 0], mu_noi_[:, 1], c='y', s=100)
    fig.savefig("/tmp/sanity-ellipse-projection.png")
    
    import sys
    sys.exit(0)
    
    ############################################################################
    
    # Compute the Mahalanobis distance.
    def Mahalanobis_distance(A, mu, X):
        X_ = X - mu
        return numpy.sum(numpy.multiply(X_.T, numpy.dot(A, X_.T)), axis=0)
    print(A.shape)
    print(X_gt.shape)
    print(X_ngt.shape)
    print(X_noi.shape)
    mu = numpy.mean(X_gt, axis=0)
    Mhlnb_gt = Mahalanobis_distance(-A, mu, X_gt)
    Mhlnb_ngt = Mahalanobis_distance(-A, mu, X_ngt)
    Mhlnb_noi = Mahalanobis_distance(-A, mu, X_noi)
    #####
    fig = plt.figure()
    ax = fig.gca()
    ax.grid()
    ax.hist(Mhlnb_ngt, bins=25, color='b')
    ax.hist(Mhlnb_gt, bins=75, color='g')
    ax.hist(Mhlnb_noi, bins=25, color='r')
    fig.savefig("/tmp/mahalanobis.png")
    #####
    
    ##### TODO: compute the Mahalanobis distance.
    
    import sys
    sys.exit(0)
    
    
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
    
    
    ##### SANITY PLOT
    
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
    filename = "/tmp/decision-boundaries-%d-%d.png" %(x_component, y_component)
    plt.savefig(filename)
    print("%s done." %filename)
    fig.clear()
    
    
    ##### SANITY PLOT (with PCA)
    
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
    
    print("# Shapes of the components")
    print(vpca.shape)
    
    v = numpy.array([[1.0, 0.0], [0.0, 1.0]])
    vpca = pca.inverse_transform(v)
    
    print("# Shapes after inverse transform of v (i.e. vpca)")
    print(vpca.shape)
    print("# Shapes X_raw")
    print(X_raw.shape)
    print("# Norms of vpca0 and vpca1")
    print(numpy.linalg.norm(vpca, axis=1))
    
    # Retrieve the coefficients of the ellipsoid.
    weights = clf.coefs_[0].flatten() # weight vector
    bias = clf.intercepts_[0].flatten() # bias vector
    # Concatenate the coefficients.
    coefs = numpy.concatenate((bias, weights))
    
    print("#####")
    print("# Weights")
    print(weights)
    print(type(weights))
    print(weights.shape)
    print("# Bias")
    print(bias)
    print(type(bias))
    print(bias.shape)
    print("# Coefs")
    print(coefs)
    print(type(coefs))
    print(coefs.shape)
    
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
        print("# K")
        print(K)
        print("# N")
        print(N)
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
        print("# Test of symmetry")
        print(numpy.all(A == A.T))
        ##### end test zone
        ##### TODO: remove plot zone
        fig = plt.figure()
        ax = fig.gca()
        cax = ax.imshow(A, interpolation='nearest', cmap='jet')
        fig.colorbar(cax)
        fig.savefig("/tmp/ellipse.png")
        ##### end plot zone
        
        # Each eigenvector of A lies along one of the axes.
        evals, evecs = numpy.linalg.eigh(A)
        
        ##### TODO: remove print zone.
        print("# Semi-axes computation")
        print("## det(A)")
        print(numpy.linalg.det(A))
        print("## evals")
        print(evals)
        ##### end print zone.
        
        # Semi-axes from reduced canonical equation.
        eaxis = numpy.sqrt(- c / evals)
        return center, eaxis, evecs
    
    
##### TODO: remove test zone (standard -> genral -> standard)
    # print("")
    # # Test.
    # t = numpy.array([1.0, 2.0])
    # s = numpy.array([0.5, 0.2])
    # O = numpy.array([[1.0, 0.0], [0.0, 1.0]])
    
    # print("# t")
    # print(t)
    # print("# s")
    # print(s)
    # print("# O")
    # print(O)
    
    # coefs_bis = ellipsoid_standard_to_general(t, s, O)
    
    # print("# coefs_bis")
    # print(coefs_bis)
    
    # t_bis, s_bis, O_bis = ellipsoid_general_to_standard(coefs_bis)
    
    # print("# t_bis")
    # print(t_bis)
    # print("# s_bis")
    # print(s_bis)
    # print("# O_bis")
    # print(O_bis)
    
    # print("")
    
    # import sys
    # sys.exit(0)
##### end test zone
    
    center, eaxis, evecs = ellipsoid_general_to_standard(coefs)
        
    print("# Conversion")
    print("# Center")
    print(center)
    print(center.shape)
    print("# Eigenaxis")
    print(eaxis)
    print(eaxis.shape)
    print("# Eigenvectors")
    print(evecs)
    print(evecs.shape)
    
    coefs_bis = ellipsoid_standard_to_general(center, eaxis, evecs)
    
    print("# Transform and untransfrom")
    print("# coefs")
    print(coefs)
    print("# coefs_bis")
    print(coefs_bis)
    
    
    # TODO: compute the projection of the eigenvectors on Vect(vpca[0, :], vpca[1, :]).
    # Projection of the center.
    shape = (1, 2)
    cprojs = pca.transform(center)
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
    
    
    print("# Center projection")
    print(cprojs)
    print(cprojs.shape)
    # print("# Eigenprojections")
    # print(eprojs)
    # print(eprojs.shape)
    
    
    
    
##### TODO: remove plot zone.
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(vpca[0, :])
    fig.savefig("/tmp/plot0.png")
    
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(vpca[1, :])
    fig.savefig("/tmp/plot1.png")
    
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(vpca[1, :] - vpca[0, :])
    fig.savefig("/tmp/plot2.png")
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
    
    
    # Plot reduced dataset.
    filename = "/tmp/reduced-dataset.png"
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
    fig.savefig(filename)
    
    # Plot reduced datset restricted to the ground truth cell.
    filename = "/tmp/reduced-dataset-true.png"
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
    fig.savefig(filename)
    
    # Plot reduced datset restricted non ground truth cells and noise.
    filename = "/tmp/reduced-dataset-false.png"
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
    fig.savefig(filename)
    
    
    
    # # First attempt (naive approach)
    
    # # Define the sigmoid function.
    # def nonlin(x, deriv=False):
    #     if (True == deriv):
    #         return x * (1.0 - x)
    #     else:
    #         return 1.0 / (1.0 + numpy.exp(-x))

    # # Initialize the weights randomly with mean 0.
    # # shape = (N + N * (N + 1) / 2, 1)
    # shape = (N, 1)
    # W = numpy.random.random(shape)
    # W = 2.0 * (W - 1.0)
    
    # print("# Weights before training:")
    # W_tmp = W.reshape(W.shape[0] / basis_proj.shape[1], W.shape[1] * basis_proj.shape[1])
    # print(W_tmp)
    
    # n_iters = 100
    # for iter in xrange(n_iters):
    #     if 0 == ((iter + 1) % 10):
    #         print("Iteration %d / %d..." %(iter + 1, n_iters))
    #     # Forward propagation.
    #     l0 = X
    #     l1 = nonlin(numpy.dot(l0, W))
    #     l1_error = y - l1
    #     l1_delta = l1_error * nonlin(l1, deriv=True)
    #     W_delta = numpy.dot(l0.T, l1_delta)
    #     W = W + W_delta
    #     if 0 == ((iter + 1) % 10):
    #         W_delta_norm = numpy.linalg.norm(W_delta)
    #         print("  Norm of the delta of the weights: %f" %W_delta_norm)
    
    
    # print("# Weights after training:")
    # W_tmp = W.reshape(W.shape[0] / basis_proj.shape[1], W.shape[1] * basis_proj.shape[1])
    # print(W_tmp)
    
    
    print("End validating.")
    
    return
