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
    
    
    ##### GROUND TRUTH CELL'S SAMPLES
    
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
    y_gt = numpy.ones((X_gt.shape[0], 1))
    
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
    
    
    ##### NON GROUND TRUTH CELL'S SAMPLES
    
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
    alpha = 2
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
    y_ngt = numpy.zeros((X_ngt.shape[0], 1))
    
    print("# X_ngt.shape")
    print(X_ngt.shape)
    
    
    ##### NOISE SAMPLES
    
    # Compute the PCA coordinates of each "non-spike" sample.
    # TODO: replace temporary solution for 'low' and 'high'.
    low = min(numpy.amin(spike_times_gt), numpy.amin(spike_times_ngt))
    high = max(numpy.amax(spike_times_gt), numpy.amin(spike_times_ngt))
    size = spike_times_ngt_tmp.size
    spike_times_noi = numpy.random.random_integers(low, high, size)
    spike_times_noi = numpy.unique(spike_times_noi)
    spike_times_noi = numpy.setdiff1d(spike_times_noi, spike_times_fbd)
    alpha = 2
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
    y_noi = numpy.zeros((X_noi.shape[0], 1))
    
    print("# X_noi.shape")
    print(X_noi.shape)
    
    
    ##### SANITY PLOTS
    
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
    
    

    ##### SAMPLES
    
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
    
    
    ##### LEARNING
    
    mode = 'decision'
    mode = 'prediction'
    
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    
    x_component = 0
    y_component = 2
    
    X = StandardScaler().fit_transform(X)
    
    pad = 0.5 # padding coefficient
    x_dif = numpy.amax(X[:, x_component]) - numpy.amin(X[:, x_component])
    x_min = numpy.amin(X[:, x_component]) - pad * x_dif
    x_max = numpy.amax(X[:, x_component]) + pad * x_dif
    y_dif = numpy.amax(X[:, y_component]) - numpy.amin(X[:, y_component])
    y_min = numpy.amin(X[:, y_component]) - pad * y_dif
    y_max = numpy.amax(X[:, y_component]) + pad * y_dif
    h = 0.1 # step size in the mesh
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h),
                            numpy.arange(y_min, y_max, h))
    
    # Preprocess dataset.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    # Declare model.
    clf = MLPClassifier(hidden_layer_sizes=(),
                        activation='logistic',
                        algorithm="l-bfgs",
                        alpha=1e-12, # L2 penalty parameter (regularization)
                        learning_rate='adaptive',
                        random_state=1,
                        tol=1e-5)
    # Train model.
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
    # Test of the 'score' method.
    y_test_pred = clf.predict(X_test)
    print("# clf.score(X_test, y_test_pred) (i.e. best possible score)")
    print(clf.score(X_test, y_test_pred))
    
    # Print prediction at (0, 0).
    print("# clf.predict([? * [0.0]])")
    if pairwise:
        print(clf.predict([(N + N * (N + 1) / 2) * [0.0]]))
    else:
        print(clf.predict([N * [0.0]]))
    
    # Compute prediction on a grid of the input space for plotting.
    shape_pre = (xx.shape[0] * xx.shape[1], X_train.shape[1])
    X_pre = numpy.zeros(shape_pre)
    X_pre[:, x_component] = xx.ravel()
    X_pre[:, y_component] = yy.ravel()
    if pairwise:
        k = 0
        for i in xrange(0, N):
            for j in xrange(i, N):
                if i == x_component and j == y_component:
                    X_pre[:, N + k] = numpy.multiply(xx.ravel(), yy.ravel())
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
    #ax.scatter(X_train[:, x_component], X_train[:, y_component], c=y_train, cmap='bwr')
    ax.scatter(X_test[:, x_component], X_test[:, y_component], c=y_test, cmap='bwr', alpha=0.6)
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
    
    n_components = 2
    
    # Compute PCA.
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
    v1 = numpy.array([1.0, 0.0])
    v2 = numpy.array([0.0, 1.0])
    pca1 = pca.inverse_transform(v1)
    pca2 = pca.inverse_transform(v2)
    print("# Norms of pca1 and pca2")
    print(numpy.linalg.norm(pca1))
    print(numpy.linalg.norm(pca2))
    # Retrieve the parameters of the ellipsoid.
    coefs = clf.coefs_[0].flatten() # weight vector
    intercepts = clf.intercepts_[0] # bias vector
    print("#####")
    print(coefs)
    print(type(coefs))
    print(coefs.shape)
    print(intercepts)
    print(type(intercepts))
    print(intercepts.shape)
    
##### TODO: remove plot zone.
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(pca1)
    fig.savefig("/tmp/plot1.png")
    
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(pca2)
    fig.savefig("/tmp/plot2.png")
    
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(pca2 - pca1)
    fig.savefig("/tmp/plot3.png")
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
    ax.scatter(X_raw_r[:, 0], X_raw_r[:, 1], c=y, cmap='bwr')
    ax.grid()
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
    ax.scatter(X_raw_r[y == 1, 0], X_raw_r[y == 1, 1], c='r')
    ax.grid()
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
    ax.scatter(X_raw_r[y == 0, 0], X_raw_r[y == 0, 1], c='b')
    ax.grid()
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
