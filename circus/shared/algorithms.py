import matplotlib
matplotlib.use('Agg')
import scipy.optimize, numpy, pylab, mdp, scipy.spatial.distance, scipy.stats


def distancematrix(data, weight=None):
    
    if weight is None:
        weight = numpy.ones(data.shape[1])/data.shape[1]    
    distances = scipy.spatial.distance.pdist(data, 'wminkowski', p=2, w=numpy.sqrt(weight))**2
    return distances

def fit_rho_delta(xdata, ydata, display=False, threshold=numpy.exp(-3**2), max_clusters=10, save=False):

    gamma = xdata * ydata

    def powerlaw(x, a, b, k): 
        with numpy.errstate(all='ignore'):
            return a*(x**k) + b

    try:
        sort_idx     = numpy.argsort(xdata)    
        result, pcov = scipy.optimize.curve_fit(powerlaw, xdata, numpy.log(ydata), [1, 1, 0])
        data_fit     = numpy.exp(powerlaw(xdata[sort_idx], result[0], result[1], result[2]))
        xaxis        = numpy.linspace(xdata.min(), xdata.max(), 1000)
        padding      = threshold
    except Exception:
        return numpy.argsort(gamma)

    if display:
        fig      = pylab.figure(figsize=(15, 5))
        ax       = fig.add_subplot(111)
        
        ax.plot(xdata, ydata, 'k.')
        ax.plot(xdata[sort_idx], data_fit)
        ax.set_yscale('log')
        ax.set_ylabel(r'$\delta$')
        ax.set_xlabel(r'$\rho$')

    idx      = numpy.where((xdata > padding) & (ydata > data_fit))[0]
    if len(idx) == 0:
        subidx = numpy.argsort(gamma)
    with numpy.errstate(all='ignore'):
        mask     = (xdata > padding).astype(int)
        value    = ydata - numpy.exp(powerlaw(xdata, result[0], result[1], result[2]))
        value   *= mask
        subidx   = numpy.argsort(value)[::-1]

        if display:
            ax.plot(xdata[subidx[:max_clusters]], ydata[subidx[:max_clusters]], 'ro')
            if save:
                pylab.savefig('%s/rho_delta_%s.png' %(save[0], save[1]))
                pylab.close()
            else:
                pylab.show()
        return subidx


def rho_estimation(data, dc=None, weight=None, update=None, compute_rho=True):

    N   = len(data)
    rho = numpy.zeros(N, dtype=numpy.float32)
        
    if update is None:
        dist = distancematrix(data, weight=weight)
        didx = lambda i,j: i*N + j - i*(i+1)/2 - i - 1

        if dc is None:
            sda      = numpy.argsort(dist)
            position = numpy.round(N*2/100.)
            dc       = dist[sda][int(position)]

        if compute_rho:
            exp_dist = numpy.exp(-(dist/dc)**2)
            for i in xrange(N):   
               rho[i] = numpy.sum(exp_dist[didx(i, numpy.arange(i+1, N))]) + numpy.sum(exp_dist[didx(numpy.arange(0, i-1), i)])  
    else:
        if weight is None:
            weight   = numpy.ones(data.shape[1])/data.shape[1]

        for i in xrange(N):
            dist     = numpy.sum(weight*(data[i] - update)**2, 1)
            exp_dist = numpy.exp(-(dist/dc)**2)
            rho[i]   = numpy.sum(exp_dist)
    return rho, dist, dc


def clustering(rho, dist, dc, smart_search=0, display=None, n_min=None, max_clusters=10, save=False):

    N                 = len(rho)
    maxd              = numpy.max(dist)
    didx              = lambda i,j: i*N + j - i*(i+1)/2 - i - 1
    ordrho            = numpy.argsort(rho)[::-1]
    rho_sorted        = rho[ordrho]
    delta, nneigh     = numpy.zeros(N, dtype=numpy.float32), numpy.zeros(N, dtype=numpy.int32)
    delta[ordrho[0]]  = -1
    for ii in xrange(N):
        delta[ordrho[ii]] = maxd
        for jj in xrange(ii):
            if ordrho[jj] > ordrho[ii]:
                xdist = dist[didx(ordrho[ii], ordrho[jj])]
            else:
                xdist = dist[didx(ordrho[jj], ordrho[ii])]

            if xdist < delta[ordrho[ii]]:
                delta[ordrho[ii]]  = xdist
                nneigh[ordrho[ii]] = ordrho[jj]
    
    delta[ordrho[0]] = delta.ravel().max()  
    threshold        = n_min * numpy.exp(-max(smart_search, 4)**2)
    clust_idx        = fit_rho_delta(rho, delta, max_clusters=max_clusters, threshold=threshold)
    
    def assign_halo(idx):
        cl      = numpy.empty(N, dtype=numpy.int32)
        cl[:]   = -1
        NCLUST  = len(idx)
        cl[idx] = numpy.arange(NCLUST)
        
        # assignation
        for i in xrange(N):
            if cl[ordrho[i]] == -1:
                cl[ordrho[i]] = cl[nneigh[ordrho[i]]]
        
        # halo
        halo = cl.copy()
        if NCLUST > 1:
            bord_rho = numpy.zeros(NCLUST, dtype=numpy.float32)
            for i in xrange(N):
                idx      = numpy.where((cl[i] < cl[i+1:N]) & (dist[didx(i, numpy.arange(i+1, N))] <= dc))[0]
                if len(idx) > 0:
                    myslice  = cl[i+1:N][idx]
                    rho_aver = (rho[i] + rho[idx]) / 2.
                    sub_idx  = numpy.where(rho_aver > bord_rho[cl[i]])[0]
                    if len(sub_idx) > 0:
                        bord_rho[cl[i]] = rho_aver[sub_idx].max()
                    sub_idx  = numpy.where(rho_aver > bord_rho[myslice])[0]
                    if len(sub_idx) > 0:
                        bord_rho[myslice[sub_idx]] = rho_aver[sub_idx]
            
            idx       = numpy.where(rho < bord_rho[cl])[0]
            halo[idx] = -1

        if n_min is not None:
            for cluster in xrange(NCLUST):
                idx = numpy.where(halo == cluster)[0]
                if len(idx) < n_min:
                    halo[idx] = -1
                    NCLUST   -= 1
        return halo, NCLUST

    #print "Try to maximize the number of clusters..."
    NCLUST = 0
    for n in xrange(max_clusters):
        halo_temp, NCLUST_temp = assign_halo(clust_idx[:n+1])
        if NCLUST_temp >= NCLUST:
            halo   = numpy.array(halo_temp).copy()
            NCLUST = NCLUST_temp

    return halo, rho, delta, clust_idx


def merging(groups, sim_same_elec, data):

    def perform_merging(groups, sim_same_elec, data):
        mask      = numpy.where(groups > -1)[0]
        clusters  = numpy.unique(groups[mask])
        dmin      = numpy.inf
        to_merge  = [None, None]
        
        for ic1 in xrange(len(clusters)):
            idx1 = numpy.where(groups == clusters[ic1])[0]
            m1   = numpy.median(data[idx1], 0)
            for ic2 in xrange(ic1, len(clusters)):
                idx2 = numpy.where(groups == clusters[ic2])[0]
                m2   = numpy.median(data[idx2], 0)
                v_n  = m1 - m2      
                pr_1 = numpy.dot(data[idx1], v_n)
                pr_2 = numpy.dot(data[idx2], v_n)

                norm = numpy.median(numpy.abs(pr_1 - numpy.median(pr_1)))**2 + numpy.median(numpy.abs(pr_2 - numpy.median(pr_2)))**2

                with numpy.errstate(all='ignore'):  
                    dist = numpy.abs(numpy.median(pr_1) - numpy.median(pr_2))/numpy.sqrt(norm)
                    
                if dist < dmin:
                    dmin     = dist
                    to_merge = [ic1, ic2]

        if dmin < sim_same_elec:
            groups[numpy.where(groups == clusters[to_merge[1]])[0]] = clusters[to_merge[0]]
            return True, groups
        
        return False, groups

    has_been_merged = True
    mask            = numpy.where(groups > -1)[0]
    clusters        = numpy.unique(groups[mask])
    merged          = [len(clusters), 0]

    while has_been_merged:
        has_been_merged, groups = perform_merging(groups, sim_same_elec, data)
        if has_been_merged:
            merged[1] += 1
    return groups, merged


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False, show=False, ax=None):

    """
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    """

    x = numpy.atleast_1d(x).astype('float64')
    if x.size < 3:
        return numpy.array([], dtype=numpy.int32)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = numpy.where(numpy.isnan(x))[0]
    if indnan.size:
        x[indnan] = numpy.inf
        dx[numpy.where(numpy.isnan(dx))[0]] = numpy.inf
    ine, ire, ife = numpy.array([[], [], []], dtype=numpy.int32)
    if not edge:
        ine = numpy.where((numpy.hstack((dx, 0)) < 0) & (numpy.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = numpy.where((numpy.hstack((dx, 0)) <= 0) & (numpy.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = numpy.where((numpy.hstack((dx, 0)) < 0) & (numpy.hstack((0, dx)) >= 0))[0]
    ind = numpy.unique(numpy.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[numpy.in1d(ind, numpy.unique(numpy.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = numpy.min(numpy.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = numpy.delete(ind, numpy.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[numpy.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = numpy.zeros(ind.size, dtype=numpy.bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = numpy.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = numpy.nan
        if valley:
            x = -x
        pylab.plot(ind, x[ind], 'ro')
        pylab.plot(x, 'k')

    return ind