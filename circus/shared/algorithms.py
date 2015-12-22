import matplotlib
matplotlib.use('Agg')
import os
os.environ['MDP_DISABLE_SKLEARN']='yes'
import scipy.optimize, numpy, pylab, mdp, scipy.spatial.distance, scipy.stats, progressbar
from circus.shared.files import load_data, write_datasets, get_overlaps

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
                pylab.savefig(os.path.join(save[0], 'rho_delta_%s.png' %(save[1])))
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
                indices = numpy.concatenate((didx(i, numpy.arange(i+1, N)), didx(numpy.arange(0, i-1), i)))
                rho[i]  = numpy.sum(exp_dist[indices])  
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

    halo, NCLUST = assign_halo(clust_idx[:max_clusters+1])

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
            for ic2 in xrange(ic1+1, len(clusters)):
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

def slice_templates(comm, params, to_remove=None, to_merge=None):

    import h5py, shutil
    parallel_hdf5  = h5py.get_config().mpi
    file_out_suff  = params.get('data', 'file_out_suff')

    if parallel_hdf5 or (comm.rank == 0):
        myfile         = h5py.File(file_out_suff + '.templates.hdf5', 'r')
        old_templates  = myfile.get('templates')
        old_limits     = myfile.get('limits')[:]
        N_e, N_t, N_tm = old_templates.shape
        if to_merge is not None:
            to_remove = []
            for count in xrange(len(to_merge)):
                remove     = to_merge[count][1]
                to_remove += [remove]

        all_templates = set(numpy.arange(N_tm/2))
        to_keep       = numpy.array(list(all_templates.difference(to_remove)))

    if parallel_hdf5:
        hfile     = h5py.File(file_out_suff + '.templates-new.hdf5', 'w', driver='mpio', comm=comm)
        positions = numpy.arange(comm.rank, len(to_keep), comm.size)
    elif comm.rank == 0:
        hfile     = h5py.File(file_out_suff + '.templates-new.hdf5', 'w')
        positions = numpy.arange(len(to_keep))
    
    if parallel_hdf5 or (comm.rank == 0):
        local_keep = to_keep[positions]
        templates  = hfile.create_dataset('templates', shape=(N_e, N_t, 2*len(to_keep)), dtype=numpy.float32, chunks=True)
        limits     = hfile.create_dataset('limits', shape=(len(to_keep), 2), dtype=numpy.float32, chunks=True)
        for count, keep in zip(positions, local_keep):
            templates[:, :, count]                = old_templates[:, :, keep]
            templates[:, :, count + len(to_keep)] = old_templates[:, :, keep + N_tm/2]
            if to_merge is None:
                new_limits = old_limits[keep]
            else:
                subset     = numpy.where(to_merge[:, 0] == keep)[0]
                if len(subset) > 0:
                    idx        = numpy.unique(to_merge[subset].flatten())
                    new_limits = [numpy.min(old_limits[idx][:, 0]), numpy.max(old_limits[idx][:, 1])]
                else:
                    new_limits = old_limits[keep]
            limits[count]  = new_limits
        hfile.close()
        myfile.close()
    
    comm.Barrier()
    if comm.rank == 0:
        os.remove(file_out_suff + '.templates.hdf5')
        shutil.move(file_out_suff + '.templates-new.hdf5', file_out_suff + '.templates.hdf5')
    

def slice_clusters(comm, params, result):
    comm.Barrier()
    import h5py, shutil
    file_out_suff  = params.get('data', 'file_out_suff')
    N_e            = params.getint('data', 'N_e')

    if comm.rank == 0:
        cfile    = h5py.File(file_out_suff + '.clusters-new.hdf5', 'w')
        to_write = ['data_', 'clusters_', 'debug_', 'times_'] 
        for ielec in xrange(N_e):
            write_datasets(cfile, to_write, result, ielec)
       
        write_datasets(cfile, ['electrodes'], result)
        cfile.close()
        os.remove(file_out_suff + '.clusters.hdf5')
        shutil.move(file_out_suff + '.clusters-new.hdf5', file_out_suff + '.clusters.hdf5')

def merging_cc(comm, params, cc_merge, parallel_hdf5=False):

    def remove(result, distances, cc_merge):
        do_merge  = True
        to_merge  = numpy.zeros((0, 2), dtype=numpy.int32)
        g_idx     = range(len(distances))
        while do_merge:
            dmax      = distances.max()
            idx       = numpy.where(distances == dmax)
            one_merge = [idx[0][0], idx[1][0]]
            do_merge  = dmax >= cc_merge

            if do_merge:

                elec_ic1  = result['electrodes'][one_merge[0]]
                elec_ic2  = result['electrodes'][one_merge[1]]
                nic1      = one_merge[0] - numpy.where(result['electrodes'] == elec_ic1)[0][0]
                nic2      = one_merge[1] - numpy.where(result['electrodes'] == elec_ic2)[0][0]
                mask1     = result['clusters_' + str(elec_ic1)] > -1
                mask2     = result['clusters_' + str(elec_ic2)] > -1
                tmp1      = numpy.unique(result['clusters_' + str(elec_ic1)][mask1])
                tmp2      = numpy.unique(result['clusters_' + str(elec_ic2)][mask2])
                elements1 = numpy.where(result['clusters_' + str(elec_ic1)] == tmp1[nic1])[0]
                elements2 = numpy.where(result['clusters_' + str(elec_ic2)] == tmp2[nic2])[0]

                if len(elements1) > len(elements2):
                    to_remove = one_merge[1]
                    to_keep   = one_merge[0]
                    elec      = elec_ic2
                    elements  = elements2
                else:
                    to_remove = one_merge[0]
                    to_keep   = one_merge[1]
                    elec      = elec_ic1
                    elements  = elements1

                result['data_' + str(elec)]     = numpy.delete(result['data_' + str(elec)], elements, axis=0)
                result['clusters_' + str(elec)] = numpy.delete(result['clusters_' + str(elec)], elements) 
                result['debug_' + str(elec)]    = numpy.delete(result['debug_' + str(elec)], elements, axis=1)   
                result['times_' + str(elec)]    = numpy.delete(result['times_' + str(elec)], elements)
                result['electrodes']            = numpy.delete(result['electrodes'], to_remove)
                distances                       = numpy.delete(distances, to_remove, axis=0)
                distances                       = numpy.delete(distances, to_remove, axis=1)
                to_merge                        = numpy.vstack((to_merge, numpy.array([g_idx[to_keep], g_idx[to_remove]])))
                g_idx.pop(to_remove)

        return to_merge, result
            
    templates      = load_data(params, 'templates')
    N_e, N_t, N_tm = templates.shape
    nb_temp        = N_tm/2
    to_merge       = []
    
    result   = []
    myfile   = get_overlaps(comm, params, extension='-merging', erase=True, parallel_hdf5=parallel_hdf5, normalize=True, maxoverlap=False)
    filename = params.get('data', 'file_out_suff') + '.overlap-merging.hdf5'

    if comm.rank > 0:
        myfile.close()
    else:
        pair      = []
        result    = load_data(params, 'clusters')
        distances = numpy.zeros((nb_temp, nb_temp), dtype=numpy.float32)
        overlap   = myfile.get('overlap')
        for i in xrange(nb_temp):
            distances[i, i+1:] = numpy.max(overlap[i, i+1:nb_temp], 1)
            distances[i+1:, i] = distances[i, i+1:]

        distances /= (N_e*N_t)
        myfile.close()
        to_merge, result = remove(result, distances, cc_merge)       

    to_merge = numpy.array(to_merge)
    to_merge = comm.bcast(to_merge, root=0)

    if len(to_merge) > 0:
        slice_templates(comm, params, to_merge=to_merge)
        slice_clusters(comm, params, result)

    if comm.rank == 0:
        os.remove(filename)

    return [nb_temp, len(to_merge)]


def delete_mixtures(comm, params, parallel_hdf5=False):
        
    def remove(result, to_remove):
        for count in xrange(len(to_remove)):
            target   = to_remove[count]
            elec     = result['electrodes'][target]
            nic      = target - numpy.where(result['electrodes'] == elec)[0][0]
            mask     = result['clusters_' + str(elec)] > -1
            tmp      = numpy.unique(result['clusters_' + str(elec)][mask])
            elements = numpy.where(result['clusters_' + str(elec)] == tmp[nic])[0]
                    
            result['data_' + str(elec)]     = numpy.delete(result['data_' + str(elec)], elements, axis=0)
            result['clusters_' + str(elec)] = numpy.delete(result['clusters_' + str(elec)], elements) 
            result['debug_' + str(elec)]    = numpy.delete(result['debug_' + str(elec)], elements, axis=1)   
            result['times_' + str(elec)]    = numpy.delete(result['times_' + str(elec)], elements)
            result['electrodes']            = numpy.delete(result['electrodes'], target)
            to_remove[to_remove > target]  -= 1
        return result

    templates      = load_data(params, 'templates')
    
    N_e, N_t, N_tm = templates.shape
    nb_temp        = N_tm/2
    merged         = [nb_temp, 0]
    mixtures       = []
    to_remove      = []

    myfile   = get_overlaps(comm, params, extension='-mixtures', erase=True, parallel_hdf5=parallel_hdf5, normalize=False, maxoverlap=False)
    filename = params.get('data', 'file_out_suff') + '.overlap-mixtures.hdf5'
    result   = []

    if comm.rank > 0:
        templates.file.close()
        myfile.close()
    else:
        norm_templates = numpy.zeros(templates.shape[2], dtype=numpy.float32)
        for i in xrange(templates.shape[2]):
            norm_templates[i] = numpy.sqrt(numpy.mean(numpy.mean(templates[:,:,i]**2,0),0))

        result    = load_data(params, 'clusters')
        best_elec = load_data(params, 'electrodes')
        limits    = load_data(params, 'limits')
        distances = numpy.zeros((nb_temp, nb_temp), dtype=numpy.float32)
        overlap   = myfile.get('overlap')
        for i in xrange(nb_temp):
            distances[i, i+1:] = numpy.argmax(overlap[i, i+1:nb_temp], 1)
            distances[i+1:, i] = distances[i, i+1:]

        import scipy.linalg
        overlap_0 = overlap[:, :, N_t]
        pbar      = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()], maxval=nb_temp).start()

        is_part_of_sum = []
        sorted_temp    = numpy.argsort(norm_templates[:nb_temp])[::-1]
        for count, k in enumerate(sorted_temp):
            idx_1      = set(numpy.where(best_elec == best_elec[k])[0])
            tmp_idx    = set(numpy.where(best_elec != best_elec[k])[0])
            idx_1      = numpy.array(list(idx_1.difference(mixtures + [k])))
            tmp_idx    = numpy.array(list(tmp_idx.difference(mixtures + [k])))
            electrodes = numpy.where(numpy.max(numpy.abs(templates[:, :, k]), axis=1) > 0)[0]
            idx_2      = []
            overlap_k  = overlap[k]
            for idx in tmp_idx:
                if best_elec[idx] in electrodes:
                    idx_2 += [idx]
            for i in idx_1:
                overlap_i = overlap[i]
                t1_vs_t1  = overlap_0[i, i]
                t_vs_t1   = overlap_k[i, distances[k, i]]
                for j in idx_2:
                    t2_vs_t2 = overlap_0[j, j]
                    t1_vs_t2 = overlap_i[j, distances[k, i] - distances[k, j]]
                    t_vs_t2  = overlap_k[j, distances[k, j]]
                    M        = numpy.vstack((numpy.hstack((t1_vs_t1, t1_vs_t2)), numpy.hstack((t1_vs_t2, t2_vs_t2))))
                    V        = numpy.hstack((t_vs_t1, t_vs_t2))
                    try:
                        [a1, a2] = numpy.dot(scipy.linalg.inv(M), V)
                    except Exception:
                        [a1, a2] = [0, 0]
                    a1_lim = limits[i]
                    a2_lim = limits[j]
                    is_a1  = (a1_lim[0] <= a1) and (a1 <= a1_lim[1])
                    is_a2  = (a2_lim[0] <= a2) and (a2 <= a2_lim[1])
                    if is_a1 and is_a2:
                        if k not in mixtures and k not in is_part_of_sum:
                            mixtures       += [k]
                            is_part_of_sum += [i, j]
            pbar.update(count)

        pbar.finish()
        templates.file.close()
        to_remove = numpy.array(mixtures)
        myfile.close()

    to_remove = comm.bcast(to_remove, root=0)

    if len(to_remove) > 0:
        slice_templates(comm, params, to_remove)
        if comm.rank == 0:
            result = remove(result, to_remove)
        slice_clusters(comm, params, result)

    if comm.rank == 0:
        os.remove(filename)

    return [nb_temp, len(mixtures)]

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

    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    ine, ire, ife = numpy.array([[], [], []], dtype=numpy.int32)
    if not edge:
        ine = numpy.where((numpy.hstack((dx, 0)) < 0) & (numpy.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = numpy.where((numpy.hstack((dx, 0)) <= 0) & (numpy.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = numpy.where((numpy.hstack((dx, 0)) < 0) & (numpy.hstack((0, dx)) >= 0))[0]
    ind = numpy.unique(numpy.hstack((ine, ire, ife)))
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
        if valley:
            x = -x
        pylab.plot(ind, x[ind], 'ro')
        pylab.plot(x, 'k')

    return ind