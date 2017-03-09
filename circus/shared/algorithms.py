import os, logging
import scipy.optimize, numpy, pylab, scipy.spatial.distance, scipy.stats
from circus.shared.files import load_data, write_datasets, get_overlaps
from circus.shared.utils import get_tqdm_progressbar
from circus.shared.messages import print_and_log
from circus.shared.probes import get_nodes_and_edges
from circus.shared.mpi import all_gather_array, SHARED_MEMORY, comm
import scipy.linalg, scipy.sparse

logger = logging.getLogger(__name__)

def distancematrix(data, ydata=None):
    
    if ydata is None:
        distances = scipy.spatial.distance.pdist(data, 'euclidean')
    else:
        distances = scipy.spatial.distance.cdist(data, ydata, 'euclidean')

    return distances.astype(numpy.float32)



def fit_rho_delta(xdata, ydata, smart_select=False, display=False, max_clusters=10, save=False):

    if smart_select:

        xmax = xdata.max()
        off  = ydata.min()
        idx  = numpy.argmin(xdata)
        a_0  = (ydata[idx] - off)/numpy.log(1 + (xmax - xdata[idx]))

        def myfunc(x, a, b, c, d):
            return a*numpy.log(1. + c*((xmax - x)**b)) + d

        def imyfunc(x, a, b, c, d):
            return numpy.exp(d)*((1. + c*(xmax - x)**b)**a)

        try:
            result, pcov = scipy.optimize.curve_fit(myfunc, xdata, ydata, p0=[a_0, 1., 1., off])
            prediction   = myfunc(xdata, result[0], result[1], result[2], result[3])
            difference   = xdata*(ydata - prediction)
            z_score      = (difference - difference.mean())/difference.std()
            subidx       = numpy.where(z_score >= 3.)[0]
        except Exception:
            subidx = numpy.argsort(xdata*numpy.log(1 + ydata))[::-1][:max_clusters]
        
    else:
        subidx = numpy.argsort(xdata*numpy.log(1 + ydata))[::-1][:max_clusters]

    if display:
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.plot(xdata, ydata, 'ko')
        ax.plot(xdata[subidx[:len(subidx)]], ydata[subidx[:len(subidx)]], 'ro')
        if smart_select:
            idx = numpy.argsort(xdata)
            ax.plot(xdata[idx], prediction[idx], 'r')
            ax.set_yscale('log')
        if save:
            pylab.savefig(os.path.join(save[0], 'rho_delta_%s.png' %(save[1])))
            pylab.close()
        else:
            pylab.show()
    return subidx, len(subidx)


def rho_estimation(data, update=None, compute_rho=True, mratio=0.01):

    N     = len(data)
    rho   = numpy.zeros(N, dtype=numpy.float32)
        
    if update is None:
        dist = distancematrix(data)
        didx = lambda i,j: i*N + j - i*(i+1)//2 - i - 1
        nb_selec = max(5, int(mratio*N))
        sdist    = numpy.zeros((N, nb_selec), dtype=numpy.float32)  
    
        if compute_rho:
            for i in xrange(N):
                indices  = numpy.concatenate((didx(i, numpy.arange(i+1, N)), didx(numpy.arange(0, i-1), i)))
                tmp      = numpy.argsort(numpy.take(dist, indices))[:nb_selec]
                sdist[i] = numpy.take(dist, numpy.take(indices, tmp))
                rho[i]   = numpy.sum(sdist[i])  

    else:
        M        = len(update[0])
        nb_selec = max(5, int(mratio*M))
        sdist    = numpy.zeros((N, nb_selec), dtype=numpy.float32)  

        for i in xrange(N):
            dist     = distancematrix(data[i].reshape(1, len(data[i])), update[0]).ravel()
            all_dist = numpy.concatenate((dist, update[1][i]))
            idx      = numpy.argsort(all_dist)[:nb_selec]
            sdist[i] = numpy.take(all_dist, idx)
            rho[i]   = numpy.sum(sdist[i])
    return rho, dist, sdist, nb_selec


def clustering(rho, dist, mratio=0.1, smart_select=False, display=None, n_min=None, max_clusters=10, save=False):

    N                 = len(rho)
    maxd              = numpy.max(dist)
    didx              = lambda i,j: i*N + j - i*(i+1)//2 - i - 1
    ordrho            = numpy.argsort(rho)[::-1]
    delta, nneigh     = numpy.zeros(N, dtype=numpy.float32), numpy.zeros(N, dtype=numpy.int32)
    delta[ordrho[0]]  = -1
    for ii in xrange(1, N):
        delta[ordrho[ii]] = maxd
        for jj in xrange(ii):
            if ordrho[jj] > ordrho[ii]:
                xdist = dist[didx(ordrho[ii], ordrho[jj])]
            else:
                xdist = dist[didx(ordrho[jj], ordrho[ii])]

            if xdist < delta[ordrho[ii]]:
                delta[ordrho[ii]]  = xdist
                nneigh[ordrho[ii]] = ordrho[jj]

    delta[ordrho[0]]        = delta.max()
    clust_idx, max_clusters = fit_rho_delta(rho, delta, smart_select=smart_select, max_clusters=max_clusters)

    def assign_halo(idx):
        cl      = numpy.empty(N, dtype=numpy.int32)
        cl[:]   = -1
        NCLUST  = len(idx)
        cl[idx] = numpy.arange(NCLUST)
        
        # assignation
        for i in xrange(N):
            if cl[ordrho[i]] == -1:
                cl[ordrho[i]] = cl[nneigh[ordrho[i]]]
        
        # halo (ignoring outliers ?)
        halo = cl.copy()
        
        if n_min is not None:

            for cluster in xrange(NCLUST):
                idx = numpy.where(halo == cluster)[0]
                if len(idx) < n_min:
                    halo[idx] = -1
                    NCLUST   -= 1
        return halo, NCLUST

    halo, NCLUST = assign_halo(clust_idx[:max_clusters+1])

    return halo, rho, delta, clust_idx[:max_clusters]


def merging(groups, sim_same_elec, data):

    def perform_merging(groups, sim_same_elec, data):
        mask      = numpy.where(groups > -1)[0]
        clusters  = numpy.unique(groups[mask])
        dmin      = numpy.inf
        to_merge  = [None, None]
        
        for ic1 in xrange(len(clusters)):
            idx1 = numpy.where(groups == clusters[ic1])[0]
            sd1  = numpy.take(data, idx1, axis=0)
            m1   = numpy.median(sd1, 0)
            for ic2 in xrange(ic1+1, len(clusters)):
                idx2 = numpy.where(groups == clusters[ic2])[0]
                sd2  = numpy.take(data, idx2, axis=0)
                m2   = numpy.median(sd2, 0)
                v_n  = m1 - m2      
                pr_1 = numpy.dot(sd1, v_n)
                pr_2 = numpy.dot(sd2, v_n)

                norm = numpy.median(numpy.abs(pr_1 - numpy.median(pr_1)))**2 + numpy.median(numpy.abs(pr_2 - numpy.median(pr_2)))**2
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

def slice_templates(params, to_remove=[], to_merge=[], extension=''):

    import shutil, h5py
    file_out_suff  = params.get('data', 'file_out_suff')

    data_file      = params.data_file
    N_e            = params.getint('data', 'N_e')
    N_total        = params.nb_channels
    N_t            = params.getint('detection', 'N_t')
    template_shift = params.getint('detection', 'template_shift')

    if comm.rank == 0:
        print_and_log(['Node 0 is slicing templates'], 'debug', logger)
        old_templates  = load_data(params, 'templates')
        old_limits     = load_data(params, 'limits')
        x, N_tm        = old_templates.shape
        norm_templates = load_data(params, 'norm-templates')

        if to_merge != []:
            for count in xrange(len(to_merge)):
                remove     = to_merge[count][1]
                to_remove += [remove]

        all_templates = set(numpy.arange(N_tm//2))
        to_keep       = numpy.array(list(all_templates.difference(to_remove)))
    
        positions  = numpy.arange(len(to_keep))

        local_keep = to_keep[positions]
        templates  = scipy.sparse.lil_matrix((N_e*N_t, 2*len(to_keep)), dtype=numpy.float32)
        hfile      = h5py.File(file_out_suff + '.templates-new.hdf5', 'w', libver='latest')
        norms      = hfile.create_dataset('norms', shape=(2*len(to_keep), ), dtype=numpy.float32, chunks=True)
        limits     = hfile.create_dataset('limits', shape=(len(to_keep), 2), dtype=numpy.float32, chunks=True)
        for count, keep in zip(positions, local_keep):

            templates[:, count]                = old_templates[:, keep]
            templates[:, count + len(to_keep)] = old_templates[:, keep + N_tm//2]
            norms[count]                       = norm_templates[keep]
            norms[count + len(to_keep)]        = norm_templates[keep + N_tm//2]
            if to_merge == []:
                new_limits = old_limits[keep]
            else:
                subset     = numpy.where(to_merge[:, 0] == keep)[0]
                if len(subset) > 0:
                    idx        = numpy.unique(to_merge[subset].flatten())
                    ratios     = norm_templates[idx]/norm_templates[keep]
                    new_limits = [numpy.min(ratios*old_limits[idx][:, 0]), numpy.max(ratios*old_limits[idx][:, 1])]
                else:
                    new_limits = old_limits[keep]
            limits[count]  = new_limits
        

        templates = templates.tocoo()
        hfile.create_dataset('temp_x', data=templates.row)
        hfile.create_dataset('temp_y', data=templates.col)
        hfile.create_dataset('temp_data', data=templates.data)
        hfile.create_dataset('temp_shape', data=numpy.array([N_e, N_t, 2*len(to_keep)], dtype=numpy.int32))
        hfile.close()

        if os.path.exists(file_out_suff + '.templates%s.hdf5' %extension):
            os.remove(file_out_suff + '.templates%s.hdf5' %extension)
        shutil.move(file_out_suff + '.templates-new.hdf5', file_out_suff + '.templates%s.hdf5' %extension)

    comm.Barrier()

    

def slice_clusters(params, result, to_remove=[], to_merge=[], extension='', light=False):
    
    import h5py, shutil
    file_out_suff  = params.get('data', 'file_out_suff')
    data_file      = params.data_file
    N_e            = params.getint('data', 'N_e')
    N_total        = params.nb_channels
    N_t            = params.getint('detection', 'N_t')
    template_shift = params.getint('detection', 'template_shift')

    if comm.rank == 0:

        print_and_log(['Node 0 is slicing clusters'], 'debug', logger)

        if to_merge != []:
            for count in xrange(len(to_merge)):
                remove     = to_merge[count][1]
                to_remove += [remove]

        all_elements = [[] for i in xrange(N_e)]
        for target in numpy.unique(to_remove):
            elec     = result['electrodes'][target]
            nic      = target - numpy.where(result['electrodes'] == elec)[0][0]
            mask     = result['clusters_' + str(elec)] > -1
            tmp      = numpy.unique(result['clusters_' + str(elec)][mask])
            all_elements[elec] += list(numpy.where(result['clusters_' + str(elec)] == tmp[nic])[0])
                    
        for elec in xrange(N_e):
            if not light:
                result['data_' + str(elec)]     = numpy.delete(result['data_' + str(elec)], all_elements[elec], axis=0)
                result['clusters_' + str(elec)] = numpy.delete(result['clusters_' + str(elec)], all_elements[elec]) 
                result['times_' + str(elec)]    = numpy.delete(result['times_' + str(elec)], all_elements[elec])
                result['peaks_' + str(elec)]    = numpy.delete(result['peaks_' + str(elec)], all_elements[elec])
            else:
                
                result['clusters_' + str(elec)] = numpy.delete(result['clusters_' + str(elec)], all_elements[elec]) 
                myfile = h5py.File(file_out_suff + '.clusters.hdf5', 'r', libver='latest')
                data   = myfile.get('data_' + str(elec))[:]
                result['data_' + str(elec)]  = numpy.delete(data, all_elements[elec], axis=0)                
                data   = myfile.get('times_' + str(elec))[:]
                result['times_' + str(elec)] = numpy.delete(data, all_elements[elec])
                data   = myfile.get('peaks_' + str(elec))[:]
                result['peaks_' + str(elec)] = numpy.delete(data, all_elements[elec])
                myfile.close()

        result['electrodes'] = numpy.delete(result['electrodes'], numpy.unique(to_remove))

        cfile    = h5py.File(file_out_suff + '.clusters-new.hdf5', 'w', libver='latest')
        to_write = ['data_', 'clusters_', 'times_', 'peaks_'] 
        for ielec in xrange(N_e):
            write_datasets(cfile, to_write, result, ielec)
       
        write_datasets(cfile, ['electrodes'], result)
        cfile.close()
        if os.path.exists(file_out_suff + '.clusters%s.hdf5' %extension):
            os.remove(file_out_suff + '.clusters%s.hdf5' %extension)
        shutil.move(file_out_suff + '.clusters-new.hdf5', file_out_suff + '.clusters%s.hdf5' %extension)

    comm.Barrier()


def slice_result(result, times):

    sub_results = []

    nb_temp = len(result['spiketimes'])
    for t in times:
        sub_result = {'spiketimes' : {}, 'amplitudes' : {}}
        for key in result['spiketimes'].keys():
            idx = numpy.where((result['spiketimes'][key] >= t[0]) & (result['spiketimes'][key] <= t[1]))[0]
            sub_result['spiketimes'][key] = result['spiketimes'][key][idx] - t[0]
            sub_result['amplitudes'][key] = result['amplitudes'][key][idx]                
        sub_results += [sub_result]

    return sub_results

def merging_cc(params, nb_cpu, nb_gpu, use_gpu):


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
                result['times_' + str(elec)]    = numpy.delete(result['times_' + str(elec)], elements)
                result['peaks_' + str(elec)]    = numpy.delete(result['peaks_' + str(elec)], elements)
                result['electrodes']            = numpy.delete(result['electrodes'], to_remove)
                distances                       = numpy.delete(distances, to_remove, axis=0)
                distances                       = numpy.delete(distances, to_remove, axis=1)
                to_merge                        = numpy.vstack((to_merge, numpy.array([g_idx[to_keep], g_idx[to_remove]])))
                g_idx.pop(to_remove)

        return to_merge, result
         
    data_file      = params.data_file
    N_e            = params.getint('data', 'N_e')
    N_total        = params.nb_channels
    N_t            = params.getint('detection', 'N_t')
    template_shift = params.getint('detection', 'template_shift')

    templates      = load_data(params, 'templates')
    x,        N_tm = templates.shape
    nb_temp        = N_tm//2
    to_merge       = []
    cc_merge       = params.getfloat('clustering', 'cc_merge')
        
    result   = []
    overlap  = get_overlaps(params, extension='-merging', erase=True, normalize=True, maxoverlap=False, verbose=False, half=True, use_gpu=use_gpu, nb_cpu=nb_cpu, nb_gpu=nb_gpu)
    filename = params.get('data', 'file_out_suff') + '.overlap-merging.hdf5'

    if comm.rank > 0:
        overlap.file.close()
    else:
        over_x     = overlap.get('over_x')[:]
        over_y     = overlap.get('over_y')[:]
        over_data  = overlap.get('over_data')[:]
        over_shape = overlap.get('over_shape')[:]
        overlap.close()

        overlap   = scipy.sparse.csr_matrix((over_data, (over_x, over_y)), shape=over_shape)
        result    = load_data(params, 'clusters')
        distances = numpy.zeros((nb_temp, nb_temp), dtype=numpy.float32)
        for i in xrange(nb_temp-1):
            distances[i, i+1:] = numpy.max(overlap[i*nb_temp+i+1:(i+1)*nb_temp].toarray(), 1)
            distances[i+1:, i] = distances[i, i+1:]

        distances /= (N_e*N_t)
        to_merge, result = remove(result, distances, cc_merge)       

    to_merge = numpy.array(to_merge)
    to_merge = comm.bcast(to_merge, root=0)
    
    if len(to_merge) > 0:
        slice_templates(params, to_merge=to_merge)
        slice_clusters(params, result)

    if comm.rank == 0:
        os.remove(filename)

    return [nb_temp, len(to_merge)]


def delete_mixtures(params, nb_cpu, nb_gpu, use_gpu):
        
    templates      = load_data(params, 'templates')
    
    data_file      = params.data_file
    N_e            = params.getint('data', 'N_e')
    N_total        = params.nb_channels
    N_t            = params.getint('detection', 'N_t')
    template_shift = params.getint('detection', 'template_shift')
    cc_merge       = params.getfloat('clustering', 'cc_merge')
    x,        N_tm = templates.shape
    nb_temp        = N_tm//2
    merged         = [nb_temp, 0]
    mixtures       = []
    to_remove      = []

    overlap  = get_overlaps(params, extension='-mixtures', erase=True, normalize=False, maxoverlap=False, verbose=False, half=True, use_gpu=use_gpu, nb_cpu=nb_cpu, nb_gpu=nb_gpu)
    filename = params.get('data', 'file_out_suff') + '.overlap-mixtures.hdf5'
    result   = []
    
    norm_templates   = load_data(params, 'norm-templates')
    templates        = load_data(params, 'templates')
    result           = load_data(params, 'clusters')
    best_elec        = load_data(params, 'electrodes')
    limits           = load_data(params, 'limits')
    nodes, edges     = get_nodes_and_edges(params)
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)

    distances = numpy.zeros((nb_temp, nb_temp), dtype=numpy.float32)

    over_x     = overlap.get('over_x')[:]
    over_y     = overlap.get('over_y')[:]
    over_data  = overlap.get('over_data')[:]
    over_shape = overlap.get('over_shape')[:]
    overlap.close()

    overlap    = scipy.sparse.csr_matrix((over_data, (over_x, over_y)), shape=over_shape)

    for i in xrange(nb_temp-1):
        distances[i, i+1:] = numpy.argmax(overlap[i*nb_temp+i+1:(i+1)*nb_temp].toarray(), 1)
        distances[i+1:, i] = distances[i, i+1:]

    all_temp  = numpy.arange(comm.rank, nb_temp, comm.size)
    overlap_0 = overlap[:, N_t].toarray().reshape(nb_temp, nb_temp)


    sorted_temp    = numpy.argsort(norm_templates[:nb_temp])[::-1][comm.rank::comm.size]
    M              = numpy.zeros((2, 2), dtype=numpy.float32)
    V              = numpy.zeros((2, 1), dtype=numpy.float32)

    to_explore = xrange(comm.rank, len(sorted_temp), comm.size)
    if comm.rank == 0:
        to_explore = get_tqdm_progressbar(to_explore)


    for count, k in enumerate(to_explore):

        k             = sorted_temp[k]
        electrodes    = numpy.take(inv_nodes, edges[nodes[best_elec[k]]])
        overlap_k     = overlap[k*nb_temp:(k+1)*nb_temp].tolil()
        is_in_area    = numpy.in1d(best_elec, electrodes)
        all_idx       = numpy.arange(len(best_elec))[is_in_area]
        been_found    = False

        for i in all_idx:
            if not been_found:
                overlap_i = overlap[i*nb_temp:(i+1)*nb_temp].tolil()
                M[0, 0]   = overlap_0[i, i]
                V[0, 0]   = overlap_k[i, distances[k, i]]
                for j in all_idx[i+1:]:
                    M[1, 1]  = overlap_0[j, j]
                    M[1, 0]  = overlap_i[j, distances[k, i] - distances[k, j]]
                    M[0, 1]  = M[1, 0]
                    V[1, 0]  = overlap_k[j, distances[k, j]]
                    try:
                        [a1, a2] = numpy.dot(scipy.linalg.inv(M), V)
                    except Exception:
                        [a1, a2] = [0, 0]
                    a1_lim   = limits[i]
                    a2_lim   = limits[j]
                    is_a1    = (a1_lim[0] <= a1) and (a1 <= a1_lim[1])
                    is_a2    = (a2_lim[0] <= a2) and (a2 <= a2_lim[1])
                    if is_a1 and is_a2:
                        new_template = (a1*templates[:, i].toarray() + a2*templates[:, j].toarray()).ravel()
                        similarity   = numpy.corrcoef(templates[:, k].toarray().ravel(), new_template)[0, 1]
                        if similarity > cc_merge:
                            if k not in mixtures:
                                mixtures  += [k]
                                been_found = True 
                                break
                                #print "Template", k, 'is sum of (%d, %g) and (%d,%g)' %(i, a1, j, a2)
    
    #print mixtures
    to_remove = numpy.unique(numpy.array(mixtures, dtype=numpy.int32))    
    to_remove = all_gather_array(to_remove, comm, 0, dtype='int32')
    
    if len(to_remove) > 0:
        slice_templates(params, to_remove)
        slice_clusters(params, result, to_remove=to_remove)

    if comm.rank == 0:
        os.remove(filename)

    return [nb_temp, len(to_remove)]

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
