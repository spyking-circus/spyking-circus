import os, logging, sys
import scipy.optimize, numpy, pylab, scipy.spatial.distance, scipy.stats
import shutil, h5py
import scipy.linalg, scipy.sparse

from circus.shared.files import load_data, write_datasets, get_overlaps, load_data_memshared
from circus.shared.utils import get_tqdm_progressbar, get_shared_memory_flag
from circus.shared.messages import print_and_log
from circus.shared.probes import get_nodes_and_edges
from circus.shared.mpi import all_gather_array, comm, gather_array, get_local_ring
import scipy.linalg, scipy.sparse
import statsmodels.api as sm
#import sklearn.metrics
from statsmodels.sandbox.regression.predstd import wls_prediction_std

logger = logging.getLogger(__name__)


def distancematrix(data, ydata=None):

    if ydata is None:
        distances = scipy.spatial.distance.pdist(data, 'euclidean')
    else:
        distances = scipy.spatial.distance.cdist(data, ydata, 'euclidean')

    return distances.astype(numpy.float32)

def fit_rho_delta(xdata, ydata, alpha=3):

    x = sm.add_constant(xdata)
    model = sm.RLM(ydata, x)
    results = model.fit()
    difference = ydata - results.fittedvalues
    factor = numpy.median(numpy.abs(difference - numpy.median(difference)))
    z_score = difference - alpha*factor*(1 + results.fittedvalues)
    centers = numpy.where(z_score >= 0)[0]
    return centers


def compute_rho(data, update=None, mratio=0.01):
    N = len(data)
    nb_selec = max(5, int(mratio*N))

    if update is None:
        dist = distancematrix(data, data)
        #dist = sklearn.metrics.pairwise_distances(data, data, n_jobs=1)
        dist_sorted = numpy.sort(dist, axis=1) # sorting each row in ascending order
        dist_sorted = dist_sorted[:, 1:nb_selec+1]
        rho = numpy.mean(dist_sorted, axis=1) # density computation
        dist = scipy.spatial.distance.squareform(dist, checks=False)
        return rho, dist, dist_sorted
    else:
        dist = distancematrix(data, update[0])
        #dist = sklearn.metrics.pairwise_distances(data, update[0], n_jobs=1)
        dist = numpy.hstack((update[1], dist))
        dist = numpy.sort(dist, axis=1) # sorting each row in ascending order
        dist = dist[:, 1:nb_selec+1]
        rho = numpy.mean(dist, axis=1) # density computation
        return rho, dist

def clustering_by_density(rho, dist, n_min, alpha=3):
    dist = scipy.spatial.distance.squareform(dist, checks=False)
    delta = compute_delta(dist, rho)
    nclus, labels, centers = find_centroids_and_cluster(dist, rho, delta, n_min, alpha)
    halolabels = halo_assign(dist, labels, centers)
    halolabels -= 1
    centers = numpy.where(numpy.in1d(centers - 1, numpy.arange(halolabels.max() + 1)))[0]
    return halolabels, rho, delta, centers

def compute_delta(dist, rho):
    rho_sort_id = numpy.argsort(rho) # index to sort
    rho_sort_id = (rho_sort_id[::-1]) # reversing sorting indexes
    sort_rho = rho[rho_sort_id] # sortig rho in ascending order
    gtmat = numpy.greater_equal(sort_rho, sort_rho[:, None]) # gtmat(i,j)=1 if rho(i)>=rho(j) and 0 otherwise
    
    sortdist = numpy.zeros_like(dist)
    sortdist = dist[rho_sort_id, :]
    sortdist = sortdist[:, rho_sort_id]    
    sortdist *= gtmat # keeping only distance to points with highest or equal rho 
    sortdist[sortdist == 0] = float("inf")

    auxdelta = numpy.min(sortdist, axis=1)
    delta = numpy.zeros_like(auxdelta) 
    delta[rho_sort_id] = auxdelta 
    delta[rho == numpy.max(rho)] = numpy.max(delta[numpy.logical_not(numpy.isinf(delta))]) # assigns max delta to the max rho
    delta[numpy.isinf(delta)] = 0
    return delta

def find_centroids_and_cluster(dist, rho, delta, n_min, alpha=3):

    npnts = len(rho)    
    centers = numpy.zeros((npnts))
    
    auxid = fit_rho_delta(rho, delta, alpha)
    nclus = len(auxid)

    centers[auxid] = numpy.arange(nclus) + 1 # assigning labels to centroids
    
    # assigning points to clusters based on their distance to the centroids
    if nclus <= 1:
        labels = numpy.ones(npnts)
    else:
        centersx = numpy.where(centers)[0] # index of centroids
        dist2cent = dist[centersx, :]
        labels = numpy.argmin(dist2cent, axis=0) + 1
        _, cluscounts = numpy.unique(labels, return_counts=True) # number of elements of each cluster
        
        small_clusters = numpy.where(cluscounts < n_min)[0] # index of 1 or 0 members clusters

        if len(small_clusters) > 0: # if there one or more 1 or 0 member cluster # if there one or more 1 or 0 member cluster
            cluslab = centers[centersx] # cluster labels
            id2rem = numpy.where(numpy.in1d(cluslab, small_clusters))[0] # ids to remove
            clusidx = numpy.delete(centersx, id2rem) # removing
            centers = numpy.zeros(len(centers))
            nclus = nclus - len(id2rem)
            centers[clusidx] = numpy.arange(nclus) + 1 # re labeling centroids            
            dist2cent = dist[centersx, :]# re compute distances from centroid to any other point
            labels = numpy.argmin(dist2cent, axis=0) + 1 # re assigns clusters 
            
    return nclus, labels, centers
    

def halo_assign(dist, labels, centers):

    halolabels = labels.copy()    
    sameclusmat = numpy.equal(labels, labels[:, None]) #
    sameclus_cent = sameclusmat[centers > 0, :] # selects only centroids
    dist2cent = dist[centers > 0, :] # distance to centroids
    dist2cluscent = dist2cent*sameclus_cent # preserves only distances to the corresponding cluster centroid
    nclusmem = numpy.sum(sameclus_cent, axis=1) # number of cluster members
        
    meandist2cent = numpy.sum(dist2cluscent, axis=1)/nclusmem # mean distance to corresponding centroid
    gt_meandist2cent = numpy.greater(dist2cluscent, meandist2cent[:, None]) # greater than the mean dist to centroid
    remids = numpy.sum(gt_meandist2cent, axis=0)
    halolabels[remids > 0] = 0 # setting to 0 the removes points
    return halolabels


def estimate_threshold(x, y, alpha):
    x = sm.add_constant(x)
    model = sm.OLS(y,x)
    results = model.fit()
    _,_,threshold = wls_prediction_std(results, alpha=alpha)
    return threshold


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
                dist = numpy.sum(v_n**2)/numpy.sqrt(norm)

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


def slice_templates(params, to_remove=[], to_merge=[], extension='',
    input_extension=''):
    """Slice templates in HDF5 file.

    Arguments:
        params
        to_remove: list (optional)
            An array of template indices to remove.
            The default value is [].
        to_merge: list | numpy.ndarray (optional)
            An array of pair of template indices to merge
            (i.e. shape = (nb_merges, 2)).
            The default value is [].
        extension: string (optional)
            The extension to use as output.
            The default value is ''.
        input_extension: string (optional)
            The extension to use as input.
            The default value is ''.
    """

    file_out_suff  = params.get('data', 'file_out_suff')

    data_file      = params.data_file
    N_e            = params.getint('data', 'N_e')
    N_total        = params.nb_channels
    hdf5_compress  = params.getboolean('data', 'hdf5_compress')
    N_t            = params.getint('detection', 'N_t')
    template_shift = params.getint('detection', 'template_shift')

    if comm.rank == 0:
        print_and_log(['Node 0 is slicing templates'], 'debug', logger)
        old_templates  = load_data(params, 'templates', extension=input_extension)
        old_limits     = load_data(params, 'limits', extension=input_extension)
        _, N_tm        = old_templates.shape
        norm_templates = load_data(params, 'norm-templates', extension=input_extension)

        # Determine the template indices to delete.
        to_delete = list(to_remove)  # i.e. copy
        if to_merge != []:
            for count in xrange(len(to_merge)):
                remove = to_merge[count][1]
                to_delete += [remove]

        # Determine the indices to keep.
        all_templates = set(numpy.arange(N_tm // 2))
        to_keep = numpy.array(list(all_templates.difference(to_delete)))

        positions = numpy.arange(len(to_keep))

        # Initialize new HDF5 file for templates.
        local_keep = to_keep[positions]
        templates  = scipy.sparse.lil_matrix((N_e*N_t, 2*len(to_keep)), dtype=numpy.float32)
        hfilename  = file_out_suff + '.templates{}.hdf5'.format('-new')
        hfile      = h5py.File(hfilename, 'w', libver='earliest')
        norms      = hfile.create_dataset('norms', shape=(2*len(to_keep), ), dtype=numpy.float32, chunks=True)
        limits     = hfile.create_dataset('limits', shape=(len(to_keep), 2), dtype=numpy.float32, chunks=True)
        # For each index to keep.
        for count, keep in zip(positions, local_keep):
            # Copy template.
            templates[:, count]                = old_templates[:, keep]
            templates[:, count + len(to_keep)] = old_templates[:, keep + N_tm//2]
            # Copy norm.
            norms[count]                       = norm_templates[keep]
            norms[count + len(to_keep)]        = norm_templates[keep + N_tm//2]
            # Copy limits.
            if to_merge == []:
                new_limits = old_limits[keep]
            else:
                subset     = numpy.where(to_merge[:, 0] == keep)[0]
                if len(subset) > 0:
                    # Index to keep is involved in merge(s) and limits need to
                    # be updated.
                    idx        = numpy.unique(to_merge[subset].flatten())
                    ratios     = norm_templates[idx] / norm_templates[keep]
                    new_limits = [
                        numpy.min(ratios * old_limits[idx][:, 0]),
                        numpy.max(ratios * old_limits[idx][:, 1])
                    ]
                else:
                    new_limits = old_limits[keep]
            limits[count]  = new_limits

        # Copy templates to file.
        templates = templates.tocoo()
        if hdf5_compress:
            hfile.create_dataset('temp_x', data=templates.row, compression='gzip')
            hfile.create_dataset('temp_y', data=templates.col, compression='gzip')
            hfile.create_dataset('temp_data', data=templates.data, compression='gzip')
        else:
            hfile.create_dataset('temp_x', data=templates.row)
            hfile.create_dataset('temp_y', data=templates.col)
            hfile.create_dataset('temp_data', data=templates.data)
        hfile.create_dataset('temp_shape', data=numpy.array([N_e, N_t, 2*len(to_keep)], dtype=numpy.int32))
        hfile.close()

        # Rename output filename.
        temporary_path = hfilename
        output_path = file_out_suff + '.templates{}.hdf5'.format(extension)
        if os.path.exists(output_path):
            os.remove(output_path)
        shutil.move(temporary_path, output_path)
    else:
        to_keep = numpy.array([])

    return to_keep


def slice_clusters(params, result, to_remove=[], to_merge=[], extension='',
    input_extension='', light=False, method='safe'):
    """Slice clusters in HDF5 templates.

    Arguments:
        params
        to_remove: list (optional)
        to_merge: list | numpy.ndarray (optional)
        extension: string (optional)
            The default value is ''.
        input_extension: string (optional)
            The default value is ''.
        light: boolean (optional)
    """

    file_out_suff  = params.get('data', 'file_out_suff')
    data_file      = params.data_file
    N_e            = params.getint('data', 'N_e')
    N_total        = params.nb_channels
    hdf5_compress  = params.getboolean('data', 'hdf5_compress')
    N_t            = params.getint('detection', 'N_t')
    template_shift = params.getint('detection', 'template_shift')

    if comm.rank == 0:

        print_and_log(['Node 0 is slicing clusters'], 'debug', logger)
        old_templates = load_data(params, 'templates', extension=input_extension)
        _, N_tm = old_templates.shape

        # Determine the template indices to delete.
        to_delete = list(to_remove)
        if to_merge != []:
            for count in xrange(len(to_merge)):
                remove     = to_merge[count][1]
                to_delete += [remove]

        # Determine the indices to keep.
        all_templates = set(numpy.arange(N_tm//2))
        to_keep = numpy.array(list(all_templates.difference(to_delete)))

        all_elements = [[] for i in xrange(N_e)]
        for target in numpy.unique(to_delete):
            elec     = result['electrodes'][target]
            nic      = target - numpy.where(result['electrodes'] == elec)[0][0]
            mask     = result['clusters_' + str(elec)] > -1
            tmp      = numpy.unique(result['clusters_' + str(elec)][mask])
            all_elements[elec] += list(numpy.where(result['clusters_' + str(elec)] == tmp[nic])[0])

        myfilename = file_out_suff + '.clusters{}.hdf5'.format(input_extension)
        myfile = h5py.File(myfilename, 'r', libver='earliest')

        for elec in xrange(N_e):
            if not light:
                result['data_' + str(elec)]     = numpy.delete(result['data_' + str(elec)], all_elements[elec], axis=0)
                result['clusters_' + str(elec)] = numpy.delete(result['clusters_' + str(elec)], all_elements[elec])
                result['times_' + str(elec)]    = numpy.delete(result['times_' + str(elec)], all_elements[elec])
                result['peaks_' + str(elec)]    = numpy.delete(result['peaks_' + str(elec)], all_elements[elec])
            else:
                result['clusters_' + str(elec)] = numpy.delete(result['clusters_' + str(elec)], all_elements[elec])
                data   = myfile.get('data_' + str(elec))[:]
                result['data_' + str(elec)]  = numpy.delete(data, all_elements[elec], axis=0)
                data   = myfile.get('times_' + str(elec))[:]
                result['times_' + str(elec)] = numpy.delete(data, all_elements[elec])
                data   = myfile.get('peaks_' + str(elec))[:]
                result['peaks_' + str(elec)] = numpy.delete(data, all_elements[elec])

        myfile.close()
        if method == 'safe':
            result['electrodes'] = numpy.delete(result['electrodes'], numpy.unique(to_delete))
        elif method == 'new':
            result['electrodes'] = result['electrodes'][to_keep]
        else:
            raise ValueError("Unexpected method value: {}".format(method))

        cfilename = file_out_suff + '.clusters{}.hdf5'.format('-new')
        cfile    = h5py.File(cfilename, 'w', libver='earliest')
        to_write = ['data_', 'clusters_', 'times_', 'peaks_']
        for ielec in xrange(N_e):
            write_datasets(cfile, to_write, result, ielec, compression=hdf5_compress)
        write_datasets(cfile, ['electrodes'], result)
        cfile.close()

        # Rename output file.
        temporary_path = cfilename
        output_path = file_out_suff + '.clusters{}.hdf5'.format(extension)
        if os.path.exists(output_path):
            os.remove(output_path)
        shutil.move(temporary_path, output_path)

    return


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
    blosc_compress = params.getboolean('data', 'blosc_compress')

    N_tm           = load_data(params, 'nb_templates')
    nb_temp        = int(N_tm//2)
    to_merge       = []
    cc_merge       = params.getfloat('clustering', 'cc_merge')
    norm           = N_e * N_t

    result   = []
    overlap  = get_overlaps(params, extension='-merging', erase=True, normalize=True, maxoverlap=False, verbose=False, half=True, use_gpu=use_gpu, nb_cpu=nb_cpu, nb_gpu=nb_gpu)
    overlap.close()
    filename = params.get('data', 'file_out_suff') + '.overlap-merging.hdf5'

    SHARED_MEMORY = get_shared_memory_flag(params)

    if not SHARED_MEMORY:
        over_x, over_y, over_data, over_shape = load_data(params, 'overlaps-raw', extension='-merging')
    else:
        over_x, over_y, over_data, over_shape = load_data_memshared(params, 'overlaps-raw', extension='-merging', use_gpu=use_gpu, nb_cpu=nb_cpu, nb_gpu=nb_gpu)

    #sub_comm, is_local = get_local_ring(True)

    #if is_local:

    distances = numpy.zeros((nb_temp, nb_temp), dtype=numpy.float32)

    to_explore = numpy.arange(nb_temp - 1)[comm.rank::comm.size]

    for i in to_explore:

        idx = numpy.where((over_x >= i*nb_temp+i+1) & (over_x < ((i+1)*nb_temp)))[0]
        local_x = over_x[idx] - (i*nb_temp+i+1)
        data = numpy.zeros((nb_temp - (i + 1), over_shape[1]), dtype=numpy.float32)
        data[local_x, over_y[idx]] = over_data[idx]
        distances[i, i+1:] = numpy.max(data, 1)/norm
        distances[i+1:, i] = distances[i, i+1:]

    #Now we need to sync everything across nodes
    distances = gather_array(distances, comm, 0, 1, 'float32', compress=blosc_compress)
    if comm.rank == 0:
        distances = distances.reshape(comm.size, nb_temp, nb_temp)
        distances = numpy.sum(distances, 0)

    #sub_comm.Barrier()
    #sub_comm.Free()

    if comm.rank == 0:
        result = load_data(params, 'clusters')
        to_merge, result = remove(result, distances, cc_merge)

    to_merge = numpy.array(to_merge)
    to_merge = comm.bcast(to_merge, root=0)

    if len(to_merge) > 0:
        slice_templates(params, to_merge=to_merge)
        slice_clusters(params, result)

    comm.Barrier()

    if comm.rank == 0:
        os.remove(filename)

    return [nb_temp, len(to_merge)]


def delete_mixtures(params, nb_cpu, nb_gpu, use_gpu):

    data_file      = params.data_file
    N_e            = params.getint('data', 'N_e')
    N_total        = params.nb_channels
    N_t            = params.getint('detection', 'N_t')
    template_shift = params.getint('detection', 'template_shift')
    cc_merge       = params.getfloat('clustering', 'cc_mixtures')
    mixtures       = []
    to_remove      = []

    filename         = params.get('data', 'file_out_suff') + '.overlap-mixtures.hdf5'
    norm_templates   = load_data(params, 'norm-templates')
    best_elec        = load_data(params, 'electrodes')
    limits           = load_data(params, 'limits')
    nodes, edges     = get_nodes_and_edges(params)
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)

    overlap = get_overlaps(params, extension='-mixtures', erase=True, normalize=False, maxoverlap=False, verbose=False, half=True, use_gpu=use_gpu, nb_cpu=nb_cpu, nb_gpu=nb_gpu)
    overlap.close()

    SHARED_MEMORY = get_shared_memory_flag(params)

    if SHARED_MEMORY:
        c_overs    = load_data_memshared(params, 'overlaps', extension='-mixtures', use_gpu=use_gpu, nb_cpu=nb_cpu, nb_gpu=nb_gpu)
    else:
        c_overs    = load_data(params, 'overlaps', extension='-mixtures')

    if SHARED_MEMORY:
        templates  = load_data_memshared(params, 'templates', normalize=False)
    else:
        templates  = load_data(params, 'templates')

    x,        N_tm = templates.shape
    nb_temp        = int(N_tm//2)
    merged         = [nb_temp, 0]

    overlap_0 = numpy.zeros(nb_temp, dtype=numpy.float32)
    distances = numpy.zeros((nb_temp, nb_temp), dtype=numpy.int32)

    for i in xrange(nb_temp-1):
        data = c_overs[i].toarray()
        distances[i, i+1:] = numpy.argmax(data[i+1:, :], 1)
        distances[i+1:, i] = distances[i, i+1:]
        overlap_0[i] = data[i, N_t]

    all_temp    = numpy.arange(comm.rank, nb_temp, comm.size)
    sorted_temp = numpy.argsort(norm_templates[:nb_temp])[::-1][comm.rank::comm.size]
    M           = numpy.zeros((2, 2), dtype=numpy.float32)
    V           = numpy.zeros((2, 1), dtype=numpy.float32)

    to_explore = xrange(comm.rank, len(sorted_temp), comm.size)
    if comm.rank == 0:
        to_explore = get_tqdm_progressbar(to_explore)

    for count, k in enumerate(to_explore):

        k             = sorted_temp[k]
        electrodes    = numpy.take(inv_nodes, edges[nodes[best_elec[k]]])
        overlap_k     = c_overs[k]
        is_in_area    = numpy.in1d(best_elec, electrodes)
        all_idx       = numpy.arange(len(best_elec))[is_in_area]
        been_found    = False
        t_k           = None

        for i in all_idx:
            t_i = None
            if not been_found:
                overlap_i = c_overs[i]
                M[0, 0]   = overlap_0[i]
                V[0, 0]   = overlap_k[i, distances[k, i]]
                for j in all_idx[i+1:]:
                    t_j = None
                    M[1, 1]  = overlap_0[j]
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
                        if t_k is None:
                            t_k = templates[:, k].toarray().ravel()
                        if t_i is None:
                            t_i = templates[:, i].toarray().ravel()
                        if t_j is None:
                            t_j = templates[:, j].toarray().ravel()
                        new_template = (a1*t_i + a2*t_j)
                        similarity   = numpy.corrcoef(t_k, new_template)[0, 1]
                        local_overlap = numpy.corrcoef(t_i, t_j)[0, 1]
                        if similarity > cc_merge and local_overlap < cc_merge:
                            if k not in mixtures:
                                mixtures  += [k]
                                been_found = True
                                #print "Template", k, 'is sum of (%d, %g) and (%d,%g)' %(i, a1, j, a2)
                                break
    sys.stderr.flush()
    #print mixtures
    to_remove = numpy.unique(numpy.array(mixtures, dtype=numpy.int32))
    to_remove = all_gather_array(to_remove, comm, 0, dtype='int32')

    if len(to_remove) > 0 and comm.rank == 0:
        result = load_data(params, 'clusters')
        slice_templates(params, to_remove)
        slice_clusters(params, result, to_remove=to_remove)

    comm.Barrier()

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
    ine, ire, ife = numpy.array([[], [], []], dtype=numpy.uint32)
    if not edge:
        ine = numpy.where((numpy.hstack((dx, 0)) < 0) & (numpy.hstack((0, dx)) > 0))[0].astype(numpy.uint32)
    else:
        if edge.lower() in ['rising', 'both']:
            ire = numpy.where((numpy.hstack((dx, 0)) <= 0) & (numpy.hstack((0, dx)) > 0))[0].astype(numpy.uint32)
        if edge.lower() in ['falling', 'both']:
            ife = numpy.where((numpy.hstack((dx, 0)) < 0) & (numpy.hstack((0, dx)) >= 0))[0].astype(numpy.uint32)

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
