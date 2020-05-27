import os
import logging
import sys
import scipy.optimize
import numpy
# import pylab
import scipy.spatial.distance
import scipy.stats
import shutil
import h5py
import scipy.linalg
import scipy.sparse

from circus.shared.files import load_data, write_datasets, get_overlaps, load_data_memshared, get_stas, load_sp_memshared, load_sp
from circus.shared.utils import get_tqdm_progressbar, get_shared_memory_flag, dip, dip_threshold, \
    batch_folding_test_with_MPA, bhatta_dist, nd_bhatta_dist, test_if_support, test_if_purity
from circus.shared.messages import print_and_log
from circus.shared.probes import get_nodes_and_edges
from circus.shared.mpi import all_gather_array, comm, gather_array

import scipy.linalg
import scipy.sparse
import statsmodels.api as sm

logger = logging.getLogger(__name__)


class DistanceMatrix(object):

    def __init__(self, size, distances=None):

        self.size = size
        self.didx = lambda i, j: i * self.size + j - i * (i + 1) // 2 - i - 1
        self.distances = distances  # condensed matrix

    def initialize(self, data, ydata=None):

        if ydata is None:
            self.distances = scipy.spatial.distance.pdist(data, 'euclidean').astype(numpy.float32)
        else:
            self.distances = scipy.spatial.distance.cdist(data, ydata, 'euclidean').astype(numpy.float32)

        return

    def get_value(self, i, j):

        if i < j:
            value = self.distances[self.didx(i, j)]
        elif i > j:
            value = self.distances[self.didx(j, i)]
        elif i == j:
            value = 0.0
        else:
            raise RuntimeError()

        return value

    def get_row(self, i, with_diag=True):

        start = self.distances[self.didx(numpy.arange(0, i), i)]
        end = self.distances[self.didx(i, numpy.arange(i + 1, self.size))]
        if with_diag:
            result = numpy.concatenate((start, numpy.array([0], dtype=numpy.float32), end))
        else:
            result = numpy.concatenate((start, end))
    
        return result

    def get_col(self, i, with_diag=True):

        return self.get_row(i, with_diag=with_diag)

    def to_dense(self):

        return scipy.spatial.distance.squareform(self.distances)

    def get_rows(self, indices, with_diag=True):

        if with_diag:
            result = numpy.zeros((len(indices), self.size), dtype=numpy.float32)
        else:
            result = numpy.zeros((len(indices), self.size - 1), dtype=numpy.float32)

        for count, i in enumerate(indices):
            result[count] = self.get_row(i, with_diag=with_diag)

        return result

    def get_cols(self, indices, with_diag=True):

        if with_diag:
            result = numpy.zeros((self.size, len(indices)), dtype=numpy.float32)
        else:
            result = numpy.zeros((self.size - 1, len(indices)), dtype=numpy.float32)

        for count, i in enumerate(indices):
            result[:, count] = self.get_col(i, with_diag=with_diag)

        return result

    def get_deltas_and_neighbors(self, rho):
        """Find the distance to and the index of the nearest point with a higher density.

        Argument:
            rho
        Returns:
            nearest_higher_rho_distances
                For each point, distance to the nearest point with a higher density (i.e. delta).
            nearest_higher_rho_indices
                For each point, index of the nearest point with a higher density (i.e. neighbor).
        """

        indices = numpy.argsort(-rho)  # sort indices by decreasing rho values
        nearest_higher_rho_indices = numpy.zeros(self.size, dtype=numpy.int)  # i.e. neighbors
        nearest_higher_rho_distances = numpy.zeros(self.size, dtype=numpy.float32)  # i.e. deltas
        for k, index in enumerate(indices):
            higher_rho_indices = indices[0:k + 1]
            higher_rho_distances = self.get_row(index)[higher_rho_indices]
            higher_rho_distances[higher_rho_distances == 0.0] = float('inf')
            nearest_index = numpy.argmin(higher_rho_distances)
            nearest_higher_rho_indices[index] = higher_rho_indices[nearest_index]
            nearest_higher_rho_distances[index] = higher_rho_distances[nearest_index]

        if len(indices) > 1:
            nearest_higher_rho_distances[indices[0]] = numpy.max(nearest_higher_rho_distances[indices[1:]])
            nearest_higher_rho_distances[numpy.isinf(nearest_higher_rho_distances)] = 0

        return nearest_higher_rho_distances, nearest_higher_rho_indices

    @property
    def max(self):

        return numpy.max(self.distances)

    def __del__(self):

        del self.distances


def fit_rho_delta(xdata, ydata, alpha=3):

    if xdata.min() == xdata.max():
        return numpy.zeros(0, dtype=numpy.int32)

    try:
        x = sm.add_constant(xdata)
        model = sm.RLM(ydata, x)
        results = model.fit()
        difference = ydata - results.fittedvalues
        factor = numpy.median(numpy.abs(difference - numpy.median(difference)))
        z_score = difference - alpha*factor*(1 + results.fittedvalues)
        centers = numpy.where(z_score >= 0)[0]
    except Exception:
        centers = numpy.zeros(0, dtype=numpy.int32)

    return centers


def compute_rho(data, update=None, mratio=0.01):

    nb_points = len(data)
    nb_selec = max(5, int(mratio * nb_points))
    rho = numpy.zeros(nb_points, dtype=numpy.float32)
    dist_sorted = {}

    if update is None:
        dist = DistanceMatrix(nb_points)
        dist.initialize(data)
        for i in range(nb_points):
            data = dist.get_row(i, with_diag=False)
            if len(data) > nb_selec:
                dist_sorted[i] = data[numpy.argpartition(data, nb_selec)[:nb_selec]]
            else:
                dist_sorted[i] = data
            rho[i] = numpy.mean(dist_sorted[i])
        answer = rho, dist, dist_sorted
    else:
        for i in range(nb_points):
            dist = scipy.spatial.distance.cdist(data[i].reshape(1, len(data[i])), update[0]).flatten()
            dist = numpy.concatenate((update[1][i], dist))
            if len(dist) > nb_selec:
                dist_sorted[i] = dist[numpy.argpartition(dist, nb_selec)[:nb_selec]]
            else:
                dist_sorted[i] = dist
            rho[i] = numpy.mean(dist_sorted[i])
        answer = rho, dist_sorted

    return answer


def clustering_by_density(rho, dist, n_min, alpha=3, halo_rejection=3):

    nb_points = len(rho)
    distances = DistanceMatrix(nb_points, distances=dist)
    deltas, neighbors = distances.get_deltas_and_neighbors(rho)
    nb_clusters, labels, centers = find_centroids_and_clusters(distances, rho, deltas, neighbors, alpha)
    halolabels = halo_assign(labels, rho, n_min, halo_rejection) - 1
    centers = numpy.where(centers - 1 >= 0)[0]
    del distances

    return halolabels, rho, deltas, centers


def find_centroids_and_clusters(dist, rho, delta, neighbors, alpha=3, method='nearest_denser_point'):
    """Find centroids and clusters.

    Arguments:
        dist
            Matrix of distances between pairs of points.
        rho
            For each point, density in its neighborhood.
        delta
            For each point, distance of the nearest point with higher density.
        neighbors
            For each point, index of the nearest point with higher density.
        alpha
        method
    """

    nb_points = len(rho)
    # Find centroids.
    centroids = numpy.zeros(nb_points, dtype=numpy.int)
    centroid_indices = fit_rho_delta(rho, delta, alpha)
    nb_clusters = len(centroid_indices)
    cluster_nbs = numpy.arange(1, nb_clusters + 1)
    centroids[centroid_indices] = cluster_nbs  # assigning cluster numbers to centroids
    # Assign each point to one cluster.
    if method == 'nearest_centroid':
        # Custom (and naive) method.
        if nb_clusters <= 1:
            labels = numpy.ones(nb_points, dtype=numpy.int)  # all points in one cluster
        else:
            distances_to_centroids = dist.get_rows(centroid_indices)
            labels = numpy.argmin(distances_to_centroids, axis=0) + 1
    elif method == 'nearest_denser_point':
        # Method described in [Rodriguez & Laio (2014)](https://science.sciencemag.org/content/344/6191/1492.full).
        if nb_clusters <= 1:
            labels = numpy.ones(nb_points, dtype=numpy.int)  # all points in one cluster
        else:
            labels = numpy.copy(centroids)
            indices = numpy.argsort(-rho)  # sort indices by decreasing density
            for index in indices:
                if labels[index] == 0:
                    labels[index] = labels[neighbors[index]]
    else:
        raise ValueError("unexpected value %s" % method)

    return nb_clusters, labels, centroids


def halo_assign(labels, rhos, n_min, halo_rejection=3):
    """Unassign outliers."""

    halolabels = labels.copy()
    for label_nb in numpy.unique(labels):
        indices = numpy.where(labels == label_nb)[0]
        median_rho = numpy.median(rhos[indices])
        # selected_indices = indices[rhos[indices] < median_rho]
        mad_rho = numpy.median(numpy.abs(rhos[indices] - median_rho))
        selected_indices = indices[rhos[indices] < (median_rho - halo_rejection*mad_rho)]  # TODO enhance?
        if len(indices) - len(selected_indices) > n_min:
            halolabels[selected_indices] = 0  # i.e. set to 0 (unassign)
    return halolabels


def merging(groups, merging_method, merging_param, data):

    def perform_merging(groups_, merging_method_, merging_param_, data_):
        mask_ = numpy.where(groups_ > -1)[0]
        clusters_ = numpy.unique(groups_[mask_])
        dmin_ = numpy.inf
        to_merge = [None, None]

        for ic1 in range(len(clusters_)):
            idx1 = numpy.where(groups_ == clusters_[ic1])[0]
            sd1 = numpy.take(data_, idx1, axis=0)

            if merging_method_ in ['distance', 'dip', 'folding', 'bhatta']:
                m1 = numpy.median(sd1, 0)
            else:
                m1 = None  # default assignment

            for ic2 in range(ic1+1, len(clusters_)):
                idx2 = numpy.where(groups_ == clusters_[ic2])[0]
                sd2 = numpy.take(data_, idx2, axis=0)

                if merging_method_ in ['distance', 'dip', 'folding', 'bhatta']:
                    m2 = numpy.median(sd2, 0)
                    v_n = (m1 - m2)
                    pr_1 = numpy.dot(sd1, v_n)
                    pr_2 = numpy.dot(sd2, v_n)
                else:
                    pr_1 = None  # default assignment
                    pr_2 = None  # default assignment

                if merging_method_ == 'folding':
                    sub_data = numpy.concatenate([pr_1, pr_2])
                    unimodal, p_value, phi, _ = batch_folding_test_with_MPA(sub_data, True)
                    if unimodal:
                        dist = p_value
                    else:
                        dist = numpy.inf
                elif merging_method_ == 'nd-folding':
                    sub_data = numpy.vstack((sd1, sd2))[:, :3]
                    unimodal, p_value, phi, _ = batch_folding_test_with_MPA(sub_data, True)
                    if unimodal:
                        dist = p_value
                    else:
                        dist = numpy.inf
                elif merging_method_ == 'dip':
                    sub_data = numpy.concatenate([pr_1, pr_2])
                    if len(sub_data) > 5:
                        dist = dip(sub_data) / dip_threshold(len(sub_data), merging_param_)
                    else:
                        dist = numpy.inf
                elif merging_method_ == 'distance':
                    med1 = numpy.median(pr_1)
                    med2 = numpy.median(pr_2)
                    mad1 = numpy.median(numpy.abs(pr_1 - med1))**2
                    mad2 = numpy.median(numpy.abs(pr_2 - med2))**2
                    norm = mad1 + mad2
                    dist = numpy.sqrt((med1 - med2)**2/norm)
                elif merging_method_ == 'bhatta':
                    try:
                        dist = bhatta_dist(pr_1, pr_2)
                    except Exception:
                        dist = numpy.inf
                elif merging_method_ == 'nd-bhatta':
                    try:
                        dist = nd_bhatta_dist(sd1.T, sd2.T)
                    except Exception:
                        dist = numpy.inf
                else:
                    raise ValueError("unexpected value: %s" % merging_method)

                if dist < dmin_:
                    dmin_ = dist
                    to_merge = [ic1, ic2]

        if merging_method_ == 'dip':
            thr_ = 1
        elif merging_method_ in ['folding', 'nd-folding', 'bhatta', 'nd-bhatta']:
            thr_ = merging_param_
        elif merging_method_ == 'distance':
            thr_ = merging_param_ / 0.674
        else:
            raise ValueError("unexpected value: %s" % merging_method_)

        if dmin_ < thr_:
            ic1, ic2 = to_merge
            c1, c2 = clusters_[ic1], clusters_[ic2]
            selection = numpy.where(groups_ == c2)[0]
            groups_[selection] = c1
            merge_ = (c1, c2)
            return True, groups_, merge_, dmin_

        return False, groups_, None, None

    has_been_merged = True
    mask = numpy.where(groups > -1)[0]
    clusters = numpy.unique(groups[mask])
    merged = [len(clusters), 0]

    if merging_method == 'dip':
        thr = 1
    elif merging_method in ['folding', 'nd-folding', 'bhatta', 'nd-bhatta']:
        thr = merging_param
    elif merging_method == 'distance':
        thr = merging_param / 0.674
    else:
        raise ValueError("unexpected value: %s" % merging_method)

    merge_history = {
        'merge': [],
        'distance': [],
        'method': merging_method,
        'threshold': thr,
    }

    while has_been_merged:
        has_been_merged, groups, merge, dmin = perform_merging(groups, merging_method, merging_param, data)
        if has_been_merged:
            merged[1] += 1
            merge_history['merge'].append(merge)
            merge_history['distance'].append(dmin)

    return groups, merged, merge_history


def slice_templates(params, to_remove=None, to_merge=None, extension='', input_extension=''):
    """Slice templates in HDF5 file.

    Arguments:
        params
        to_remove: none | list (optional)
            An array of template indices to remove.
            The default value is None.
        to_merge: none | list | numpy.ndarray (optional)
            An array of pair of template indices to merge
            (i.e. shape = (nb_merges, 2)).
            The default value is None.
        extension: string (optional)
            The extension to use as output.
            The default value is ''.
        input_extension: string (optional)
            The extension to use as input.
            The default value is ''.
    """

    if to_remove is None:
        to_remove = []
    if to_merge is None:
        to_merge = []

    file_out_suff = params.get('data', 'file_out_suff')

    data_file = params.data_file
    n_e = params.getint('data', 'N_e')
    n_total = params.nb_channels
    hdf5_compress = params.getboolean('data', 'hdf5_compress')
    n_t = params.getint('detection', 'N_t')
    template_shift = params.getint('detection', 'template_shift')
    has_support = test_if_support(params, input_extension)
    has_purity = test_if_purity(params, input_extension)
    fine_amplitude = params.getboolean('clustering', 'fine_amplitude')

    if comm.rank == 0:
        print_and_log(['Node 0 is slicing templates'], 'debug', logger)
        old_templates = load_data(params, 'templates', extension=input_extension)
        old_limits = load_data(params, 'limits', extension=input_extension)
        if has_support:
            old_supports = load_data(params, 'supports', extension=input_extension)
        else:
            old_supports = None  # default assignment
        if has_purity:
            old_purity = load_data(params, 'purity', extension=input_extension)
        else:
            old_purity = None  # default assignment
        _, n_tm = old_templates.shape
        norm_templates = load_data(params, 'norm-templates', extension=input_extension)

        # Determine the template indices to delete.
        to_delete = list(to_remove)  # i.e. copy
        if len(to_merge) > 0:
            for count in range(len(to_merge)):
                remove = to_merge[count][1]
                to_delete += [remove]

        # Determine the indices to keep.
        all_templates = set(numpy.arange(n_tm // 2))
        to_keep = numpy.array(list(all_templates.difference(to_delete)))

        positions = numpy.arange(len(to_keep))

        # Initialize new HDF5 file for templates.
        local_keep = to_keep[positions]
        templates = scipy.sparse.lil_matrix((n_e * n_t, 2 * len(to_keep)), dtype=numpy.float32)
        hfilename = file_out_suff + '.templates{}.hdf5'.format('-new')
        hfile = h5py.File(hfilename, 'w', libver='earliest')
        norms = hfile.create_dataset('norms', shape=(2 * len(to_keep), ), dtype=numpy.float32, chunks=True)
        limits = hfile.create_dataset('limits', shape=(len(to_keep), 2), dtype=numpy.float32, chunks=True)
        if has_support:
            supports = hfile.create_dataset('supports', shape=(len(to_keep), n_e), dtype=numpy.bool, chunks=True)
        else:
            supports = None  # default assignment

        if has_purity:
            purity = hfile.create_dataset('purity', shape=(len(to_keep), ), dtype=numpy.float32, chunks=True)
        else:
            purity = None
        # For each index to keep.
        for count, keep in zip(positions, local_keep):
            # Copy template.
            templates[:, count] = old_templates[:, keep]
            templates[:, count + len(to_keep)] = old_templates[:, keep + n_tm // 2]
            # Copy norm.
            norms[count] = norm_templates[keep]
            norms[count + len(to_keep)] = norm_templates[keep + n_tm // 2]
            if has_support:
                supports[count] = old_supports[keep]

            # Copy limits.
            if len(to_merge) == 0:
                new_limits = old_limits[keep]
                if has_purity:
                    new_purity = old_purity[keep]
            else:
                subset = numpy.where(to_merge[:, 0] == keep)[0]
                if len(subset) > 0:
                    # pylab.subplot(211)
                    # pylab.plot(templates[:, count].toarray().flatten())
                    # ymin, ymax = pylab.ylim()
                    # pylab.subplot(212)
                    # for i in to_merge[subset]:
                    #     pylab.plot(old_templates[:, i[1]].toarray().flatten())
                    # pylab.ylim(ymin, ymax)
                    # pylab.savefig('merge_%d.png' % count)
                    # pylab.close()
                    # Index to keep is involved in merge(s) and limits need to
                    # be updated.
                    idx = numpy.unique(to_merge[subset].flatten())
                    ratios = norm_templates[idx] / norm_templates[keep]
                    new_limits = [
                        numpy.min(ratios * old_limits[idx][:, 0]),
                        numpy.max(ratios * old_limits[idx][:, 1])
                    ]
                    if has_purity:
                        new_purity = numpy.mean(old_purity[idx])
                else:
                    new_limits = old_limits[keep]
                    if has_purity:
                        new_purity = old_purity[keep]
            if not fine_amplitude:
                limits[count] = new_limits
            else:
                limits[count] = [0.5, 1.5]
            if has_purity:
                purity[count] = new_purity

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
        hfile.create_dataset('temp_shape', data=numpy.array([n_e, n_t, 2 * len(to_keep)], dtype=numpy.int32))
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


def slice_clusters(
        params, result, to_remove=None, to_merge=None, extension='', input_extension='', light=False, method='safe'
):
    """Slice clusters in HDF5 templates.

    Arguments:
        params
        result
        to_remove: none | list (optional)
        to_merge: none | list | numpy.ndarray (optional)
        extension: string (optional)
            The default value is ''.
        input_extension: string (optional)
            The default value is ''.
        light: boolean (optional)
        method: string (optional)
    """

    if to_remove is None:
        to_remove = []
    if to_merge is None:
        to_merge = []

    file_out_suff = params.get('data', 'file_out_suff')
    data_file = params.data_file
    n_e = params.getint('data', 'N_e')
    n_total = params.nb_channels
    hdf5_compress = params.getboolean('data', 'hdf5_compress')
    n_t = params.getint('detection', 'N_t')
    template_shift = params.getint('detection', 'template_shift')
    debug = params.getboolean('clustering', 'debug')

    if comm.rank == 0:

        print_and_log(['Node 0 is slicing clusters'], 'debug', logger)
        old_templates = load_data(params, 'templates', extension=input_extension)
        _, n_tm = old_templates.shape

        # Determine the template indices to delete.
        to_delete = list(to_remove)
        if len(to_merge) > 0:
            for count in range(len(to_merge)):
                remove = to_merge[count][1]
                to_delete += [remove]

        # Determine the indices to keep.
        all_templates = set(numpy.arange(n_tm // 2))
        to_keep = numpy.array(list(all_templates.difference(to_delete)))

        all_elements = [[] for _ in range(n_e)]
        for target in numpy.unique(to_delete):
            elec = result['electrodes'][target]
            nic = target - numpy.where(result['electrodes'] == elec)[0][0]
            mask = result['clusters_' + str(elec)] > -1
            tmp = numpy.unique(result['clusters_' + str(elec)][mask])
            all_elements[elec] += list(numpy.where(result['clusters_' + str(elec)] == tmp[nic])[0])

        myfilename = file_out_suff + '.clusters{}.hdf5'.format(input_extension)
        myfile = h5py.File(myfilename, 'r', libver='earliest')

        for elec in range(n_e):
            if not light:
                result['data_' + str(elec)] = numpy.delete(result['data_' + str(elec)], all_elements[elec], axis=0)
                result['clusters_' + str(elec)] = numpy.delete(result['clusters_' + str(elec)], all_elements[elec])
                result['times_' + str(elec)] = numpy.delete(result['times_' + str(elec)], all_elements[elec])
                result['peaks_' + str(elec)] = numpy.delete(result['peaks_' + str(elec)], all_elements[elec])
                if debug:
                    result['rho_' + str(elec)] = numpy.delete(result['rho_' + str(elec)], all_elements[elec])
                    result['delta_' + str(elec)] = numpy.delete(result['delta_' + str(elec)], all_elements[elec])
            else:
                result['clusters_' + str(elec)] = numpy.delete(result['clusters_' + str(elec)], all_elements[elec])
                data = myfile.get('data_' + str(elec))[:]
                result['data_' + str(elec)] = numpy.delete(data, all_elements[elec], axis=0)
                data = myfile.get('times_' + str(elec))[:]
                result['times_' + str(elec)] = numpy.delete(data, all_elements[elec])
                data = myfile.get('peaks_' + str(elec))[:]
                result['peaks_' + str(elec)] = numpy.delete(data, all_elements[elec])
                data = myfile.get('noise_times_' + str(elec))[:]
                result['noise_times_' + str(elec)] = data
                if debug:
                    data = myfile.get('rho_' + str(elec))[:]
                    result['rho_' + str(elec)] = numpy.delete(data, all_elements[elec])
                    data = myfile.get('delta_' + str(elec))[:]
                    result['delta_' + str(elec)] = numpy.delete(data, all_elements[elec])

        myfile.close()
        if method == 'safe':
            result['electrodes'] = numpy.delete(result['electrodes'], numpy.unique(to_delete))
        elif method == 'new':
            result['electrodes'] = result['electrodes'][to_keep]
        else:
            raise ValueError("Unexpected method value: {}".format(method))

        cfilename = file_out_suff + '.clusters{}.hdf5'.format('-new')
        cfile = h5py.File(cfilename, 'w', libver='earliest')
        to_write = ['data_', 'clusters_', 'times_', 'peaks_', 'noise_times_']
        if debug:
            to_write += ['rho_', 'delta_']
        for ielec in range(n_e):
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

    for t in times:
        sub_result = {'spiketimes': {}, 'amplitudes': {}}
        for key in result['spiketimes'].keys():
            spike_times = result['spiketimes'][key]
            spike_times = spike_times.ravel()
            amplitudes = result['amplitudes'][key]
            amplitudes = amplitudes.ravel()
            indices = numpy.where((spike_times >= t[0]) & (spike_times <= t[1]))[0]
            sub_result['spiketimes'][key] = spike_times[indices] - t[0]
            sub_result['amplitudes'][key] = amplitudes[indices]
        sub_results += [sub_result]

    return sub_results
    

def merging_cc(params, nb_cpu, nb_gpu, use_gpu):

    def remove(result_, distances_, cc_merge_):
        do_merge = True
        to_merge_ = numpy.zeros((0, 2), dtype=numpy.int32)
        g_idx = range(len(distances_))
        while do_merge:
            dmax = distances_.max()
            idx_ = numpy.where(distances_ == dmax)
            one_merge = [idx_[0][0], idx_[1][0]]
            do_merge = dmax >= cc_merge_

            if do_merge:

                elec_ic1 = result_['electrodes'][one_merge[0]]
                elec_ic2 = result_['electrodes'][one_merge[1]]
                nic1 = one_merge[0] - numpy.where(result_['electrodes'] == elec_ic1)[0][0]
                nic2 = one_merge[1] - numpy.where(result_['electrodes'] == elec_ic2)[0][0]
                mask1 = result_['clusters_' + str(elec_ic1)] > -1
                mask2 = result_['clusters_' + str(elec_ic2)] > -1
                tmp1 = numpy.unique(result_['clusters_' + str(elec_ic1)][mask1])
                tmp2 = numpy.unique(result_['clusters_' + str(elec_ic2)][mask2])
                elements1 = numpy.where(result_['clusters_' + str(elec_ic1)] == tmp1[nic1])[0]
                elements2 = numpy.where(result_['clusters_' + str(elec_ic2)] == tmp2[nic2])[0]

                if len(elements1) > len(elements2):
                    to_remove = one_merge[1]
                    to_keep = one_merge[0]
                    elec = elec_ic2
                    elements = elements2
                else:
                    to_remove = one_merge[0]
                    to_keep = one_merge[1]
                    elec = elec_ic1
                    elements = elements1

                result_['data_' + str(elec)] = numpy.delete(result_['data_' + str(elec)], elements, axis=0)
                result_['clusters_' + str(elec)] = numpy.delete(result_['clusters_' + str(elec)], elements)
                result_['times_' + str(elec)] = numpy.delete(result_['times_' + str(elec)], elements)
                result_['peaks_' + str(elec)] = numpy.delete(result_['peaks_' + str(elec)], elements)
                result_['electrodes'] = numpy.delete(result_['electrodes'], to_remove)
                distances_ = numpy.delete(distances_, to_remove, axis=0)
                distances_ = numpy.delete(distances_, to_remove, axis=1)
                to_merge_ = numpy.vstack((to_merge_, numpy.array([g_idx[to_keep], g_idx[to_remove]])))
                g_idx.pop(to_remove)

        return to_merge_, result_

    data_file = params.data_file
    n_e = params.getint('data', 'N_e')
    n_total = params.nb_channels
    n_t = params.getint('detection', 'N_t')
    template_shift = params.getint('detection', 'template_shift')
    blosc_compress = params.getboolean('data', 'blosc_compress')

    n_tm = load_data(params, 'nb_templates')
    nb_temp = int(n_tm // 2)
    to_merge = []
    cc_merge = params.getfloat('clustering', 'cc_merge')
    norm = n_e * n_t
    decimation = params.getboolean('clustering', 'decimation')
    adapted_cc = params.getboolean('clustering', 'adapted_cc')
    adapted_thr = params.getint('clustering', 'adapted_thr')

    if cc_merge < 1:

        result = []
        overlap = get_overlaps(
            params, extension='-merging', erase=True, normalize=True, maxoverlap=False, verbose=False, half=True,
            use_gpu=use_gpu, nb_cpu=nb_cpu, nb_gpu=nb_gpu, decimation=decimation
        )
        overlap.close()
        filename = params.get('data', 'file_out_suff') + '.overlap-merging.hdf5'

        SHARED_MEMORY = get_shared_memory_flag(params)

        if not SHARED_MEMORY:
            over_x, over_y, over_data, sub_over, over_sorted, over_shape = load_data(
                params, 'overlaps-raw', extension='-merging'
            )
        else:
            over_x, over_y, over_data, sub_over, over_sorted, over_shape, mpi_memory = load_data_memshared(
                params, 'overlaps-raw', extension='-merging', use_gpu=use_gpu, nb_cpu=nb_cpu, nb_gpu=nb_gpu
            )


        to_explore = numpy.arange(nb_temp)[comm.rank::comm.size]
        distances = numpy.zeros((len(to_explore), nb_temp), dtype=numpy.float32)

        res = []
        res2 = []
        for i in to_explore:
            res += [i * nb_temp, (i + 1) * nb_temp]
            res2 += [i, i+1]

        bounds = numpy.searchsorted(over_x, res, 'left')
        bounds_2 = numpy.searchsorted(sub_over[over_sorted], res2, 'left')

        duration = over_shape[1] // 2
        mask_duration = (over_y < duration)

        import gc

        for count, i in enumerate(to_explore):

            xmin, xmax = bounds[2*count:2*(count+1)]
            local_x = over_x[xmin:xmax] - (i * nb_temp)
            local_y = over_y[xmin:xmax]
            local_data = over_data[xmin:xmax]

            xmin, xmax = bounds_2[2*count:2*(count+1)]
            nslice = over_sorted[xmin:xmax][mask_duration[over_sorted[xmin:xmax]]]

            local_x = numpy.concatenate((local_x, over_x[nslice] // nb_temp))
            local_y = numpy.concatenate((local_y, (over_shape[1] - 1) - over_y[nslice]))
            local_data = numpy.concatenate((local_data, over_data[nslice]))

            data = scipy.sparse.csr_matrix((local_data, (local_x, local_y)), shape=(nb_temp, over_shape[1]), dtype=numpy.float32)
            distances[count, :] = data.max(1).toarray().flatten()
            del local_x, local_y, local_data, data, nslice
            gc.collect()

        distances /= norm

        # Now we need to sync everything across nodes.
        distances = gather_array(distances, comm, 0, 1, 'float32', compress=blosc_compress)
        if comm.rank == 0:
            indices = []
            for idx in range(comm.size):
                indices += list(numpy.arange(idx, nb_temp, comm.size))
            indices = numpy.argsort(indices).astype(numpy.int32)

            distances = distances[indices, :]
            line = numpy.arange(nb_temp)
            distances[line, line] = 0

            #distances = numpy.maximum(distances, distances.T)

        comm.Barrier()

        if comm.rank == 0:
            if adapted_cc:
                common_supports = load_data(params, 'common-supports')
                exponents = numpy.exp(-common_supports/adapted_thr)
                distances = distances ** exponents
            result = load_data(params, 'clusters')
            to_merge, result = remove(result, distances, cc_merge)

        to_merge = numpy.array(to_merge)
        to_merge = comm.bcast(to_merge, root=0)

        if len(to_merge) > 0:
            slice_templates(params, to_merge=to_merge)
            slice_clusters(params, result)

        comm.Barrier()

        del result, over_x, over_y, over_data, over_sorted, sub_over

        if comm.rank == 0:
            os.remove(filename)

        if SHARED_MEMORY:
            for memory in mpi_memory:
                memory.Free()

    return [nb_temp, len(to_merge)]


def compute_error(good_values, bad_values, bounds):

    fn = numpy.sum((good_values < bounds[0]) | (good_values > bounds[1]))
    fp = numpy.sum((bounds[0] <= bad_values) & (bad_values <= bounds[1]))
    tp = numpy.sum((bounds[0] <= good_values) & (good_values <= bounds[1]))
    tn = numpy.sum((bad_values < bounds[0]) | (bad_values > bounds[1]))

    #precision = tp / (tp + fp)
    #recall = tp / (tp + fp)
    #f1_score = 1 - 2*(precision * recall)/(precision + recall)

    denom = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    
    if denom > 0:
        mcc = 1 - (tp*tn - fp*fn)/numpy.sqrt(denom)
    else:
        mcc = 1

    return mcc

def score(x, good_values, bad_values):
    # We want a minimal error, with the larger bounds that are possible
    return compute_error(good_values, bad_values, x) + 1e-2*(2 - (x[1] - x[0]))**2


def refine_amplitudes(params, nb_cpu, nb_gpu, use_gpu, normalization=True, debug_plots=''):
    
    data_file = params.data_file
    template_shift = params.getint('detection', 'template_shift')
    norm_templates = load_data(params, 'norm-templates')
    best_elec = load_data(params, 'electrodes')
    limits = load_data(params, 'limits')
    fine_amplitude = params.getboolean('clustering', 'fine_amplitude')
    N_e = params.getint('data', 'N_e')
    N_t = params.getint('detection', 'N_t')
    n_total = params.nb_channels
    clusters = load_data(params, 'clusters-nodata')
    file_out_suff = params.get('data', 'file_out_suff')
    plot_path = os.path.join(params.get('data', 'file_out_suff'), 'plots')
    nodes, edges = get_nodes_and_edges(params)
    inv_nodes = numpy.zeros(n_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.arange(len(nodes))
    hdf5_compress = params.getboolean('data', 'hdf5_compress')
    blosc_compress = params.getboolean('data', 'blosc_compress')
    tmp_path_loc = os.path.join(os.path.abspath(params.get('data', 'file_out_suff')), 'tmp')

    max_snippets = 250
    max_noise_snippets = min(max_snippets, 10000 // N_e)
    # thr_similarity = 0.25

    SHARED_MEMORY = get_shared_memory_flag(params)
    
    if SHARED_MEMORY:
        templates, mpi_memory_1 = load_data_memshared(params, 'templates', normalize=False, transpose=True)
    else:
        templates = load_data(params, 'templates')
        templates = templates.T

    supports = load_data(params, 'supports')
    n_tm, x = templates.shape
    nb_temp = int(n_tm // 2)
    norm_templates = load_data(params, 'norm-templates')[:nb_temp]
    norm_templates *= numpy.sqrt(N_e * N_t)
    norm_2 = norm_templates ** 2
    sindices = inv_nodes[nodes]

    # For each electrode, get the local cluster labels.
    indices = {}
    for i in range(N_e):
        labels = numpy.unique(clusters['clusters_%d' % i])
        labels = labels[labels > -1]
        indices[i] = list(labels)

    mask_intersect = numpy.zeros((nb_temp, nb_temp), dtype=numpy.bool)
    for i in range(nb_temp):
        for j in range(i, nb_temp):
            mask_intersect[i, j] = numpy.any(supports[i]*supports[j])

    mask_intersect = numpy.maximum(mask_intersect, mask_intersect.T)

    all_sizes = {}
    all_temp = numpy.arange(comm.rank, nb_temp, comm.size)
    all_elec = numpy.arange(comm.rank, N_e, comm.size)

    if comm.rank == 0:
        to_explore = get_tqdm_progressbar(params, all_temp)
    else:
        to_explore = all_temp

    # First we gather all the snippets for the final templates.

    clusters_info = {}
    
    all_snippets = {'all' : {}, 'noise' : {}}
    for key in ['all', 'noise']:
        all_snippets[key]['x'] = [numpy.zeros(0, dtype=numpy.int32)]
        all_snippets[key]['data'] = [numpy.zeros(0, dtype=numpy.float32)]

    for i in to_explore:  # for each cluster...

        ref_elec = best_elec[i]  # i.e. electrode of the cluster

        times = clusters['times_%d' % ref_elec]
        labels = clusters['clusters_%d' % ref_elec]
        peaks = clusters['peaks_%d' % ref_elec]
        position = numpy.where(best_elec[:i] == ref_elec)[0]
        tgt_label = indices[ref_elec][len(position)]  # i.e. local cluster label (per electrode)
        idx = numpy.where(labels == tgt_label)[0]

        clusters_info[i] = {
            'electrode_nb': ref_elec,
            'local_cluster_nb': tgt_label,
        }

        if peaks[idx][0] == 0:
            p = 'pos'
        elif peaks[idx][0] == 1:
            p = 'neg'
        else:
            raise ValueError("unexpected value {}".format(peaks[idx][0]))

        idx_i = numpy.random.permutation(idx)[:max_snippets]
        times_i = times[idx_i]
        labels_i = labels[idx_i]
        snippets = get_stas(params, times_i, labels_i, ref_elec, neighs=sindices, nodes=nodes, pos=p)

        nb_snippets, nb_electrodes, nb_times_steps = snippets.shape
        snippets = numpy.ascontiguousarray(snippets.reshape(nb_snippets, nb_electrodes * nb_times_steps).T)

        for j in range(nb_temp):
            if mask_intersect[i, j]:
                data = templates[j].dot(snippets)[0].astype(numpy.float32)
                all_snippets['all']['x'].append((j*nb_temp + i)*numpy.ones(len(data), dtype=numpy.int32))
                all_snippets['all']['data'].append(data)

        all_sizes[i] = snippets.shape[1]

    noise_amplitudes = {}
    for i in range(nb_temp):
        noise_amplitudes[i] = [numpy.zeros(0, dtype=numpy.float32)]

    if comm.rank == 0:
        to_explore = get_tqdm_progressbar(params, all_elec)
    else:
        to_explore = all_elec

    for elec in to_explore:
        times = clusters['noise_times_' + str(elec)]

        idx = len(times)
        idx_i = numpy.random.permutation(idx)[:max_noise_snippets]
        times_i = times[idx_i]
        labels_i = numpy.zeros(idx)
        snippets = get_stas(params, times_i, labels_i, elec, neighs=sindices, nodes=nodes, auto_align=False)

        nb_snippets, nb_electrodes, nb_times_steps = snippets.shape
        snippets = numpy.ascontiguousarray(snippets.reshape(nb_snippets, nb_electrodes * nb_times_steps).T)

        for j in range(nb_temp):
            data = templates[j].dot(snippets)[0].astype(numpy.float32)
            noise_amplitudes[j].append(data)

    for i in range(nb_temp):
        amplitudes = numpy.concatenate(noise_amplitudes.pop(i))
        all_snippets['noise']['x'].append(i*numpy.ones(len(amplitudes), dtype=numpy.int32))
        all_snippets['noise']['data'].append(amplitudes)

    filename = os.path.join(tmp_path_loc, 'sp.h5')

    if comm.rank == 0:
        if not os.path.exists(tmp_path_loc):
            os.makedirs(tmp_path_loc)

        if os.path.exists(filename):
            os.remove(filename)

        hfile = h5py.File(filename, 'w', libver='earliest')

    for k in ['all', 'noise']:

        for key in ['x', 'data']:
            data = numpy.concatenate(all_snippets[k].pop(key))
            if key == 'x':
                data = gather_array(data, comm, dtype='int32', compress=blosc_compress)
            else:
                data = gather_array(data, comm, dtype='float32')

            # We sort by x indices for faster retrieval later
            if comm.rank == 0:
                if key == 'x':
                    indices = numpy.argsort(data).astype(numpy.int32)
                
                data = data[indices]

                if hdf5_compress:
                    hfile.create_dataset('%s/over_%s' %(k, key), data=data, compression='gzip')
                else:
                    hfile.create_dataset('%s/over_%s' %(k, key), data=data)
            del data

    # We need to gather the sparse arrays.
    if comm.rank == 0:
        del indices
        hfile.close()

    comm.Barrier()
    ## Once all data are saved, we need to load them with shared mpi_memory
    if SHARED_MEMORY:
        all_snippets, mpi_memory_2 = load_sp_memshared(filename, nb_temp)
    else:
        all_snippets = load_sp(filename, nb_temp)

    comm.Barrier()
    if comm.rank == 0:
        os.remove(filename)

    #del all_snippets
    # And finally, we set a_min/a_max optimally for all the template.
    purity_level = numpy.zeros(len(all_temp), dtype=numpy.float32)
    max_nb_chances = numpy.zeros(len(all_temp), dtype=numpy.float32)
    if fine_amplitude:
        bounds = numpy.zeros((len(all_temp), 2), dtype=numpy.float32)

    for count, i in enumerate(all_temp):

        # First, we collect admissible snippets (according to their (normalized) scalar products).
        good_values = all_snippets[i, i]  / norm_2[i]
        center = 1 #numpy.median(good_values)
        if normalization:
            tgt_values = all_snippets[i, i] / norm_templates[i]
        else:
            tgt_values = all_snippets[i, i]

        bad_values = {}
        neutral_values = {}
        nb_chances = numpy.zeros(all_sizes[i], dtype=numpy.int32)
        for j in range(nb_temp):
            # if (similarity[i, j] >= thr_similarity) and (i != j):
            if i != j and mask_intersect[i, j]:
                if normalization:
                    # Use the normalized scalar products.
                    ref_values = all_snippets[j, j] / norm_templates[j]  # i.e. snippets of j projected on template i
                    values = all_snippets[i, j] / norm_templates[i]  # i.e. snippets of j projected on template i
                    ref2_values = all_snippets[j, i]  / norm_templates[j] # i.e. snippets of i projected on template j
                else:
                    # Use the scalar products (not normalized).
                    ref_values = all_snippets[j, j]  # i.e. snippets of j projected on template i
                    values = all_snippets[i, j]  # i.e. snippets of j projected on template i
                    ref2_values = all_snippets[j, i]  # i.e. snippets of i projected on template j

                selection = ref_values <= values  # i.e. snippets of j on which a fit with template i is tried *before* a fit with template j
                bad_values[j] = all_snippets[i, j][selection]  / norm_2[i]
                selection = ref_values > values   # i.e. snippets of j on which a fit with template i is tried *after* a fit with template j
                neutral_values[j] = all_snippets[i, j][selection] / norm_2[i]

                selection = tgt_values <= ref2_values # i.e. snippets of i on which a fit with template j is tried *before* a fit with template i
                nb_chances[selection] += 1

        bad_values['noise'] = all_snippets[i, 'noise'] / norm_2[i]

        if len(bad_values) > 0:
            all_bad_values = numpy.concatenate([
                values
                for values in bad_values.values()
            ])
        else:
            all_bad_values = numpy.zeros(0, dtype=numpy.float32)

        if len(neutral_values) > 0:
            all_neutral_values = numpy.concatenate([
                values
                for values in neutral_values.values()
            ])
        else:
            all_neutral_values = numpy.zeros(0, dtype=numpy.float32)

        # Then we need to fix a_min and a_max to minimize the error

        very_good_values = good_values

        if fine_amplitude:
            res = scipy.optimize.differential_evolution(score, bounds=[(0,1), (1, 2)], args=(very_good_values, all_bad_values))
            a_min, a_max = res.x
            bounds[count] = [a_min, a_max]
        else:
            a_min, a_max = limits[i]

        error = compute_error(very_good_values, all_bad_values, [a_min, a_max])

        purity_level[count] = min(1, 1 - error)

        mask = (a_min <= good_values) & (good_values <= a_max)
        if numpy.sum(mask) > 0:
            max_nb_chances[count] = numpy.median(nb_chances[mask])
        else:
            max_nb_chances[count] = numpy.nan

        if debug_plots not in ['None', '']:

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(2)
            s = 2 ** 2
            # ...
            linewidth = 0.3
            ax[0].axhline(y=0.0, color='gray', linewidth=linewidth)
            ax[0].axhline(y=a_min, color='tab:blue', linewidth=linewidth)
            ax[0].axhline(y=center, color='gray', linewidth=linewidth)
            ax[0].axhline(y=a_max, color='tab:blue', linewidth=linewidth)
            # Plot neutral amplitudes.
            x = numpy.random.uniform(size=all_neutral_values.size)
            y = all_neutral_values
            color = 'gray'
            ax[0].scatter(x, y, s=s, color=color, alpha=0.1)
            # Plot good amplitudes.
            x1 = numpy.random.uniform(size=good_values.size)
            y = good_values
            color = 'tab:green'
            ax[0].scatter(x1, y, s=s, color=color)
            # ...
            color = 'tab:green'
            for x_, y_ in zip(x1, y):
                if y_ > a_max:
                    ax[0].plot([x_, x_], [a_max, y_], color=color, linewidth=0.3)
                if y_ < a_min:
                    ax[0].plot([x_, x_], [a_min, y_], color=color, linewidth=0.3)
            # ...
            x2 = numpy.random.uniform(size=all_bad_values.size)
            y = all_bad_values
            color = 'tab:red'
            ax[0].scatter(x2, y, s=s, color=color)
            # ...
            color = 'tab:red'
            for x_, y_ in zip(x2, y):
                if center < y_ < a_max:
                    ax[0].plot([x_, x_], [a_max, y_], color=color, linewidth=0.3)
                if a_min < y_ < center:
                    ax[0].plot([x_, x_], [a_min, y_], color=color, linewidth=0.3)
            # Hide the right and top spines
            ax[0].spines['right'].set_visible(False)
            ax[0].spines['top'].set_visible(False)
            # ...
            ax[0].set_ylabel("amplitude")
            # ax.set_xticklabels([])
            ax[0].set_xticks([])
            ax[0].set_title('%g good / %g bad / %g error' %(len(good_values), len(all_bad_values), error))
            
            ax[1].axhline(y=0.0, color='gray', linewidth=linewidth)
            ax[1].axhline(y=a_min, color='tab:blue', linewidth=linewidth)
            ax[1].axhline(y=center, color='gray', linewidth=linewidth)
            ax[1].axhline(y=a_max, color='tab:blue', linewidth=linewidth)
            
            # Plot good amplitudes.
            y = good_values
            r = ax[1].scatter(x1, y, s=s, c=nb_chances)
            fig.colorbar(r, ax=ax[1])

            # Hide the right and top spines
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['top'].set_visible(False)
            ax[1].set_title('Average nb_chances %g' %numpy.mean(nb_chances))
            # ...
            ax[1].set_ylabel("amplitude")
            # ax.set_xticklabels([])
            ax[1].set_xticks([])

            plt.tight_layout()
            # Save and close figure.
            output_path = os.path.join(
                plot_path,
                "amplitude_interval_t{}_e{}_c{}.{}".format(
                    i,
                    clusters_info[i]['electrode_nb'],
                    clusters_info[i]['local_cluster_nb'],
                    debug_plots
                )
            )
            fig.savefig(output_path)
            plt.close(fig)

    comm.Barrier()

    if fine_amplitude:
        bounds = gather_array(bounds, comm, shape=1)
    
    purity_level = gather_array(purity_level, comm)
    max_nb_chances = gather_array(max_nb_chances, comm)

    if SHARED_MEMORY:
        for memory in mpi_memory_1 + mpi_memory_2:
            memory.Free()

    if comm.rank == 0:
        file_name = file_out_suff + '.templates.hdf5'
        hfile = h5py.File(file_name, 'r+', libver='earliest')

        indices = []
        for idx in range(comm.size):
            indices += list(numpy.arange(idx, nb_temp, comm.size))

        indices = numpy.argsort(indices).astype(numpy.int32)

        if fine_amplitude:
            hfile['limits'][:] = bounds[indices]
        if 'purity' not in hfile.keys():
            hfile.create_dataset('purity', data=purity_level[indices])
            hfile.create_dataset('nb_chances', data=max_nb_chances[indices])
        else:
            hfile['purity'][:] = purity_level[indices]
            hfile['nb_chances'][:] = max_nb_chances[indices]
        hfile.close()

    return


def delete_mixtures(params, nb_cpu, nb_gpu, use_gpu):

    data_file = params.data_file
    n_e = params.getint('data', 'N_e')
    n_total = params.nb_channels
    n_t = params.getint('detection', 'N_t')
    template_shift = params.getint('detection', 'template_shift')
    cc_merge = params.getfloat('clustering', 'cc_merge')
    mixtures = []
    norm = n_e * n_t
    # to_remove = []  # TODO remove (not used)?

    filename = params.get('data', 'file_out_suff') + '.overlap-mixtures.hdf5'
    norm_templates = load_data(params, 'norm-templates')
    best_elec = load_data(params, 'electrodes')
    limits = load_data(params, 'limits')
    nodes, edges = get_nodes_and_edges(params)
    inv_nodes = numpy.zeros(n_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.arange(len(nodes))
    has_support = test_if_support(params, '')
    adapted_cc = params.getboolean('clustering', 'adapted_cc')
    adapted_thr = params.getint('clustering', 'adapted_thr')

    overlap = get_overlaps(
        params, extension='-mixtures', erase=True, normalize=True, maxoverlap=False, verbose=False, half=True,
        use_gpu=use_gpu, nb_cpu=nb_cpu, nb_gpu=nb_gpu, decimation=False
    )
    overlap.close()

    SHARED_MEMORY = get_shared_memory_flag(params)

    if SHARED_MEMORY:
        c_overs, mpi_memory_1 = load_data_memshared(
            params, 'overlaps', extension='-mixtures', use_gpu=use_gpu, nb_cpu=nb_cpu, nb_gpu=nb_gpu
        )
    else:
        c_overs = load_data(
            params, 'overlaps', extension='-mixtures'
        )

    if SHARED_MEMORY:
        templates, mpi_memory_2 = load_data_memshared(params, 'templates', normalize=True)
    else:
        templates = load_data(params, 'templates')

    x, n_tm = templates.shape
    nb_temp = int(n_tm // 2)
    offset = n_t - 1
    # merged = [nb_temp, 0]  # TODO remove (not used)?

    if has_support:
        supports = load_data(params, 'supports')
    else:
        supports = {}
        supports = numpy.zeros((nb_temp, n_e), dtype=numpy.bool)
        for t in range(nb_temp):
            elecs = numpy.take(inv_nodes, edges[nodes[best_elec[t]]])
            supports[t, elecs] = True

    overlap_0 = numpy.zeros(nb_temp, dtype=numpy.float32)
    distances = numpy.zeros((nb_temp, nb_temp), dtype=numpy.int32)

    if adapted_cc:
        common_supports = load_data(params, 'common-supports')
        exponents = numpy.exp(-common_supports/adapted_thr)

    for i in range(nb_temp - 1):
        data = c_overs[i].toarray()
        distances[i, i + 1:] = numpy.argmax(data[i + 1:, :], 1)
        distances[i + 1:, i] = distances[i, i + 1:]
        overlap_0[i] = data[i, n_t - 1]

    all_temp = numpy.arange(comm.rank, nb_temp, comm.size)
    sorted_temp = numpy.argsort(norm_templates[:nb_temp])[::-1]
    M = numpy.zeros((2, 2), dtype=numpy.float32)
    V = numpy.zeros((2, 1), dtype=numpy.float32)

    to_explore = range(comm.rank, nb_temp, comm.size)
    if comm.rank == 0:
        to_explore = get_tqdm_progressbar(params, to_explore)

    for count, k in enumerate(to_explore):

        k = sorted_temp[k]
        overlap_k = c_overs[k]
        electrodes = numpy.where(supports[k])[0]
        candidates = {}
        for t1 in range(nb_temp):
            candidates[t1] = []
            masks = numpy.logical_or(supports[t1], supports[t1:])
            masks = numpy.all(masks[:, electrodes], 1)
            if t1 != k:
                for count, t2 in enumerate(range(t1, nb_temp)):
                    is_candidate = masks[count]
                    if is_candidate and t2 != k and t2 != t1:
                        candidates[t1] += [t2]

        been_found = False
        t_k = None

        for i in candidates.keys():
            t_i = None
            if not been_found and len(candidates[i]) > 0:
                overlap_i = c_overs[i]
                M[0, 0] = overlap_0[i]
                V[0, 0] = overlap_k[i, distances[k, i]]
                for j in candidates[i]:
                    t_j = None
                    value = (distances[k, i] - distances[k, j])//2 + offset
                    M[1, 1] = overlap_0[j]
                    M[1, 0] = overlap_i[j, value]
                    M[0, 1] = M[1, 0]
                    V[1, 0] = overlap_k[j, distances[k, j]]
                    try:
                        [a1, a2] = numpy.dot(scipy.linalg.inv(M), V)
                    except Exception:
                        [a1, a2] = [0, 0]
                    a1_lim = limits[i]
                    a2_lim = limits[j]
                    is_a1 = (a1_lim[0] <= a1) and (a1 <= a1_lim[1])
                    is_a2 = (a2_lim[0] <= a2) and (a2 <= a2_lim[1])
                    if is_a1 and is_a2:
                        if t_k is None:
                            t_k = templates[:, k].toarray().ravel()
                        if t_i is None:
                            t_i = templates[:, i].toarray().ravel()
                        if t_j is None:
                            t_j = templates[:, j].toarray().ravel()
                        new_template = (a1 * t_i + a2 * t_j)
                        similarity = numpy.corrcoef(t_k, new_template)[0, 1]
                        local_overlap = numpy.corrcoef(t_i, t_j)[0, 1]
                        if adapted_cc:
                            shared_support = numpy.sum(numpy.logical_or(supports[i], supports[j])*supports[k])
                            exponent = numpy.exp(-shared_support/adapted_thr)
                            mytest1 = similarity**exponent > cc_merge
                            mytest2 = local_overlap**exponents[i, j] < 0.5
                        else:
                            mytest1 = similarity > cc_merge
                            mytest2 = local_overlap < 0.5
                        if mytest1 and mytest2:
                            if k not in mixtures:
                                mixtures += [k]
                                been_found = True
                                break

    sys.stderr.flush()
    to_remove = numpy.unique(numpy.array(mixtures, dtype=numpy.int32))
    to_remove = all_gather_array(to_remove, comm, 0, dtype='int32')

    if len(to_remove) > 0 and comm.rank == 0:
        result = load_data(params, 'clusters')
        slice_templates(params, to_remove)
        slice_clusters(params, result, to_remove=to_remove)

    comm.Barrier()

    del c_overs

    if comm.rank == 0:
        os.remove(filename)

    if SHARED_MEMORY:
        for memory in mpi_memory_1:
            memory.Free()
        for memory in mpi_memory_2:
            memory.Free()

    return [nb_temp, len(to_remove)]
