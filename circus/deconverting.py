import h5py
import numpy as np
import os
import sys

import circus
import circus.shared.files as io

from circus.shared.algorithms import slice_templates, slice_clusters
from circus.shared.files import load_data
from circus.shared.messages import print_and_log, init_logging
from circus.shared.utils import logging, comm, query_yes_no


def load_phy_results(input_path):

    def load_cluster_group(path):

        with open(path, mode='r') as file_:
            cluster_group = file_.read()
        cluster_group = cluster_group.split('\n')
        cluster_group = cluster_group[1:-1]
        cluster_group = {
            int(e.split('\t')[0]): e.split('\t')[1].strip()
            for e in cluster_group
        }

        return cluster_group

    spike_times_path = os.path.join(input_path, 'spike_times.npy')
    spike_times = np.load(spike_times_path)
    spike_times = spike_times.astype(np.int)

    spike_templates_path = os.path.join(input_path, 'spike_templates.npy')
    spike_templates = np.load(spike_templates_path)
    spike_templates = spike_templates.astype(np.int)

    templates_path = os.path.join(input_path, 'templates.npy')
    templates = np.load(templates_path)

    spike_clusters_path = os.path.join(input_path, 'spike_clusters.npy')
    if os.path.isfile(spike_clusters_path):
        spike_clusters = np.load(spike_clusters_path)
        spike_clusters = spike_clusters.astype(np.int)
    else:
        spike_clusters = spike_templates

    cluster_group_path = os.path.join(input_path, 'cluster_group.tsv')
    if os.path.isfile(cluster_group_path):
        cluster_group = load_cluster_group(cluster_group_path)
    else:
        nb_templates = len(templates)
        cluster_group = {
            template_id: 'unsorted'
            for template_id in range(0, nb_templates)
        }

    phy_results = {
        'spike_times': spike_times,
        'spike_templates': spike_templates,
        'spike_clusters': spike_clusters,
        'templates': templates,
        'cluster_group': cluster_group,
    }

    return phy_results


def main(params, nb_cpu, nb_gpu, use_gpu, extension):

    assert comm.rank == 0

    input_extension = extension

    logger = init_logging(params.logfile)
    logger = logging.getLogger('circus.deconverting')
    # Retrieve parameters.
    input_path = params.get('data', 'file_out_suff') + input_extension + '.GUI'
    output_path = params.get('data', 'file_out_suff')
    output_extension = '-deconverted'
    clusters_path = output_path + '.clusters{}.hdf5'.format(output_extension)
    templates_path = output_path + '.templates{}.hdf5'.format(output_extension)
    result_path = output_path + '.result{}.hdf5'.format(output_extension)

    # Check if input path exists.
    if not os.path.isdir(input_path):
        print_and_log([
            "Can't find directory: {}".format(input_path),
            "You must first export your results into the phy format (use the converting method)."
        ], 'error', logger)
        sys.exit(0)

    # Check if results are already present.
    if os.path.isfile(clusters_path) \
    and os.path.isfile(templates_path) \
    and os.path.isfile(result_path):
        print_and_log([
            "Phy results already imported.",
            "Delete the following files to run the deconversion again:",
            "  - {}".format(result_path),
            "  - {}".format(templates_path),
            "  - {}".format(clusters_path),
        ], 'error', logger)
        if query_yes_no("Do you want to delete these files?"):
            os.remove(result_path)
            os.remove(templates_path)
            os.remove(clusters_path)
        else:
            sys.exit(0)

    # Check if results are partially present.
    if os.path.isfile(clusters_path) \
    or os.path.isfile(templates_path) \
    or os.path.isfile(result_path):
        print_and_log([
            "Phy results partially imported.",
            "Delete the following files to be able to run the deconversion:",
        ] + [
            "  - {}".format(path)
            for path in [result_path, templates_path, clusters_path]
            if os.path.isfile(path)
        ], 'error', logger)
        if query_yes_no("Do you want to delete these files?"):
            for path in [result_path, templates_path, clusters_path]:
                if os.path.isfile(path):
                    os.remove(path)
        else:
            sys.exit(0)

    # Log introduction message.
    message = "Importing data from the phy GUI with {} CPUs...".format(nb_cpu)
    print_and_log([message], 'info', logger)

    # Read phy results.
    phy_results = load_phy_results(input_path)
    spike_templates = phy_results['spike_templates']
    spike_clusters = phy_results['spike_clusters']
    cluster_group = phy_results['cluster_group']
    templates = phy_results['templates']

    # print_and_log(["{}".format(phy_results)], 'debug', logger)

    # Read spyking-circus results.
    templates_input_path = output_path + ".templates{}.hdf5".format(input_extension)
    templates_input_file = h5py.File(templates_input_path, mode='r', libver='earliest')
    overlaps = templates_input_file.get('maxoverlap')[:]
    try:
        lag = templates_input_file.get('maxlag')[:]
    except TypeError:  # i.e. 'maxlag' not in HDF5 file.
        lag = np.zeros(overlaps.shape, dtype=np.int32)
    shape = templates_input_file.get('temp_shape')[:]

    # Set correct lag option.
    correct_lag = True

    # Determine association map between spike templates and spike clusters.
    templates_to_clusters = {}
    for spike_template, spike_cluster in zip(spike_templates, spike_clusters):
        templates_to_clusters[spike_template] = spike_cluster

    # Determine association map between spike cluster and default spike template.
    clusters_to_templates = {}
    for spike_cluster in cluster_group:
        clusters_to_templates[spike_cluster] = None

    # # TODO remove templates without spikes.
    # print_and_log([  # TODO remove.
    #     "Removing templates without spikes..."
    # ], 'info', logger)
    # electrodes = io.load_data(params, 'electrodes', extension=input_extension)  # TODO remove duplicate.
    # clusters = io.load_data(params, 'clusters', extension=input_extension)  # TODO remove duplicate
    # for spike_template, _ in templates_to_clusters.items():
    #     # Retrieve the prefered electrode.
    #     elec_ic = electrodes[spike_template]
    #     # Retrieve template index among templates with same prefered electrodeself.
    #     first_index = np.where(electrodes == elec_ic)[0][0]
    #     nic = spike_template - first_index
    #     # Retrieve the cluster label.
    #     label = 'clusters_{}'.format(elec_ic)
    #     # Select the points labelled by the clustering.
    #     mask = clusters[label] > -1
    #     # Retrieve the labels used by the clustering.
    #     tmp = np.unique(clusters[label][mask])
    #     # Retrieve the number of points labelled for both templates.
    #     cluster_label = tmp[nic]
    #     elements = np.where(clusters[label] == cluster_label)[0]
    #     # ...
    #     if len(elements) == 0:
    #         print_and_log([
    #             "template {} has no spike".format(spike_template)
    #         ], 'info', logger)
    # raise NotImplementedError

    to_merge = []
    to_remove = []

    # Do all the merges.
    old_results = io.load_data(params, 'results', extension=input_extension)
    electrodes = io.load_data(params, 'electrodes', extension=input_extension)
    clusters = io.load_data(params, 'clusters', extension=input_extension)
    for spike_template, spike_cluster in templates_to_clusters.items():
        spike_group = cluster_group[spike_cluster]
        if spike_group in ['good', 'unsorted']:
            if clusters_to_templates[spike_cluster] is None:
                clusters_to_templates[spike_cluster] = spike_template
            else:
                # Retrieve pair of templates to merge.
                default_spike_template = clusters_to_templates[spike_cluster]
                one_merge = [default_spike_template, spike_template]
                # Retrieve the prefered electrode for both template.
                elec_ic1 = electrodes[one_merge[0]]
                elec_ic2 = electrodes[one_merge[1]]
                # Retrieve template index among templates with same prefered electrode for both templates.
                first_index1 = np.where(electrodes == elec_ic1)[0][0]
                first_index2 = np.where(electrodes == elec_ic2)[0][0]
                nic1 = one_merge[0] - first_index1
                nic2 = one_merge[1] - first_index2
                # Retrieve the cluster label for both templates.
                label1 = 'clusters_{}'.format(elec_ic1)
                label2 = 'clusters_{}'.format(elec_ic2)
                # Select the points labelled by the clustering for both templates.
                mask1 = clusters[label1] > -1
                mask2 = clusters[label2] > -1
                # Retrieve the labels used by the clustering for both templates.
                tmp1 = np.unique(clusters[label1][mask1])
                tmp2 = np.unique(clusters[label2][mask2])
                # Retrieve the number of points labelled for both templates.
                cluster_label1 = tmp1[nic1]
                cluster_label2 = tmp2[nic2]
                elements1 = np.where(clusters[label1] == cluster_label1)[0]
                elements2 = np.where(clusters[label2] == cluster_label2)[0]
                # Determine index to keep and index to delete.
                if len(elements1) > len(elements2):
                    to_delete = one_merge[1]
                    to_keep = one_merge[0]
                    elec = elec_ic2
                    elements = elements2
                else:
                    to_delete = one_merge[0]
                    to_keep = one_merge[1]
                    elec = elec_ic1
                    elements = elements1
                # print_and_log([
                #     "one_merge: {}".format(one_merge),
                #     "elec_ic1, elec_ic2: {}, {}".format(elec_ic1, elec_ic2),
                #     "nic1, nic2: {}, {}".format(nic1, nic2),
                #     "tmp1, tmp2: {}, {}".format(tmp1, tmp2),
                #     "cluster_label1, cluster_label2: {}, {}".format(cluster_label1, cluster_label2),
                #     "to_keep, to_delete: {}, {}".format(to_keep, to_delete)
                #     ] , 'debug', logger
                # )
                # Merge templates (if necessary).
                if to_keep != to_delete:
                    key1 = 'temp_{}'.format(to_keep)
                    key2 = 'temp_{}'.format(to_delete)
                    amplitudes1 = old_results['amplitudes'][key1]
                    amplitudes2 = old_results['amplitudes'][key2]
                    old_results['amplitudes'][key1] = np.vstack((amplitudes1, amplitudes2))
                    spiketimes1 = old_results['spiketimes'][key1]
                    spiketimes2 = old_results['spiketimes'][key2]
                    if correct_lag:
                        spiketimes2 = spiketimes2.astype(np.int64)
                        spiketimes2 += lag[to_keep, to_delete]
                        spiketimes2 = spiketimes2.astype(np.uint32)
                    old_results['spiketimes'][key1] = np.concatenate((spiketimes1, spiketimes2))
                    indices = np.argsort(old_results['spiketimes'][key1])
                    old_results['amplitudes'][key1] = old_results['amplitudes'][key1][indices]
                    old_results['spiketimes'][key1] = old_results['spiketimes'][key1][indices]
                    old_results['amplitudes'].pop(key2)
                    old_results['spiketimes'].pop(key2)
                # Update internal variables.
                clusters_to_templates[spike_cluster] = to_keep
                to_merge.append((to_keep, to_delete))
        elif spike_group in ['mua', 'noise']:
            to_remove.append(spike_template)
        else:
            message = "Unexpected group value: {}".format(spike_group)
            raise ValueError(message)

    # Remove unmentioned templates (e.g. without any fitted spike).
    old_templates = load_data(params, 'templates', extension=input_extension)
    initial_nb_templates = old_templates.shape[1] // 2
    all_spike_templates = set(range(0, initial_nb_templates))
    mentioned_spike_templates = set(templates_to_clusters.keys())
    unmentioned_spike_templates = list(all_spike_templates - mentioned_spike_templates)
    # print_and_log(["unmentioned templates: {}".format(unmentioned_spike_templates)], 'info', logger)
    to_remove.extend(unmentioned_spike_templates)

    if to_merge == []:
        to_merge = np.zeros((0, 2), dtype=np.int)
    else:
        to_merge = np.array(to_merge)
        to_merge = to_merge[np.lexsort((to_merge[:, 1], to_merge[:, 0])), :]
    to_remove.sort()

    # Log some information.
    nb_merges = to_merge.shape[0]
    nb_removals = len(to_remove)
    final_nb_templates = initial_nb_templates - nb_merges - nb_removals
    print_and_log([
        "Manual sorting with the Python GUI (i.e. phy):",
        "  initial number of templates: {}".format(initial_nb_templates),
        "  number of merges: {}".format(nb_merges),
        "  number of removals: {}".format(nb_removals),
        "  final number of templates: {}".format(final_nb_templates),
    ], 'info', logger)

    # Slice templates.
    to_keep = slice_templates(params, to_merge=to_merge, to_remove=to_remove,
        extension=output_extension, input_extension=extension)
    # print_and_log([
    #     "to_merge (passed to slice_templates: {}".format(to_merge),
    #     "to_remove (passed to slice_templates: {}".format(to_remove),
    #     "to_keep (returned be slice_templates): {}".format(to_keep),
    # ], 'info', logger)

    # Slice clusters.
    light = True
    dataname = 'clusters-light' if light else 'clusters'
    clusters = io.load_data(params, dataname, extension=extension)
    slice_clusters(params, clusters, to_merge=to_merge, to_remove=to_remove,
        extension=output_extension, input_extension=extension, light=light,
        method='new')

    # Finalize result.
    # nb_templates = templates.shape[0]
    # template_indices = np.arange(0, nb_templates)
    # to_delete = list(to_remove)
    # for one_merge in to_merge:
    #     to_delete.append(one_merge[1])
    # to_keep = set(np.unique(template_indices)) - set(to_delete)
    # to_keep = np.array(list(to_keep))
    # to_keep = np.sort(to_keep)
    # # Would be correct if we could sort 'to_keep' in 'slice_templates'.
    # print_and_log([
    #     "to_keep: {}".format(to_keep)
    # ], 'idebug', logger)
    new_results = {
        'spiketimes': {},
        'amplitudes': {},
    }
    for k, template_index in enumerate(to_keep):
        old_key = 'temp_{}'.format(template_index)
        new_key = 'temp_{}'.format(k)
        new_results['spiketimes'][new_key] = old_results['spiketimes'].pop(old_key)
        new_results['amplitudes'][new_key] = old_results['amplitudes'].pop(old_key)
        # Check if the number of spikes is not equal to 0.
        nb_spikes = len(new_results['spiketimes'][new_key])
        if nb_spikes == 0:
            print_and_log([
                "{} - template {} has no spikes".format(k, template_index)
            ], 'error', logger)
    keys = ['spiketimes', 'amplitudes']
    # TODO add support for [fitting] collect_all=True (not supported).
    # Save new result to output file.
    result_output_path = output_path + ".result{}.hdf5".format(output_extension)
    result_output_file = h5py.File(result_output_path, mode='w', libver='earliest')
    for key in keys:
        result_output_file.create_group(key)
        for temp in new_results[key].keys():
            tmp_path = "{}/{}".format(key, temp)
            result_output_file.create_dataset(tmp_path, data=new_results[key][temp])
    result_output_file.close()
    # Add additional information to templates file.
    templates_output_path = output_path + ".templates{}.hdf5".format(output_extension)
    templates_output_file = h5py.File(templates_output_path, mode='r+', libver='earliest')
    new_shape = (len(to_keep), len(to_keep))
    version = templates_output_file.create_dataset('version', data=np.array(circus.__version__.split('.'), dtype=np.int32))
    maxoverlaps = templates_output_file.create_dataset('maxoverlap', shape=new_shape, dtype=np.float32)
    maxlag = templates_output_file.create_dataset('maxlag', shape=new_shape, dtype=np.int32)
    for k, index in enumerate(to_keep):
        maxoverlaps[k, :] = overlaps[index, to_keep]
        maxlag[k, :] = lag[index, to_keep]
    templates_output_file.close()

    # Log conclusion message.
    message = "Data from the phy GUI imported."
    print_and_log([message], 'info', logger)

    return
