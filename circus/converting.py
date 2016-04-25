"""Conversion script from template matching files to Kwik.

There are several improvements:

* Don't load all features and masks in memory (need for KwikCreator to accept
  features and masks as generators, not lists).

"""

from .shared.utils import *
import os
import os.path as op
import shutil
import circus

import logging

import numpy as np
import h5py
from circus.shared.files import print_error, print_info, print_and_log, write_datasets, get_results, read_probe, load_data, get_nodes_and_edges, load_data, get_stas
from circus.shared.utils import get_progressbar

def main(filename, params, nb_cpu, nb_gpu, use_gpu):

    params         = circus.shared.utils.io.load_parameters(filename)
    sampling_rate  = float(params.getint('data', 'sampling_rate'))
    data_dtype     = params.get('data', 'data_dtype')
    file_out_suff  = params.get('data', 'file_out_suff')
    data_offset    = params.getint('data', 'data_offset')
    probe          = read_probe(params)
    output_path    = params.get('data', 'file_out_suff') + '.GUI'
    extension      = ''
    N_e            = params.getint('data', 'N_e')
    N_t            = params.getint('data', 'N_t')

    def generate_mapping(probe):
        p         = {}
        positions = []
        nodes     = []
        for key in probe['channel_groups'].keys():
            p.update(probe['channel_groups'][key]['geometry'])
            nodes     +=  probe['channel_groups'][key]['channels']
            positions += [p[channel] for channel in probe['channel_groups'][key]['channels']]
        idx       = numpy.argsort(nodes)
        positions = numpy.array(positions)[idx]
        return positions

    def get_max_loc_channel(params):
        nodes, edges    = circus.shared.utils.io.get_nodes_and_edges(params)
        max_loc_channel = 0
        for key in edges.keys():
            if len(edges[key]) > max_loc_channel:
                max_loc_channel = len(edges[key])
        return max_loc_channel

    def write_results(path, params, extension):
        result     = get_results(params, extension)
        spikes     = numpy.zeros(0, dtype=numpy.int64)
        clusters   = numpy.zeros(0, dtype=numpy.int32)
        amplitudes = numpy.zeros(0, dtype=numpy.float32)
        for key in result['spiketimes'].keys():
            temp_id    = int(key.split('_')[-1])
            data       = result['spiketimes'].pop(key)
            spikes     = numpy.concatenate((spikes, data.astype(numpy.int64)))
            amplitudes = numpy.concatenate((amplitudes, result['amplitudes'][key][:, 0]))
            clusters   = numpy.concatenate((clusters, temp_id*numpy.ones(len(data), dtype=numpy.int32)))
        
        idx = numpy.argsort(spikes)

        numpy.save(os.path.join(output_path, 'spike_templates'), clusters[idx])
        numpy.save(os.path.join(output_path, 'spike_times'), spikes[idx])
        numpy.save(os.path.join(output_path, 'amplitudes'), amplitudes[idx])
        return spikes[idx], clusters[idx]


    def write_templates(path, params, extension):

        max_loc_channel = get_max_loc_channel(params)
        templates       = load_data(params, 'templates', extension)
        N_tm            = templates.shape[1]//2
        to_write        = numpy.zeros((N_tm, N_t, N_e), dtype=numpy.float32)
        mapping         = numpy.zeros((N_tm, max_loc_channel), dtype=numpy.int32)

        for t in xrange(N_tm):
            tmp  = templates[:, t].toarray().reshape(N_e, N_t).T
            x, y = tmp.nonzero()
            to_write[t, x, y]                = tmp[x, y] 
            nb_loc                           = len(numpy.unique(y))
            mapping[t, numpy.arange(nb_loc)] = numpy.unique(y)

        numpy.save(os.path.join(output_path, 'templates'), to_write)
        numpy.save(os.path.join(output_path, 'templates_ind'), mapping)
        return N_tm


    def write_pcs(path, params, comm, extension, spikes, labels, mode=2):

        max_loc_channel = get_max_loc_channel(params)
        nb_features     = params.getint('whitening', 'output_dim')
        nodes, edges    = get_nodes_and_edges(params)
        N_total         = params.getint('data', 'N_total')
        templates       = load_data(params, 'templates', extension)
        N_tm            = templates.shape[1]//2
        pc_features_ind = numpy.zeros((N_tm, max_loc_channel), dtype=numpy.int32)
        clusters        = load_data(params, 'clusters', extension)
        best_elec       = clusters['electrodes']
        inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
        inv_nodes[nodes] = numpy.argsort(nodes)

        for count, elec in enumerate(best_elec):
            nb_loc                = len(edges[nodes[elec]])
            pc_features_ind[count, numpy.arange(nb_loc)] = inv_nodes[edges[nodes[elec]]]

        basis_proj, basis_rec = load_data(params, 'basis')

        if mode == 0:
            nb_pcs = len(spikes)
        elif mode == 1:
            nb_pcs = 0
            for target in xrange(N_tm):
                nb_pcs += min(500, len(numpy.where(labels == target)[0]))

        pc_features = numpy.zeros((nb_pcs, nb_features, max_loc_channel), dtype=numpy.float32)
        count       = 0

        pbar    = get_progressbar(N_tm)
        all_idx = numpy.zeros(0, dtype=numpy.int32)
        for target in xrange(N_tm):
            if mode == 1:
                idx     = numpy.random.permutation(numpy.where(labels == target)[0])[:500]
                all_idx = numpy.concatenate((all_idx, idx))
            elif mode == 0:
                idx  = numpy.where(labels == target)[0]
            elec     = best_elec[target]
            indices  = inv_nodes[edges[nodes[elec]]]
            labels_i = target*numpy.ones(len(idx))
            times_i  = numpy.take(spikes, idx)
            sub_data = get_stas(params, times_i, labels_i, elec, neighs=indices, nodes=nodes)
            pcs      = numpy.dot(sub_data, basis_proj)
            pcs      = numpy.swapaxes(pcs, 1,2)
            if mode == 1:
                pc_features[count:count+len(idx), :, :len(indices)] = pcs                    
                count += len(idx)
            else:
                pc_features[idx, :, :len(indices)] = pcs

            pbar.update(target)
        pbar.finish()
        
        numpy.save(os.path.join(output_path, 'pc_features'), pc_features) # nspikes, nfeat, n_loc_chan
        numpy.save(os.path.join(output_path, 'pc_feature_ind'), pc_features_ind) #n_templates, n_loc_chan
        if mode == 1:
            numpy.save(os.path.join(output_path, 'pc_feature_spike_ids'), all_idx)


    do_export = True
    if comm.rank == 0:
        if os.path.exists(output_path):
            key = ''
            while key not in ['y', 'n']:
                print("Export already made! Do you want to erase everything? (y)es / (n)o ")
                key = raw_input('')
                if key =='y':
                    do_export = True
                    comm.bcast(numpy.array([1], dtype=numpy.int32), root=0)
                else:
                    do_export = False
                    comm.bcast(numpy.array([0], dtype=numpy.int32), root=0)
            if do_export:
                shutil.rmtree(output_path)
    else:
        if os.path.exists(output_path):
            do_export = bool(comm.bcast(numpy.array([0], dtype=numpy.int32), root=0))

    if do_export:

        if comm.rank == 0:
            os.makedirs(output_path)
            print_and_log(["Exporting data for the phy GUI with %d CPUs..." %nb_cpu], 'info', params)
        
        comm.Barrier()
        
        if comm.rank == 0:
            numpy.save(os.path.join(output_path, 'whitening_mat'), load_data(params, 'spatial_whitening'))
            numpy.save(os.path.join(output_path, 'channel_positions'), generate_mapping(probe))
            nodes, edges   = get_nodes_and_edges(params)
            numpy.save(os.path.join(output_path, 'channel_map'), nodes)

            spikes, clusters = write_results(output_path, params, extension)    
            N_tm = write_templates(output_path, params, extension)
            similarities = h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r+', libver='latest').get('maxoverlap')
            norm = params.getint('data', 'N_e')*params.getint('data', 'N_t')
            numpy.save(os.path.join(output_path, 'similar_templates'), similarities[:N_tm, :N_tm]/norm)
        
        comm.Barrier()

        make_pcs = 2
        if comm.rank == 0:

            key = ''
            while key not in ['a', 's', 'n']:
                print ("Do you want SpyKING CIRCUS to export PCs? (a)ll / (s)ome / (n)o")
                key = raw_input('')
                
            if key == 'a':
                make_pcs = 0
                comm.bcast(numpy.array([0], dtype=numpy.int32), root=0)
            elif key == 's':
                make_pcs = 1
                comm.bcast(numpy.array([1], dtype=numpy.int32), root=0)
            elif key == 'n':
                comm.bcast(numpy.array([2], dtype=numpy.int32), root=0)
                if os.path.exists(os.path.join(output_path, 'pc_features.npy')):
                    os.remove(os.path.join(output_path, 'pc_features.npy'))
                if os.path.exists(os.path.join(output_path, 'pc_feature_ind.npy')):
                    os.remove(os.path.join(output_path, 'pc_feature_ind.npy'))
        else:
            make_pcs = comm.bcast(numpy.array([0], dtype=numpy.int32), root=0)
            make_pcs = make_pcs[0]

        comm.Barrier()
        print comm.rank, make_pcs
        #write_pcs(output_path, params, comm, extension, spikes, clusters, make_pcs)
