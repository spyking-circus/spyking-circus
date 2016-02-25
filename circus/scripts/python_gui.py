#!/usr/bin/env python
import os
import sys
import subprocess
import pkg_resources
import circus
import tempfile
import numpy, h5py
from circus.shared.files import print_error, print_info, print_and_log, write_datasets, get_results, read_probe, load_data, get_nodes_and_edges, load_data

def main():

    argv = sys.argv

    if len(sys.argv) < 2:
        print_and_log(['No data file!'], 'error', params)
        sys.exit(0)


    filename       = os.path.abspath(sys.argv[1])

    if len(sys.argv) == 2:
        filename   = os.path.abspath(sys.argv[1])
        extension  = ''
    elif len(sys.argv) == 3:
        filename   = os.path.abspath(sys.argv[1])
        extension  = sys.argv[2]

    params         = circus.shared.utils.io.load_parameters(filename)
    sampling_rate  = float(params.getint('data', 'sampling_rate'))
    data_dtype     = params.get('data', 'data_dtype')
    gain           = 1
    file_out_suff  = params.get('data', 'file_out_suff')
    data_offset    = params.getint('data', 'data_offset')
    probe          = read_probe(params)
    output_path    = params.get('data', 'file_out_suff') + '.GUI'
    N_e            = params.getint('data', 'N_e')
    N_t            = params.getint('data', 'N_t')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    def generate_matlab_mapping(probe):
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
        return spikes[idx]


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


    def write_pcs(path, params, extension, spikes):

        max_loc_channel = get_max_loc_channel(params)
        clusters        = load_data(params, 'clusters', extension)
        best_elec       = clusters['electrodes']
        nb_features     = params.getint('whitening', 'output_dim')
        nodes, edges    = get_nodes_and_edges(params)
        templates       = load_data(params, 'templates', extension)
        N_tm            = templates.shape[1]//2
        pc_features     = numpy.zeros((0, nb_features, max_loc_channel), dtype=numpy.float32)
        pc_features_ind = numpy.zeros((N_tm, max_loc_channel), dtype=numpy.int32)

        for count, elec in enumerate(best_elec):
            nb_loc                = len(edges[elec])
            pc_features_ind[count, numpy.arange(nb_loc)] = edges[elec]

        for target in xrange(N_tm):
            elec     = clusters['electrodes'][target]
            nic      = target - numpy.where(clusters['electrodes'] == elec)[0][0]
            mask     = clusters['clusters_' + str(elec)] > -1
            tmp      = numpy.unique(clusters['clusters_' + str(elec)][mask])
            indices  = numpy.where(clusters['clusters_' + str(elec)] == tmp[nic])[0]
            x, y        = clusters['data_' + str(elec)][indices, :].shape
            data        = clusters['data_' + str(elec)][indices, :].reshape(x, nb_features, y//nb_features)
            difference  = max_loc_channel - data.shape[2]
            to_fill     = numpy.zeros((x, nb_features, difference))
            to_write    = numpy.concatenate((data, to_fill), axis=2)
            pc_features = numpy.concatenate((pc_features, to_write), axis=0)

        
        numpy.save(os.path.join(output_path, 'pc_features'), pc_features) # nspikes, nfeat, n_loc_chan
        numpy.save(os.path.join(output_path, 'pc_feature_ind'), pc_features_ind) #n_templates, n_loc_chan

    print_and_log(["Exporting data for the phy GUI..."], 'info', params)
    
    numpy.save(os.path.join(output_path, 'whitening_mat'), numpy.linalg.inv(load_data(params, 'spatial_whitening')))
    numpy.save(os.path.join(output_path, 'channel_positions'), generate_matlab_mapping(probe))
    nodes, edges   = get_nodes_and_edges(params)
    numpy.save(os.path.join(output_path, 'channel_map'), nodes)
    similarities = h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r+', libver='latest').get('maxoverlap')
    numpy.save(os.path.join(output_path, 'templates_similarities'), similarities)

    spikes = write_results(output_path, params, extension)    
    write_templates(output_path, params, extension)
    write_pcs(output_path, params, extension, spikes)
    



    

