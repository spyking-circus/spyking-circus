from circus.shared.utils import *
import circus.shared.files as io
import os
# import os.path as op
import shutil
# import circus
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

from circus.shared.probes import get_nodes_and_edges
from colorama import Fore
from circus.shared.messages import print_and_log, init_logging
from circus.shared.utils import query_yes_no, apply_patch_for_similarities, test_if_support, test_if_purity


def get_rpv(spikes, sampling_rate, duration=5e-3):
    idx = numpy.where(numpy.diff(spikes) < int(duration * sampling_rate))[0]
    if len(spikes) > 0:
        return len(idx) / float(len(spikes))
    else:
        return numpy.inf


def main(params, nb_cpu, nb_gpu, use_gpu, extension):

    _ = init_logging(params.logfile)
    logger = logging.getLogger('circus.converting')
    data_file = params.data_file
    file_out_suff = params.get('data', 'file_out_suff')
    probe = params.probe
    output_path = params.get('data', 'file_out_suff') + extension + '.GUI'
    N_e = params.getint('data', 'N_e')
    prelabelling = params.getboolean('converting', 'prelabelling')
    N_t = params.getint('detection', 'N_t')
    erase_all = params.getboolean('converting', 'erase_all')
    export_pcs = params.get('converting', 'export_pcs')
    export_all = params.getboolean('converting', 'export_all')
    sparse_export = params.getboolean('converting', 'sparse_export')
    rpv_threshold = params.getfloat('converting', 'rpv_threshold')
    if export_all and not params.getboolean('fitting', 'collect_all'):
        if comm.rank == 0:
            print_and_log(['Export unfitted spikes only if [fitting] collect_all is True'], 'error', logger)
        sys.exit(0)

    def generate_mapping(probe):
        p = {}
        positions = []
        nodes = []
        shanks = []
        for key in probe['channel_groups'].keys():
            p.update(probe['channel_groups'][key]['geometry'])
            nodes += probe['channel_groups'][key]['channels']
            positions += [p[channel] for channel in probe['channel_groups'][key]['channels']]
            shanks += [key] * len(probe['channel_groups'][key]['channels'])
        positions = numpy.array(positions)
        shanks = numpy.array(shanks)
        return positions, shanks

    def get_max_loc_channel(params, extension):
        if test_if_support(params, extension):
            supports = io.load_data(params, 'supports', extension)
            max_loc_channel = numpy.sum(supports, 1).max()
        else:
            nodes, edges = get_nodes_and_edges(params)
            max_loc_channel = 0
            for key in edges.keys():
                if len(edges[key]) > max_loc_channel:
                    max_loc_channel = len(edges[key])
        return max_loc_channel

    def write_results(path, params, extension):
        result = io.get_results(params, extension)
        spikes = [numpy.zeros(0, dtype=numpy.uint64)]
        clusters = [numpy.zeros(0, dtype=numpy.uint32)]
        amplitudes = [numpy.zeros(0, dtype=numpy.double)]
        N_tm = len(result['spiketimes'])

        has_purity = test_if_purity(params, extension)
        rpvs = []

        if prelabelling:
            labels = []
            norms = io.load_data(params, 'norm-templates', extension)
            norms = norms[:len(norms) // 2]
            if has_purity:
                purity = io.load_data(params, 'purity', extension)

        for key in result['spiketimes'].keys():
            temp_id = int(key.split('_')[-1])
            myspikes = result['spiketimes'].pop(key).astype(numpy.uint64)
            spikes.append(myspikes)
            myamplitudes = result['amplitudes'].pop(key).astype(numpy.double)
            amplitudes.append(myamplitudes[:, 0])
            clusters.append(temp_id*numpy.ones(len(myamplitudes), dtype=numpy.uint32))
            rpv = get_rpv(myspikes, params.data_file.sampling_rate)
            rpvs += [[temp_id, rpv]]
            if prelabelling:
                if has_purity:
                    if rpv <= rpv_threshold:
                        if purity[temp_id] > 0.75:
                            labels += [[temp_id, 'good']]
                    else:
                        if purity[temp_id] > 0.75:
                            labels += [[temp_id, 'mua']]
                        else:
                            labels += [[temp_id, 'noise']]
                else:
                    median_amp = numpy.median(myamplitudes[:, 0])
                    std_amp = numpy.std(myamplitudes[:, 0])
                    if rpv <= rpv_threshold and numpy.abs(median_amp - 1) < 0.25:
                        labels += [[temp_id, 'good']]
                    else:
                        if median_amp < 0.5:
                            labels += [[temp_id, 'mua']]
                        elif norms[temp_id] < 0.1:
                            labels += [[temp_id, 'noise']]

        if export_all:
            print_and_log(["Last %d templates are unfitted spikes on all electrodes" % N_e], 'info', logger)
            garbage = io.load_data(params, 'garbage', extension)
            for key in garbage['gspikes'].keys():
                elec_id = int(key.split('_')[-1])
                data = garbage['gspikes'].pop(key).astype(numpy.uint64)
                spikes.append(data)
                amplitudes.append(numpy.ones(len(data)))
                clusters.append((elec_id + N_tm)*numpy.ones(len(data), dtype=numpy.uint32))

        if prelabelling:
            f = open(os.path.join(output_path, 'cluster_group.tsv'), 'w')
            f.write('cluster_id\tgroup\n')
            for l in labels:
                f.write('%s\t%s\n' % (l[0], l[1]))
            f.close()

        # f = open(os.path.join(output_path, 'cluster_rpv.tsv'), 'w')
        # f.write('cluster_id\trpv\n')
        # for l in rpvs:
        #     f.write('%s\t%s\n' % (l[0], l[1]))
        # f.close()

        spikes = numpy.concatenate(spikes).astype(numpy.uint64)
        amplitudes = numpy.concatenate(amplitudes).astype(numpy.double)
        clusters = numpy.concatenate(clusters).astype(numpy.uint32)

        idx = numpy.argsort(spikes)
        numpy.save(os.path.join(output_path, 'spike_templates'), clusters[idx])
        numpy.save(os.path.join(output_path, 'spike_times'), spikes[idx])
        numpy.save(os.path.join(output_path, 'amplitudes'), amplitudes[idx])
        return

    def write_templates(path, params, extension):

        max_loc_channel = get_max_loc_channel(params, extension)
        templates = io.load_data(params, 'templates', extension)
        N_tm = templates.shape[1] // 2
        nodes, edges = get_nodes_and_edges(params)

        if sparse_export:
            n_channels_max = 0
            for t in range(N_tm):
                data = numpy.sum(numpy.sum(templates[:, t].toarray().reshape(N_e, N_t), 1) != 0) 
                if data > n_channels_max:
                    n_channels_max = data
        else:
            n_channels_max = N_e

        if export_all:
            to_write_sparse = numpy.zeros((N_tm + N_e, N_t, n_channels_max), dtype=numpy.float32)
            mapping_sparse = -1 * numpy.ones((N_tm + N_e, n_channels_max), dtype=numpy.int32)
        else:
            to_write_sparse = numpy.zeros((N_tm, N_t, n_channels_max), dtype=numpy.float32)
            mapping_sparse = -1 * numpy.ones((N_tm, n_channels_max), dtype=numpy.int32)

        has_purity = test_if_purity(params, extension)
        if has_purity:
            purity = io.load_data(params, 'purity', extension)
            f = open(os.path.join(output_path, 'cluster_purity.tsv'), 'w')
            f.write('cluster_id\tpurity\n')
            for i in range(N_tm):
                f.write('%d\t%g\n' % (i, purity[i]))
            f.close()

        for t in range(N_tm):
            tmp = templates[:, t].toarray().reshape(N_e, N_t).T
            x, y = tmp.nonzero()
            nb_loc = len(numpy.unique(y))
                
            if sparse_export:
                all_positions = numpy.zeros(y.max()+1, dtype=numpy.int32)
                all_positions[numpy.unique(y)] = numpy.arange(nb_loc, dtype=numpy.int32)
                pos = all_positions[y]
                to_write_sparse[t, x, pos] = tmp[x, y]
                mapping_sparse[t, numpy.arange(nb_loc)] = numpy.unique(y)
            else:
                pos = y
                to_write_sparse[t, x, pos] = tmp[x, y]

        if export_all:
            garbage = io.load_data(params, 'garbage', extension)
            for t in range(N_tm, N_tm + N_e):
                elec = t - N_tm
                spikes = garbage['gspikes'].pop('elec_%d' % elec).astype(numpy.int64)
                spikes = numpy.random.permutation(spikes)[:100]
                mapping_sparse[t, 0] = t - N_tm
                waveform = io.get_stas(params, times_i=spikes, labels_i=np.ones(len(spikes)), src=elec, neighs=[elec], nodes=nodes, mean_mode=True)
                
                nb_loc = 1

                if sparse_export:
                    to_write_sparse[t, :, 0] = waveform
                else:
                    to_write_sparse[t, :, elec] = waveform

        numpy.save(os.path.join(output_path, 'templates'), to_write_sparse)

        if sparse_export:
            numpy.save(os.path.join(output_path, 'template_ind'), mapping_sparse)

        return N_tm

    def write_pcs(path, params, extension, N_tm, mode=0):

        spikes = numpy.load(os.path.join(output_path, 'spike_times.npy'))
        labels = numpy.load(os.path.join(output_path, 'spike_templates.npy'))
        max_loc_channel = get_max_loc_channel(params, extension)
        nb_features = params.getint('whitening', 'output_dim')
        sign_peaks = params.get('detection', 'peaks')
        nodes, edges = get_nodes_and_edges(params)
        N_total = params.getint('data', 'N_total')
        has_support = test_if_support(params, extension)
        if has_support:
            supports = io.load_data(params, 'supports', extension)
        else:
            inv_nodes = numpy.zeros(N_total, dtype=numpy.int32)
            inv_nodes[nodes] = numpy.arange(len(nodes))

        if export_all:
            nb_templates = N_tm + N_e
        else:
            nb_templates = N_tm

        pc_features_ind = numpy.zeros((nb_templates, max_loc_channel), dtype=numpy.int32)            
        best_elec = io.load_data(params, 'electrodes', extension)
        if export_all:
            best_elec = numpy.concatenate((best_elec, numpy.arange(N_e)))

        if has_support:
            for count, support in enumerate(supports):
                nb_loc = numpy.sum(support)
                pc_features_ind[count, numpy.arange(nb_loc)] = numpy.where(support == True)[0]
        else:
            for count, elec in enumerate(best_elec):
                nb_loc = len(edges[nodes[elec]])
                pc_features_ind[count, numpy.arange(nb_loc)] = inv_nodes[edges[nodes[elec]]]

        if sign_peaks in ['negative', 'both']:
            basis_proj, basis_rec = io.load_data(params, 'basis')
        elif sign_peaks in ['positive']:
            basis_proj, basis_rec = io.load_data(params, 'basis-pos')

        to_process = numpy.arange(comm.rank, nb_templates, comm.size)

        all_offsets = numpy.zeros(nb_templates, dtype=numpy.int32)
        for target in range(nb_templates):
            if mode == 0:
                all_offsets[target] = len(numpy.where(labels == target)[0])
            elif mode == 1:
                all_offsets[target] = min(500, len(numpy.where(labels == target)[0]))

        all_paddings = numpy.concatenate(([0], numpy.cumsum(all_offsets)))
        total_pcs = numpy.sum(all_offsets)

        pc_file = os.path.join(output_path, 'pc_features.npy')
        pc_file_ids = os.path.join(output_path, 'pc_feature_spike_ids.npy')

        from numpy.lib.format import open_memmap

        if comm.rank == 0:
            pc_features = open_memmap(pc_file, shape=(total_pcs, nb_features, max_loc_channel), dtype=numpy.float32, mode='w+')
            if mode == 1:
                pc_ids = open_memmap(pc_file_ids, shape=(total_pcs, ), dtype=numpy.int32, mode='w+')

        comm.Barrier()
        pc_features = open_memmap(pc_file, mode='r+')
        if mode == 1:
            pc_ids = open_memmap(pc_file_ids, mode='r+')

        to_explore = range(comm.rank, nb_templates, comm.size)

        if comm.rank == 0:
            to_explore = get_tqdm_progressbar(params, to_explore)

        all_idx = numpy.zeros(0, dtype=numpy.int32)
        for gcount, target in enumerate(to_explore):

            count = all_paddings[target]
            
            if mode == 1:
                idx = numpy.random.permutation(numpy.where(labels == target)[0])[:500]
                pc_ids[count:count+len(idx)] = idx
            elif mode == 0:
                idx = numpy.where(labels == target)[0]

            elec = best_elec[target]

            if has_support:
                if target >= len(supports):
                    indices = [target - N_tm]
                else:
                    indices = numpy.where(supports[target])[0]
            else:
                indices = inv_nodes[edges[nodes[elec]]]
            labels_i = target*numpy.ones(len(idx))
            times_i = numpy.take(spikes, idx).astype(numpy.int64)
            sub_data = io.get_stas(params, times_i, labels_i, elec, neighs=indices, nodes=nodes, auto_align=False)
            
            pcs = numpy.dot(sub_data, basis_proj)
            pcs = numpy.swapaxes(pcs, 1, 2)
            if mode == 0:
                pc_features[idx, :, :len(indices)] = pcs                    
            elif mode == 1:
                pc_features[count:count+len(idx), :, :len(indices)] = pcs

        comm.Barrier()

        if comm.rank == 0:
            numpy.save(os.path.join(output_path, 'pc_feature_ind'), pc_features_ind.astype(numpy.uint32))  # n_templates, n_loc_chan

    do_export = True
    if comm.rank == 0:
        if os.path.exists(output_path):
            if not erase_all:
                do_export = query_yes_no(Fore.WHITE + "Export already made! Do you want to erase everything?", default=None)

            if do_export:
                if os.path.exists(os.path.abspath('.phy')):
                    shutil.rmtree(os.path.abspath('.phy'))
                shutil.rmtree(output_path)
        if do_export:
            comm.bcast(numpy.array([1], dtype=numpy.int32), root=0)
        else:
            comm.bcast(numpy.array([0], dtype=numpy.int32), root=0)
    else:
        do_export = bool(comm.bcast(numpy.array([0], dtype=numpy.int32), root=0))

    comm.Barrier()

    if do_export:

        apply_patch_for_similarities(params, extension)

        if comm.rank == 0:
            os.makedirs(output_path)
            print_and_log(["Exporting data for the phy GUI with %d CPUs..." % nb_cpu], 'info', logger)

            if params.getboolean('whitening', 'spatial'):
                whitening_mat = io.load_data(params, 'spatial_whitening').astype(numpy.double)
                numpy.save(os.path.join(output_path, 'whitening_mat'), whitening_mat)
                numpy.save(os.path.join(output_path, 'whitening_mat_inv'), numpy.linalg.inv(whitening_mat))
            else:
                numpy.save(os.path.join(output_path, 'whitening_mat'), numpy.eye(N_e))

            positions, shanks = generate_mapping(probe)
            numpy.save(os.path.join(output_path, 'channel_positions'), positions.astype(numpy.double))
            numpy.save(os.path.join(output_path, 'channel_shanks'), shanks.astype(numpy.double))
            nodes, edges = get_nodes_and_edges(params)
            numpy.save(os.path.join(output_path, 'channel_map'), nodes.astype(numpy.int32))

            write_results(output_path, params, extension) 
 
            N_tm = write_templates(output_path, params, extension)

            template_file = h5py.File(file_out_suff + '.templates%s.hdf5' % extension, 'r', libver='earliest')
            similarities = template_file.get('maxoverlap')[:]
            template_file.close()
            norm = N_e*N_t

            if export_all:
                to_write = numpy.zeros((N_tm + N_e, N_tm + N_e), dtype=numpy.single)
                to_write[:N_tm, :N_tm] = (similarities[:N_tm, :N_tm] / norm).astype(numpy.single)
            else:
                to_write = (similarities[:N_tm, :N_tm] / norm).astype(numpy.single)
            numpy.save(os.path.join(output_path, 'similar_templates'), to_write)

            comm.bcast(numpy.array([N_tm], dtype=numpy.int32), root=0)

        else:
            N_tm = int(comm.bcast(numpy.array([0], dtype=numpy.int32), root=0))

        comm.Barrier()

        make_pcs = 2
        if comm.rank == 0:

            if export_pcs == 'prompt':
                key = ''
                while key not in ['a', 's', 'n']:
                    print(Fore.WHITE + "Do you want SpyKING CIRCUS to export PCs? (a)ll / (s)ome / (n)o")
                    key = raw_input('')
            else:
                key = export_pcs

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
        if make_pcs < 2:
            write_pcs(output_path, params, extension, N_tm, make_pcs)

        supported_by_phy = ['raw_binary', 'mcs_raw_binary', 'mda']
        file_format = data_file.description
        gui_params = {}

        if file_format in supported_by_phy:
            if not params.getboolean('data', 'overwrite'):
                gui_params['dat_path'] = r"%s" % params.get('data', 'data_file_no_overwrite')
            else:
                if params.get('data', 'stream_mode') == 'multi-files':
                    data_file = params.get_data_file(source=True, has_been_created=False)
                    gui_params['dat_path'] = "["
                    for f in data_file.get_file_names():
                        gui_params['dat_path'] += 'r"%s", ' % f
                    gui_params['dat_path'] += "]"
                else:
                    gui_params['dat_path'] = 'r"%s"' % params.get('data', 'data_file')
        else:
            gui_params['dat_path'] = 'giverandomname.dat'
        gui_params['n_channels_dat'] = params.nb_channels
        gui_params['n_features_per_channel'] = 5
        gui_params['dtype'] = data_file.data_dtype
        if 'data_offset' in data_file.params.keys():
            gui_params['offset'] = data_file.data_offset
        gui_params['sample_rate'] = params.rate
        gui_params['dir_path'] = output_path
        gui_params['hp_filtered'] = True

        f = open(os.path.join(output_path, 'params.py'), 'w')
        for key, value in gui_params.items():
            if key in ['dir_path', 'dtype']:
                f.write('%s = r"%s"\n' % (key, value))
            else:
                f.write("%s = %s\n" % (key, value))
        f.close()
