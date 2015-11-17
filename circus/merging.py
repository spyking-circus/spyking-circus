from .shared.utils import *
import pylab

def main(filename, params, nb_cpu, nb_gpu, use_gpu):

    import h5py
    parallel_hdf5 = h5py.get_config().mpi

    #################################################################
    sampling_rate  = params.getint('data', 'sampling_rate')
    N_e            = params.getint('data', 'N_e')
    N_t            = params.getint('data', 'N_t')
    N_total        = params.getint('data', 'N_total')
    file_out_suff  = params.get('data', 'file_out_suff')
    file_out       = params.get('data', 'file_out')
    cc_gap         = params.getfloat('merging', 'cc_gap')
    cc_overlap     = params.getfloat('merging', 'cc_overlap')
    cc_bin         = params.getfloat('merging', 'cc_bin')
    cc_average     = params.getfloat('merging', 'cc_average')
    make_plots     = params.getboolean('merging', 'make_plots')
    plot_path      = os.path.join(params.get('data', 'data_file_noext'), 'plots')
    
    bin_size       = int(cc_bin * sampling_rate * 1e-3)
    delay_average  = int(cc_average/cc_bin)
    max_delay      = max(100, cc_average)

    if comm.rank == 0:
        templates      = io.load_data(params, 'templates')[:]
        limits         = io.load_data(params, 'limits')
        clusters       = io.load_data(params, 'clusters')
        result         = io.load_data(params, 'results')
        overlap        = h5py.File(file_out_suff + '.overlap.hdf5').get('maxoverlap')[:]
        overlap       /= templates.shape[0] * templates.shape[1]

        io.purge(file_out_suff, '-merged')
        if make_plots:
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            io.purge(plot_path, 'merging')

        to_average     = range(max_delay + 1 - delay_average, max_delay + 1 + delay_average)
    
    def reversed_corr(spike_1, spike_2, max_delay):
        t1b     = numpy.unique(numpy.floor(spike_1/bin_size))
        t2b     = numpy.unique(numpy.floor(spike_2/bin_size))
        t2b_inv = t2b[-1] + t2b[0] - t2b

        x_cc    = numpy.ones(2*max_delay+1)*(len(t1b) + len(t2b))
        y_cc    = numpy.copy(x_cc)

        for d in xrange(2*max_delay+1):
            t2b_shifted     = t2b + (d - max_delay)
            gsum            = numpy.unique(numpy.concatenate((t1b, t2b_shifted)))
            x_cc[d]        -= len(gsum)
            t2b_inv_shifted = t2b_inv + (d - max_delay)
            gsum            = numpy.unique(numpy.concatenate((t1b, t2b_inv_shifted)))
            y_cc[d]        -= len(gsum)
        return x_cc, y_cc

    def perform_merging(all_pairs, templates, clusters, overlap, result, limits):
        
        for count in xrange(len(all_pairs)):
            pairs      = all_pairs[count]
            key        = 'temp_' + str(pairs[0])
            key2       = 'temp_' + str(pairs[1])
            spikes     = result['spiketimes'][key2]
            amplitudes = result['amplitudes'][key2]
            n1, n2     = len(result['amplitudes'][key2]), len(result['amplitudes'][key])
            result['amplitudes'][key] = numpy.vstack((result['amplitudes'][key].reshape(n2, 2), amplitudes.reshape(n1, 2)))
            result['spiketimes'][key] = numpy.concatenate((result['spiketimes'][key], spikes))
            idx                       = numpy.argsort(result['spiketimes'][key])
            result['spiketimes'][key] = result['spiketimes'][key][idx]
            result['amplitudes'][key] = result['amplitudes'][key][idx]
        
            offset   = 0
            for temp_id in xrange(templates.shape[2]/2):
                if temp_id == pairs[1]:
                    offset += 1
                    key     = 'temp_' + str(temp_id)
                    result['spiketimes'].pop(key)
                    result['amplitudes'].pop(key)
                else:
                    key_before = 'temp_' + str(temp_id)
                    key_after  = 'temp_' + str(temp_id - offset)
                    result['spiketimes'][key_after] = result['spiketimes'].pop(key_before)
                    result['amplitudes'][key_after] = result['amplitudes'].pop(key_before)

            overlap  *= templates.shape[0] * templates.shape[1]
            indices   = [pairs[1], pairs[1] + templates.shape[2]/2]
            templates = numpy.delete(templates, indices, axis=2)
            overlap   = numpy.delete(overlap, indices, axis=0)
            overlap   = numpy.delete(overlap, indices, axis=1)
            overlap  /= templates.shape[0] * templates.shape[1]
            limits    = numpy.delete(limits, pairs[1], axis=0)

            elec      = clusters['electrodes'][pairs[1]]
            nic       = pairs[1] - numpy.where(clusters['electrodes'] == elec)[0][0]
            mask      = clusters['clusters_' + str(elec)] > -1
            tmp       = numpy.unique(clusters['clusters_' + str(elec)][mask])
            elements  = numpy.where(clusters['clusters_' + str(elec)] == tmp[nic])[0]

            clusters['electrodes']            = numpy.delete(clusters['electrodes'], pairs[1])
            clusters['clusters_' + str(elec)] = numpy.delete(clusters['clusters_' + str(elec)], elements) 
            clusters['debug_' + str(elec)]    = numpy.delete(clusters['debug_' + str(elec)], elements, axis=1)  
            clusters['data_' + str(elec)]     = numpy.delete(clusters['data_' + str(elec)], elements, axis=0) 
            clusters['times_' + str(elec)]    = numpy.delete(clusters['times_' + str(elec)], elements)

            all_pairs[all_pairs >= pairs[1]] -= 1
        return templates, clusters, overlap, result, limits

    if comm.rank == 0:
        print "Merging similar templates..."
    
    if comm.rank == 0:
        do_merging = True
        count      = 0
        nb_init    = templates.shape[2]/2
        while do_merging:
            print "Searching for pairs to merge [iteration %d]..." %count
            all_overlaps = []
            all_pairs    = []
            all_corrs    = []
            all_x_cc     = []
            all_y_cc     = []
            spikes       = result['spiketimes']
            nb_before    = templates.shape[2]/2
            all_mergings = numpy.zeros((0, 2), dtype=numpy.int32)
            d_mergings   = numpy.zeros(0, dtype=numpy.int32)
            idx_mergings = []

            gidx         = 0
            for temp_id1 in xrange(nb_before):
                if len(spikes['temp_' + str(temp_id1)]) > 10:
                    for temp_id2 in xrange(temp_id1+1, nb_before):
                        if len(spikes['temp_' + str(temp_id2)]) > 10:
                            if overlap[temp_id1, temp_id2] > cc_overlap:
                                x_cc, y_cc    = reversed_corr(spikes['temp_' + str(temp_id1)], spikes['temp_' + str(temp_id2)], max_delay)
                                all_overlaps += [overlap[temp_id1, temp_id2]]
                                d1            = y_cc
                                d2            = x_cc
                                all_x_cc     += [d2]
                                all_y_cc     += [d1]
                                dd1           = d1[[to_average]]
                                dd2           = d2[to_average]
                                all_corrs    += [(numpy.mean(dd1) - numpy.mean(dd2))/(numpy.mean(dd1) + numpy.mean(dd2) + 1)]
                                all_pairs    += [[temp_id1, temp_id2]]
                                distance      = all_overlaps[-1]*all_corrs[-1]
                                if distance > cc_gap:
                                    if (not (temp_id1 in all_mergings)) and (not (temp_id2 in all_mergings)):
                                        all_mergings  = numpy.vstack((all_mergings, numpy.array([temp_id1, temp_id2])))
                                        d_mergings    = numpy.concatenate((d_mergings, [distance]))
                                        idx_mergings += [gidx]
                                    else:
                                        for i in xrange(len(all_mergings)):
                                            if (temp_id1 in all_mergings[i]) or (temp_id2 in all_mergings[i]):
                                                if distance > d_mergings[i]:
                                                    all_mergings[i] = [temp_id1, temp_id2]
                                                    d_mergings[i]   = distance
                                                    idx_mergings[i] = gidx  
                                gidx += 1

            if len(all_mergings) > 0:
                if make_plots:
                    m = numpy.array(all_overlaps)*numpy.array(all_corrs)
                    pylab.figure()
                    pylab.subplot(121)
                    pylab.plot(m, '.')
                    pylab.plot(m[idx_mergings], 'r.')
                    pylab.xlabel('Pairs')
                    pylab.ylabel('Merging criteria')
                    pylab.subplot(122)
                    pylab.title('Merged CCs')
                    pylab.xlabel('Time [ms]')
                    arg_idx      = numpy.argsort(d_mergings)
                    idx_mergings = numpy.array(idx_mergings)
                    pylab.imshow(numpy.array(all_x_cc)[idx_mergings[arg_idx]], aspect='auto', interpolation='nearest')
                    xmin, xmax = pylab.xlim()
                    ymin, ymax = pylab.ylim()
                    pylab.xticks(numpy.linspace(xmin, xmax, 5), numpy.round(numpy.linspace(-max_delay*cc_bin, max_delay*cc_bin, 5), 1))
                    pylab.plot([-cc_average + max_delay, -cc_average + max_delay], [ymin, ymax], 'r--')
                    pylab.plot([cc_average + max_delay, cc_average + max_delay], [ymin, ymax], 'r--')
                    pylab.xlim(xmin, xmax)
                    pylab.ylim(ymin, ymax)
                    pylab.tight_layout()
                    pylab.savefig(os.path.join(plot_path, 'merging-%d.pdf' %count))

                templates, clusters, overlap, result, limits = perform_merging(all_mergings, templates, clusters, overlap, result, limits)
                print "Automatic merging of %d pairs..." %len(all_mergings)
            else:
                do_merging = False

            count += 1

        if comm.rank == 0:
            print "We merged a total of", nb_init - templates.shape[2]/2, "templates" 

        
        hdf5storage.savemat(file_out_suff + '.amplitudes-merged', result['amplitudes'])
        hdf5storage.savemat(file_out_suff + '.spiketimes-merged', result['spiketimes'])
        hfile = h5py.File(file_out_suff + '.templates-merged.hdf5', 'w')
        cfile = h5py.File(file_out_suff + '.clusters-merged.hdf5', 'w')
        io.write_datasets(hfile, ['templates', 'limits'], {'templates' : templates, 'limits' : limits})
        hfile.close()
        to_write = ['data_', 'clusters_', 'debug_', 'times_']
        for ielec in xrange(N_e):
            io.write_datasets(cfile, to_write, clusters, ielec)
        io.write_datasets(cfile, ['electrodes'], clusters)
        cfile.close()
    
    comm.Barrier()
    io.get_overlaps(comm, params, extension='-merged', parallel_hdf5=parallel_hdf5)
