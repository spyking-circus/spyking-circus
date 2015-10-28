from .shared.utils import *
import pylab

def main(filename, params, nb_cpu, nb_gpu, use_gpu):

    #################################################################
    sampling_rate  = params.getint('data', 'sampling_rate')
    N_e            = params.getint('data', 'N_e')
    N_t            = params.getint('data', 'N_t')
    N_total        = params.getint('data', 'N_total')
    file_out_suff  = params.get('data', 'file_out_suff')
    file_out       = params.get('data', 'file_out')
    cc_gap         = params.getfloat('merging', 'cc_gap')
    cc_overlap     = params.getfloat('merging', 'cc_overlap')
    
    bin_size       = int(2e-3 * sampling_rate)
    max_delay      = 100

    templates      = io.load_data(params, 'templates')
    clusters       = io.load_data(params, 'clusters')
    result         = io.load_data(params, 'results')
    overlap        = hdf5storage.loadmat(file_out_suff + '.overlap.mat')['maxoverlap']
    overlap       /= templates.shape[0] * templates.shape[1]

    io.purge(file_out_suff, '-merged')

    delay_average  = 20
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

    def perform_merging(all_pairs, templates, clusters, overlap, result):
        
        for count in xrange(len(all_pairs)):
            pairs      = all_pairs[count]
            print "Automatic merging of templates", pairs
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
        return templates, clusters, overlap, result

    if comm.rank == 0:
        print "Merging similar templates..."
    do_merging = True
    nb_init    = templates.shape[2]/2
    while do_merging:

        all_overlaps = []
        all_pairs    = []
        all_corrs    = []
        spikes       = result['spiketimes']
        nb_before    = templates.shape[2]/2
        all_mergings = numpy.zeros((0, 2), dtype=numpy.int32)

        for temp_id1 in xrange(nb_before):
            if len(spikes['temp_' + str(temp_id1)]) > 0:
                for temp_id2 in xrange(temp_id1+1, nb_before):
                    if len(spikes['temp_' + str(temp_id2)]) > 0:
                        if overlap[temp_id1, temp_id2] > cc_overlap:
                            x_cc, y_cc    = reversed_corr(spikes['temp_' + str(temp_id1)], spikes['temp_' + str(temp_id2)], max_delay)
                            all_overlaps += [overlap[temp_id1, temp_id2]]
                            d1            = numpy.mean(y_cc[to_average])
                            d2            = numpy.mean(x_cc[to_average])
                            all_corrs    += [(d1 - d2)/(d1 + d2 + 1)]
                            all_pairs    += [[temp_id1, temp_id2]]

                            distances     = (1 - all_overlaps[-1])*all_corrs[-1]
                            if distances > cc_gap:
                                if (not (temp_id1 in all_mergings)) and (not (temp_id2 in all_mergings)):
                                    all_mergings = numpy.vstack((all_mergings,  numpy.array([temp_id1, temp_id2])))

        if len(all_mergings) > 0:
            templates, clusters, overlap, result = perform_merging(all_mergings, templates, clusters, overlap, result)
        else:
            do_merging = False

        #distances  = (1 - numpy.array(all_overlaps)) * numpy.array(all_corrs)
        #d          = numpy.argmax(distances)
        #if distances[d] > 0.01:
        #    pairs = all_pairs[d]
        #    templates, clusters, overlap, result = perform_merging(pairs, templates, clusters, overlap, result)
        #else:
        #    do_merging = False

    if comm.rank == 0:
        print "We merged", nb_init - templates.shape[2]/2, "templates" 

    if templates.shape[2]/2 < nb_init:
        hdf5storage.savemat(file_out_suff + '.amplitudes-merged', result['amplitudes'])
        hdf5storage.savemat(file_out_suff + '.spiketimes-merged', result['spiketimes'])
        hdf5storage.savemat(file_out_suff + '.templates-merged', {'templates' : templates})
        hdf5storage.savemat(file_out_suff + '.clusters-merged', clusters)
        io.get_overlaps(params, extension='-merged')
