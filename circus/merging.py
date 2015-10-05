from .shared.utils import *

def main(filename, params, nb_cpu, use_gpu):
    #################################################################
    sampling_rate  = params.getint('data', 'sampling_rate')
    fitting_bin    = 30
    chunk_size     = fitting_bin*params.getint('data', 'chunk_size')
    file_out       = params.get('data', 'file_out')
    file_out_suff  = params.get('data', 'file_out_suff')
    sim_same_elec  = params.getfloat('merging', 'sim_same_elec')
    conf_std       = params.getfloat('merging', 'conf_std')
    conf_mean      = params.getfloat('merging', 'conf_mean')
    refrac_ratio   = params.getfloat('merging', 'refrac_ratio')
    #################################################################

    templates      = io.load_data(params, 'templates')
    clusters       = io.load_data(params, 'clusters')
    result         = io.load_data(params, 'results')
    overlap        = io.get_overlaps(params)
    overlap       /= templates.shape[0] * templates.shape[1]

    merge_similar     = True
    remove_positive   = True
    remove_noise      = True
    remove_mixture    = True
    sort_templates    = False

    io.purge(file_out_suff, '-merged')
    io.purge(file_out_suff, '-final')

    ### Here, we do the obvious merging, if templates are almost the same
    ### and peaking on the same preferred electrodes. This is a rather safe
    ### and conservative merging steps, just removing duplicates
    ### Note that we do several passes, as long as there are merging

    if merge_similar:

        nb_before = templates.shape[2]/2
        if comm.rank == 0:
            print "Merging similar templates..."
        do_merging = True
        while do_merging:
            merged      = []
            old_elec    = -1
            for temp_id in xrange(templates.shape[2]/2):
                if temp_id not in merged:
                    best_elec    = clusters['electrodes'][temp_id]
                    if best_elec != old_elec:
                        elec_padding = 0
                        clust_ids    = numpy.unique(clusters['clusters_' + str(best_elec)])
                        clust_ids    = clust_ids[clust_ids > -1]

                    subset       = numpy.where(best_elec == clusters['electrodes'])[0]
                    if len(subset) > 1:
                        comp         = overlap[subset, temp_id, :]
                        similarities = numpy.max(comp, axis=1)
                        indices      = numpy.argsort(similarities)[::-1]
                        match        = subset[indices[1]]

                        if similarities[indices[1]] > sim_same_elec:
                            #print "Merging", temp_id, 'and', match, "with similarity", similarities[indices[1]]
                            key        = 'temp_' + str(temp_id)
                            key2       = 'temp_' + str(match)
                            spikes     = result['spiketimes'][key2]
                            amplitudes = result['amplitudes'][key2]
                            n1, n2     = len(result['amplitudes'][key2]), len(result['amplitudes'][key])
                            result['amplitudes'][key] = numpy.vstack((result['amplitudes'][key].reshape(n2, 2), amplitudes.reshape(n1, 2)))
                            result['spiketimes'][key] = numpy.concatenate((result['spiketimes'][key], spikes))
                            idx                       = numpy.argsort(result['spiketimes'][key])
                            result['spiketimes'][key] = result['spiketimes'][key][idx]
                            result['amplitudes'][key] = result['amplitudes'][key][idx]

                            merged                   += [match]
                            idx_target                = numpy.where(clust_ids[indices[1]] == clusters['clusters_' + str(best_elec)])[0]
                            clusters['clusters_' + str(best_elec)][idx_target] = clust_ids[elec_padding]
                    old_elec  = best_elec
                elec_padding += 1

            offset = 0
            for temp_id in xrange(templates.shape[2]/2):
                if temp_id in merged:
                    offset += 1
                    key     = 'temp_' + str(temp_id)
                    result['spiketimes'].pop(key)
                    result['amplitudes'].pop(key)
                if temp_id not in merged:
                    key_before = 'temp_' + str(temp_id)
                    key_after  = 'temp_' + str(temp_id - offset)
                    result['spiketimes'][key_after] = result['spiketimes'].pop(key_before)
                    result['amplitudes'][key_after] = result['amplitudes'].pop(key_before)

            overlap  *= templates.shape[0] * templates.shape[1]
            indices   = numpy.concatenate((numpy.array(merged), numpy.array(merged) + templates.shape[2]/2))
            templates = numpy.delete(templates, indices, axis=2)
            overlap   = numpy.delete(overlap, indices, axis=0)
            overlap   = numpy.delete(overlap, indices, axis=1)
            overlap  /= templates.shape[0] * templates.shape[1]
            #Need to do some merging on the clusters
            clusters['electrodes'] = numpy.delete(clusters['electrodes'], merged)
            #clusters['clusters_' + str(best_elec)]
            if len(merged) == 0:
                do_merging = False

        if comm.rank == 0:
            print "We kept", templates.shape[2]/2, "templates out of", nb_before

    ### Here we remove the obvious non-spiking templates, i.e. all those that have
    ### a positive peak instead of a negative peak. They were kept during the fit, but
    ### they are more than likely artefacts, so they can safely be removed
    if remove_positive:
        if comm.rank == 0:
            print "Removing templates with a positive peak (artefacts)..."
        nb_before = templates.shape[2]/2
        artefacts = []
        old_elec  = -1
        for temp_id in xrange(nb_before):
            positive_peak = numpy.max(templates[:,:,temp_id]) == numpy.max(abs(templates[:,:,temp_id]))
            best_elec     = clusters['electrodes'][temp_id]
            if best_elec != old_elec:
                elec_padding = 0
                clust_ids    = numpy.unique(clusters['clusters_' + str(best_elec)])
                clust_ids    = clust_ids[clust_ids > -1]

            if positive_peak:
                #print "Template", temp_id, 'is removed as a positive artefact'
                artefacts += [temp_id]
                idx_target = numpy.where(clust_ids[elec_padding] == clusters['clusters_' + str(best_elec)])[0]
                clusters['clusters_' + str(best_elec)][idx_target] = -1

            old_elec      = best_elec
            elec_padding += 1

        offset = 0
        for temp_id in xrange(nb_before):
            if temp_id in artefacts:
                offset += 1
                key     = 'temp_' + str(temp_id)
                result['spiketimes'].pop(key)
                result['amplitudes'].pop(key)
            if temp_id not in artefacts:
                key_before = 'temp_' + str(temp_id)
                key_after  = 'temp_' + str(temp_id - offset)
                result['spiketimes'][key_after] = result['spiketimes'].pop(key_before)
                result['amplitudes'][key_after] = result['amplitudes'].pop(key_before)

        indices   = numpy.concatenate((numpy.array(artefacts), numpy.array(artefacts) + nb_before))
        templates = numpy.delete(templates, indices, axis=2)
        clusters['electrodes'] = numpy.delete(clusters['electrodes'], artefacts)
        if comm.rank == 0:
            print "We kept", templates.shape[2]/2, "templates out of", nb_before

    ### Here we clear the remaining templates, by removing the noise, and keeping
    ### only the good values of the fits, i.e. those that should be around 1
    ### Note that we process data per chunks, because those distribution can
    ### change over time...

    if remove_noise:

        nb_before = templates.shape[2]/2
        if comm.rank == 0:
            print "Denoising the templates..."
            pbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()], maxval=nb_before).start()

        max_time = 0
        for temp_id in xrange(nb_before):
            key        = 'temp_' + str(temp_id)
            times      = result['spiketimes'][key]
            if len(times) > 0:
                if times[-1] > max_time:
                    max_time = times[-1]

        for temp_id in xrange(nb_before):
            key        = 'temp_' + str(temp_id)
            times      = result['spiketimes'][key]
            if len(times) > 0:
                all_params = numpy.zeros((0, 7))
                amplitudes = result['amplitudes'][key][:, 0]
                chunks     = numpy.arange(0, times[-1], chunk_size)
                all_idx    = numpy.zeros(0, dtype=numpy.int32)
                for gidx in xrange(len(chunks)):
                    indices    = numpy.where((times >= chunks[gidx]) & (times < chunks[gidx] + chunk_size))[0]
                    subset     = amplitudes[indices]
                    if len(subset) > 1000:
                        y, x       = numpy.histogram(subset, 500)
                        parameters = fit_noise(x[1:], y/float(sum(y)), conf_mean)
                        if parameters is not None:
                            all_params = numpy.vstack((all_params, parameters))
                        else:
                            all_params = numpy.vstack((all_params, numpy.zeros(7)))
                    else:
                        all_params = numpy.vstack((all_params, numpy.zeros(7)))

                #pylab.plot(result['spiketimes'][key], result['amplitudes'][key][:, 0], 'k.')
                smooth_size = chunk_size/fitting_bin
                more_chunks = numpy.arange(0, max_time, smooth_size)
                smoothing   = numpy.minimum(5, 2*int(len(more_chunks)/4) + 1)
                padding     = smoothing/2

                # If the fit fails, we do not denoise there
                idx_good    = numpy.where(all_params[:, 1] > 0)[0]
                idx_bad     = numpy.where(all_params[:, 1] == 0)[0]
                if len(idx_good) > 0:
                    all_params[idx_bad, 1] = numpy.mean(all_params[idx_good, 1])
                    all_params[idx_bad, 2] = numpy.mean(all_params[idx_good, 2])
                else:
                    all_params[idx_bad, 1] = 1
                    all_params[idx_bad, 2] = 1
                m   = algo.smooth(numpy.interp(more_chunks, chunks, all_params[:, 1]), smoothing)[padding:-padding]
                s   = algo.smooth(numpy.interp(more_chunks, chunks, all_params[:, 2]), smoothing)[padding:-padding]
                threshold      = conf_std*s
                for gidx in xrange(len(more_chunks)):
                    indices    = numpy.where((times >= more_chunks[gidx]) & (times < more_chunks[gidx] + smooth_size))[0]
                    subset     = amplitudes[indices]
                    idx        = numpy.where((numpy.abs(subset - m[gidx]) <= numpy.mean(threshold)) | (subset >= m[gidx]))[0]
                    all_idx    = numpy.concatenate((all_idx, indices[idx]))

                #print "%g percent removed as noise for template" %(100*(1-float(len(all_idx))/len(times))), temp_id
                result['amplitudes'][key] = result['amplitudes'][key][all_idx]
                result['spiketimes'][key] = result['spiketimes'][key][all_idx]

            if comm.rank == 0:
                pbar.update(temp_id)

        if comm.rank == 0:
            pbar.finish()

    if remove_mixture:

        if comm.rank == 0:
            print "Removing templates with too many refractory violations (mixtures)..."

        nb_before = templates.shape[2]/2
        artefacts = []
        old_elec  = -1
        for temp_id in xrange(nb_before):
            key        = 'temp_' + str(temp_id)
            best_elec  = clusters['electrodes'][temp_id]
            times      = result['spiketimes'][key]
            if best_elec != old_elec:
                elec_padding = 0
                clust_ids    = numpy.unique(clusters['clusters_' + str(best_elec)])
                clust_ids    = clust_ids[clust_ids > -1]

            if len(times) > 0:
                isis = numpy.diff(times)
                idx  = numpy.where(isis < 2e-3*sampling_rate)[0]
                if (len(idx)/float(len(times))) > refrac_ratio:
                    artefacts += [temp_id]
                    idx_target = numpy.where(clust_ids[elec_padding] == clusters['clusters_' + str(best_elec)])[0]
                    clusters['clusters_' + str(best_elec)][idx_target] = -1

            old_elec      = best_elec
            elec_padding += 1

        offset = 0
        for temp_id in xrange(nb_before):
            if temp_id in artefacts:
                offset += 1
                key     = 'temp_' + str(temp_id)
                result['spiketimes'].pop(key)
                result['amplitudes'].pop(key)
            if temp_id not in artefacts:
                key_before = 'temp_' + str(temp_id)
                key_after  = 'temp_' + str(temp_id - offset)
                result['spiketimes'][key_after] = result['spiketimes'].pop(key_before)
                result['amplitudes'][key_after] = result['amplitudes'].pop(key_before)

        indices   = numpy.concatenate((numpy.array(artefacts), numpy.array(artefacts) + nb_before))
        templates = numpy.delete(templates, indices, axis=2)
        clusters['electrodes'] = numpy.delete(clusters['electrodes'], artefacts)
        if comm.rank == 0:
            print "We kept", templates.shape[2]/2, "templates out of", nb_before


    hdf5storage.savemat(file_out_suff + '.amplitudes-merged', result['amplitudes'])
    hdf5storage.savemat(file_out_suff + '.spiketimes-merged', result['spiketimes'])
    hdf5storage.savemat(file_out_suff + '.templates-merged', {'templates' : templates})
    hdf5storage.savemat(file_out_suff + '.clusters-merged', clusters)
    io.get_overlaps(params, extension='-merged')

    if sort_templates:
        print "Sorting cells..."
        sort_cells(params, extension='-merged')
        io.purge(file_out_suff, '-merged')
