import numpy, scipy
import pylab
import os

from circus.shared.files import load_parameters, load_data, load_chunk, get_results, get_nodes_and_edges, get_results, read_probe
import numpy, pylab
from circus.shared import algorithms as algo

def view_fit(file_name, t_start=0, t_stop=1, n_elec=2, fit_on=True, square=True, templates=None, save=False):
    
    params          = load_parameters(file_name)
    N_e             = params.getint('data', 'N_e')
    N_t             = params.getint('data', 'N_t')
    N_total         = params.getint('data', 'N_total')
    sampling_rate   = params.getint('data', 'sampling_rate')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    spike_thresh     = params.getfloat('data', 'spike_thresh')
    template_shift   = params.getint('data', 'template_shift')
    nodes, edges     = get_nodes_and_edges(params)
    chunk_size       = (t_stop - t_start)*sampling_rate
    padding          = (t_start*sampling_rate*N_total, t_start*sampling_rate*N_total)
    suff             = params.get('data', 'suffix')

    if do_spatial_whitening:
        spatial_whitening  = load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = load_data(params, 'temporal_whitening')

    thresholds       = load_data(params, 'thresholds')
    data, data_shape = load_chunk(params, 0, chunk_size*N_total, padding=padding, chunk_size=chunk_size, nodes=nodes)
    
    if do_spatial_whitening:
        data = numpy.dot(data, spatial_whitening)
    if do_temporal_whitening:
        data = scipy.ndimage.filters.convolve1d(data, temporal_whitening, axis=0, mode='constant')

    try:
        result    = load_data(params, 'results')
    except Exception:
        result    = {'spiketimes' : {}, 'amplitudes' : {}}
    if fit_on:
        curve     = numpy.zeros((N_e, (t_stop-t_start)*sampling_rate), dtype=numpy.float32)
        count     = 0
        limit     = (t_stop-t_start)*sampling_rate-template_shift+1
        if templates is None:
            try:
                templates = load_data(params, 'templates')
            except Exception:
                templates = numpy.zeros((0, 0, 0))
        for key in result['spiketimes'].keys():
            elec  = int(key.split('_')[1])
            lims  = (t_start*sampling_rate + template_shift, t_stop*sampling_rate - template_shift-1)
            idx   = numpy.where((result['spiketimes'][key] > lims[0]) & (result['spiketimes'][key] < lims[1]))
            for spike, (amp1, amp2) in zip(result['spiketimes'][key][idx], result['amplitudes'][key][idx]):
                count += 1
                spike -= t_start*sampling_rate
                tmp1   = templates[:, elec].toarray().reshape(N_e, N_t)
                tmp2   = templates[:, elec+templates.shape[1]/2].toarray().reshape(N_e, N_t)
                
                curve[:, spike-template_shift:spike+template_shift+1] += amp1*tmp1 + amp2*tmp2
        print "Number of spikes", count

    if not numpy.iterable(n_elec):
        if square:
            idx = numpy.random.permutation(numpy.arange(N_e))[:n_elec**2]
        else:
            idx = numpy.random.permutation(numpy.arange(N_e))[:n_elec]
    else:
        idx    = n_elec
        n_elec = numpy.sqrt(len(idx))
    pylab.figure()
    for count, i in enumerate(idx):
        if square:
            pylab.subplot(n_elec, n_elec, count + 1)
            if (numpy.mod(count, n_elec) != 0):
                pylab.setp(pylab.gca(), yticks=[])
            else:
                pylab.ylabel('Signal')
            if (count < n_elec*(n_elec - 1)):
                pylab.setp(pylab.gca(), xticks=[])
            else:
                pylab.xlabel('Time [ms]')
        else:
            pylab.subplot(n_elec, 1, count + 1)
            if count != (n_elec - 1):
                pylab.setp(pylab.gca(), xticks=[])
            else:
                pylab.xlabel('Time [ms]')
                
        pylab.plot(data[:, i], '0.25')
        if fit_on:
            pylab.plot(curve[i], 'r')
        xmin, xmax = pylab.xlim()
        pylab.plot([xmin, xmax], [-thresholds[i], -thresholds[i]], 'k--')
        pylab.plot([xmin, xmax], [thresholds[i], thresholds[i]], 'k--')
        pylab.title('Electrode %d' %i)
        if (square and not (count < n_elec*(n_elec - 1))) or (not square and not count != (n_elec - 1)):
            x, y = pylab.xticks()
            pylab.xticks(x, numpy.round(x/sampling_rate, 2))

        pylab.ylim(-2*thresholds[i], 2*thresholds[i])
    pylab.tight_layout()
    if save:
        pylab.savefig(os.path.join(save[0], save[1] + suff +'.pdf'))
        pylab.close()
    else:
        pylab.show()


def view_clusters(data, rho, delta, centers, halo, injected=None, dc=None, save=False):

    import mdp
    fig = pylab.figure(figsize=(15, 10))
    ax  = fig.add_subplot(231)
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$\delta$')
    ax.set_title(r'$d_c = %g$' %dc)
    ax.plot(rho, delta, 'o', color='black')
    ax.set_yscale('log')

    import matplotlib.colors as colors
    my_cmap   = pylab.get_cmap('jet')
    cNorm     = colors.Normalize(vmin=numpy.min(halo), vmax=numpy.max(halo))
    scalarMap = pylab.cm.ScalarMappable(norm=cNorm, cmap=my_cmap)

    for i in centers:
        if halo[i] > -1:
            colorVal = scalarMap.to_rgba(halo[i])
            ax.plot(rho[i], delta[i], 'o', color=colorVal)

    pca = mdp.nodes.PCANode(output_dim=3)
    visu_data = pca(data.astype(numpy.double))
    assigned  = numpy.where(halo > -1)[0]

    try:
        ax = fig.add_subplot(232)
        ax.scatter(visu_data[assigned,0], visu_data[assigned,1], c=halo[assigned], cmap=my_cmap, linewidth=0)
        ax.set_xlabel('Dim 0')
        ax.set_ylabel('Dim 1')

        ax = fig.add_subplot(233)
        ax.scatter(visu_data[assigned,0], visu_data[assigned,2], c=halo[assigned], cmap=my_cmap, linewidth=0)
        ax.set_xlabel('Dim 0')
        ax.set_ylabel('Dim 2')
                
        ax = fig.add_subplot(235)
        ax.scatter(visu_data[assigned,1], visu_data[assigned,2], c=halo[assigned], cmap=my_cmap, linewidth=0)
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
    except Exception:
        pass

    try:

        import matplotlib.colors as colors
        my_cmap   = pylab.get_cmap('winter')

        ax = fig.add_subplot(236)
        idx = numpy.argsort(rho)
        ax.scatter(visu_data[idx,0], visu_data[idx,1], c=rho[idx], cmap=my_cmap)
        ax.scatter(visu_data[centers, 0], visu_data[centers, 1], c='r')
        if injected is not None:
            ax.scatter(visu_data[injected, 0], visu_data[injected, 1], c='b')
        ax.set_xlabel('Dim 0')
        ax.set_ylabel('Dim 1')
    except Exception:
        pass

    ax = fig.add_subplot(234)
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$\delta$')
    ax.set_title('Putative Cluster Centers')
    ax.plot(rho, delta, 'o', color='black')
    ax.plot(rho[centers], delta[centers], 'o', color='r')
    ax.set_yscale('log')
    pylab.tight_layout()
    if save:
        pylab.savefig(os.path.join(save[0], 'cluster_%s.pdf' %save[1]))
        pylab.close()
    else:
        pylab.show()
    del fig


def view_waveforms_clusters(data, halo, threshold, templates, amps_lim, n_curves=200, save=False):
    
    nb_templates = templates.shape[1]
    n_panels     = numpy.ceil(numpy.sqrt(nb_templates))
    mask         = numpy.where(halo > -1)[0]
    clust_idx    = numpy.unique(halo[mask])
    fig          = pylab.figure()    
    square       = True
    center       = len(data[0] - 1)/2
    for count, i in enumerate(xrange(nb_templates)):
        if square:
            pylab.subplot(n_panels, n_panels, count + 1)
            if (numpy.mod(count, n_panels) != 0):
                pylab.setp(pylab.gca(), yticks=[])
            if (count < n_panels*(n_panels - 1)):
                pylab.setp(pylab.gca(), xticks=[])
        
        subcurves = numpy.where(halo == clust_idx[count])[0]
        for k in numpy.random.permutation(subcurves)[:n_curves]:
            pylab.plot(data[k], '0.5')
        
        pylab.plot(templates[:, count], 'r')
        
##### TODO: remove debug zone
        # print("# Print `amps_lim` size")
        # print(numpy.size(amps_lim))
        # print("# Print `count`")
        # print(count)
        # print("# Print `amps_lim[count]` size")
        # try:
        #     print(numpy.size(amps_lim[count]))
        # except:
        #     print("Error (index is out of bounds)")
##### end debug zone
        
        pylab.plot(amps_lim[count][0]*templates[:, count], 'b', alpha=0.5)
        pylab.plot(amps_lim[count][1]*templates[:, count], 'b', alpha=0.5)
        
        xmin, xmax = pylab.xlim()
        pylab.plot([xmin, xmax], [-threshold, -threshold], 'k--')
        pylab.plot([xmin, xmax], [threshold, threshold], 'k--')
        #pylab.ylim(-1.5*threshold, 1.5*threshold)
        ymin, ymax = pylab.ylim()
        pylab.plot([center, center], [ymin, ymax], 'k--')
        pylab.title('Cluster %d' %i)

    if nb_templates > 0:
        pylab.tight_layout()
    if save:
        pylab.savefig(os.path.join(save[0], 'waveforms_%s.pdf' %save[1]))
        pylab.close()
    else:
        pylab.show()
    del fig




def view_waveforms(file_name, temp_id, n_spikes=2000):
    
    params          = load_parameters(file_name)
    N_e             = params.getint('data', 'N_e')
    N_total         = params.getint('data', 'N_total')
    sampling_rate   = params.getint('data', 'sampling_rate')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    spike_thresh     = params.getfloat('data', 'spike_thresh')
    file_out_suff    = params.get('data', 'file_out_suff')
    N_t              = params.getint('data', 'N_t')
    nodes, edges     = get_nodes_and_edges(params)
    chunk_size       = N_t
    
    if do_spatial_whitening:
        spatial_whitening  = load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = load_data(params, 'temporal_whitening')

    try:
        result    = load_data(params, 'results')
    except Exception:
        result    = {'spiketimes' : {}, 'amplitudes' : {}}
    spikes        = result['spiketimes']['temp_'+str(temp_id)]
    thresholds    = load_data(params, 'thresholds')    
    
    curve     = numpy.zeros((n_spikes, N_e, N_t), dtype=numpy.float32)
    count     = 0
    try:
        templates = load_data(params, 'templates')
    except Exception:
        templates = numpy.zeros((0, 0, 0))
    
    for count, t_spike in enumerate(numpy.random.permutation(spikes)[:n_spikes]):
        padding          = ((t_spike - int(N_t-1)/2)*N_total, (t_spike - int(N_t-1)/2)*N_total)
        data, data_shape = load_chunk(params, 0, chunk_size*N_total, padding=padding, chunk_size=chunk_size, nodes=nodes)
        if do_spatial_whitening:
            data = numpy.dot(data, spatial_whitening)
        if do_temporal_whitening:
            data = scipy.ndimage.filters.convolve1d(data, temporal_whitening, axis=0, mode='constant')
        
        curve[count] = data.T
    pylab.subplot(121)
    pylab.imshow(numpy.mean(curve, 0), aspect='auto')
    pylab.subplot(122)
    pylab.imshow(templates[:,:,temp_id], aspect='auto')
    pylab.show()    
    return curve

def view_isolated_waveforms(file_name, t_start=0, t_stop=1):
    
    params          = load_parameters(file_name)
    N_e             = params.getint('data', 'N_e')
    N_total         = params.getint('data', 'N_total')
    sampling_rate   = params.getint('data', 'sampling_rate')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    spike_thresh     = params.getfloat('data', 'spike_thresh')
    file_out_suff    = params.get('data', 'file_out_suff')
    N_t              = params.getint('data', 'N_t')
    nodes, edges     = get_nodes_and_edges(params)
    chunk_size       = (t_stop - t_start)*sampling_rate
    padding          = (t_start*sampling_rate*N_total, t_start*sampling_rate*N_total)

    if do_spatial_whitening:
        spatial_whitening  = load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = load_data(params, 'temporal_whitening')

    thresholds       = load_data(params, 'thresholds')
    data, data_shape = load_chunk(params, 0, chunk_size*N_total, padding=padding, chunk_size=chunk_size, nodes=nodes)
       
    peaks      = {}
    n_spikes   = 0

    if do_spatial_whitening:
        data = numpy.dot(data, spatial_whitening)
    if do_temporal_whitening: 
        for i in xrange(N_e):
            data[:, i] = numpy.convolve(data[:, i], temporal_whitening, 'same')
            peaks[i]   = algo.detect_peaks(data[:, i], thresholds[i], valley=True, mpd=0)
            n_spikes  += len(peaks[i])

    curve = numpy.zeros((n_spikes, N_t-1), dtype=numpy.float32)
    print "We found", n_spikes, "spikes"
    
    count = 0
    for electrode in xrange(N_e):
        for i in xrange(len(peaks[electrode])):
            peak_time = peaks[electrode][i]
            if (peak_time > N_t/2):
                curve[count] = data[peak_time - N_t/2:peak_time + N_t/2, electrode]
            count += 1

    pylab.subplot(111)
    pylab.imshow(curve, aspect='auto')
    pylab.show()    
    return curve



def view_triggers(file_name, triggers, n_elec=2, square=True, xzoom=None, yzoom=None, n_curves=100):
    
    params          = load_parameters(file_name)
    N_e             = params.getint('data', 'N_e')
    N_total         = params.getint('data', 'N_total')
    sampling_rate   = params.getint('data', 'sampling_rate')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    spike_thresh     = params.getfloat('data', 'spike_thresh')
    file_out_suff    = params.get('data', 'file_out_suff')
    N_t              = params.getint('data', 'N_t')
    nodes, edges     = get_nodes_and_edges(params)
    chunk_size       = N_t
    
    if do_spatial_whitening:
        spatial_whitening  = load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = load_data(params, 'temporal_whitening')
   
    thresholds = load_data(params, 'thresholds')    
    
    curve      = numpy.zeros((len(triggers), N_e, N_t), dtype=numpy.float32)
    count      = 0
    
    for count, t_spike in enumerate(triggers):
        padding          = ((t_spike - N_t/2)*N_total, (t_spike - N_t/2)*N_total)
        data, data_shape = load_chunk(params, 0, N_t*N_total, padding=padding, chunk_size=chunk_size, nodes=nodes)
        if do_spatial_whitening:
            data = numpy.dot(data, spatial_whitening)
        if do_temporal_whitening:
            data = scipy.ndimage.filters.convolve1d(data, temporal_whitening, axis=0, mode='constant')
        
        curve[count] = data.T
    pylab.subplot(111)
    pylab.imshow(numpy.mean(curve, 0), aspect='auto') 

    if not numpy.iterable(n_elec):
        if square:
            idx = numpy.random.permutation(numpy.arange(N_e))[:n_elec**2]
        else:
            idx = numpy.random.permutation(numpy.arange(N_e))[:n_elec]
    else:
        idx    = n_elec
        n_elec = numpy.sqrt(len(idx))
    pylab.figure()
    for count, i in enumerate(idx):
        if square:
            pylab.subplot(n_elec, n_elec, count + 1)
            if (numpy.mod(count, n_elec) != 0):
                pylab.setp(pylab.gca(), yticks=[])
            if (count < n_elec*(n_elec - 1)):
                pylab.setp(pylab.gca(), xticks=[])
        else:
            pylab.subplot(n_elec, 1, count + 1)
            if count != (n_elec - 1):
                pylab.setp(pylab.gca(), xticks=[])
        for k in numpy.random.permutation(numpy.arange(len(curve)))[:n_curves]:
            pylab.plot(curve[k, i, :], '0.25')
        pylab.plot(numpy.mean(curve, 0)[i], 'r')
        xmin, xmax = pylab.xlim()
        pylab.plot([xmin, xmax], [-thresholds[i], -thresholds[i]], 'k--')
        pylab.plot([xmin, xmax], [thresholds[i], thresholds[i]], 'k--')
        pylab.title('Electrode %d' %i)
        if xzoom:
            pylab.xlim(xzoom[0], xzoom[1])
        pylab.ylim(-2*thresholds[i], 2*thresholds[i])
        if yzoom:
            pylab.ylim(yzoom[0], yzoom[1])
    pylab.tight_layout()
    pylab.show()
    return curve


def view_performance(file_name, triggers, lims=(150,150)):
    
    params          = load_parameters(file_name)
    N_e             = params.getint('data', 'N_e')
    N_total         = params.getint('data', 'N_total')
    sampling_rate   = params.getint('data', 'sampling_rate')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    spike_thresh     = params.getfloat('data', 'spike_thresh')
    file_out_suff    = params.get('data', 'file_out_suff')
    N_t              = params.getint('data', 'N_t')
    nodes, edges     = get_nodes_and_edges(params)
    chunk_size       = N_t
    
    if do_spatial_whitening:
        spatial_whitening  = load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = load_data(params, 'temporal_whitening')

    thresholds       = load_data(params, 'thresholds')    
    
    try:
        result    = load_data(params, 'results')
    except Exception:
        result    = {'spiketimes' : {}, 'amplitudes' : {}}

    curve     = numpy.zeros((len(triggers), len(result['spiketimes'].keys()), lims[1]+lims[0]), dtype=numpy.int32)
    count     = 0
    
    for count, t_spike in enumerate(triggers):
        for key in result['spiketimes'].keys():
            elec  = int(key.split('_')[1])
            idx   = numpy.where((result['spiketimes'][key] > t_spike - lims[0]) & (result['spiketimes'][key] <  t_spike + lims[0]))
            curve[count, elec, t_spike - result['spiketimes'][key][idx]] += 1
    pylab.subplot(111)
    pylab.imshow(numpy.mean(curve, 0), aspect='auto') 
    return curve


def view_templates(file_name, temp_id=0, best_elec=None, templates=None):

    params          = load_parameters(file_name)
    N_e             = params.getint('data', 'N_e')
    N_total         = params.getint('data', 'N_total')
    sampling_rate   = params.getint('data', 'sampling_rate')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    spike_thresh     = params.getfloat('data', 'spike_thresh')
    file_out_suff    = params.get('data', 'file_out_suff')
    N_t              = params.getint('data', 'N_t')
    nodes, edges     = get_nodes_and_edges(params)
    chunk_size       = N_t
    N_total          = params.getint('data', 'N_total')
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)

    if templates is None:
        templates    = load_data(params, 'templates')
    clusters         = load_data(params, 'clusters')
    probe            = read_probe(params)

    positions = {}
    for i in probe['channel_groups'].keys():
        positions.update(probe['channel_groups'][i]['geometry'])
    xmin = 0
    xmax = 0
    ymin = 0
    ymax = 0
    scaling = 10*numpy.max(numpy.abs(templates[:,:,temp_id]))
    for i in xrange(N_e):
        if positions[i][0] < xmin:
            xmin = positions[i][0]
        if positions[i][0] > xmax:
            xmax = positions[i][0]
        if positions[i][1] < ymin:
            ymin = positions[i][0]
        if positions[i][1] > ymax:
            ymax = positions[i][1]
    if best_elec is None:
        best_elec = clusters['electrodes'][temp_id]
    elif best_elec == 'auto':
        best_elec = numpy.argmin(numpy.min(templates[:, :, temp_id], 1))
    pylab.figure()
    for count, i in enumerate(xrange(N_e)):
        x, y     = positions[i]
        xpadding = ((x - xmin)/(float(xmax - xmin) + 1))*(2*N_t)
        ypadding = ((y - ymin)/(float(ymax - ymin) + 1))*scaling

        if i == best_elec:
            c='r'
        elif i in inv_nodes[edges[nodes[best_elec]]]:
            c='k'
        else: 
            c='0.5'
        pylab.plot(xpadding + numpy.arange(0, N_t), ypadding + templates[i, :, temp_id], color=c)
    pylab.tight_layout()
    pylab.setp(pylab.gca(), xticks=[], yticks=[])
    pylab.xlim(xmin, 3*N_t)
    pylab.show()    
    return best_elec

def view_raw_templates(file_name, n_temp=2, square=True):

    N_e, N_t, N_tm = templates.shape
    if not numpy.iterable(n_temp):
        if square:
            idx = numpy.random.permutation(numpy.arange(N_tm/2))[:n_temp**2]
        else:
            idx = numpy.random.permutation(numpy.arange(N_tm/2))[:n_temp]
    else:
        idx = n_temp

    import matplotlib.colors as colors
    my_cmap   = pylab.get_cmap('winter')
    cNorm     = colors.Normalize(vmin=0, vmax=N_e)
    scalarMap = pylab.cm.ScalarMappable(norm=cNorm, cmap=my_cmap)

    pylab.figure()
    for count, i in enumerate(idx):
        if square:
            pylab.subplot(n_temp, n_temp, count + 1)
            if (numpy.mod(count, n_temp) != 0):
                pylab.setp(pylab.gca(), yticks=[])
            if (count < n_temp*(n_temp - 1)):
                pylab.setp(pylab.gca(), xticks=[])
        else:
            pylab.subplot(len(idx), 1, count + 1)
            if count != (len(idx) - 1):
                pylab.setp(pylab.gca(), xticks=[])
        for j in xrange(N_e):
            colorVal = scalarMap.to_rgba(j)
            pylab.plot(templates[j, :, i], color=colorVal)

        pylab.title('Template %d' %i)
    pylab.tight_layout()
    pylab.show()    

def view_whitening(data):
    pylab.subplot(121)
    pylab.imshow(data['spatial'], interpolation='nearest')
    pylab.title('Spatial')
    pylab.xlabel('# Electrode')
    pylab.ylabel('# Electrode')
    pylab.colorbar()
    pylab.subplot(122)
    pylab.title('Temporal')
    pylab.plot(data['temporal'])
    pylab.xlabel('Time [ms]')
    x, y = pylab.xticks()
    pylab.xticks(x, (x-x[-1]/2)/10)
    pylab.tight_layout()


def view_masks(file_name, t_start=0, t_stop=1, n_elec=0):

    params          = load_parameters(file_name)
    N_e             = params.getint('data', 'N_e')
    N_total         = params.getint('data', 'N_total')
    sampling_rate   = params.getint('data', 'sampling_rate')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    spike_thresh     = params.getfloat('data', 'spike_thresh')
    file_out_suff    = params.get('data', 'file_out_suff')
    N_t              = params.getint('data', 'N_t')
    nodes, edges     = get_nodes_and_edges(params)
    chunk_size       = (t_stop - t_start)*sampling_rate
    padding          = (t_start*sampling_rate*N_total, t_start*sampling_rate*N_total)
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)
    safety_time      = int(params.getfloat('clustering', 'safety_time')*sampling_rate*1e-3)

    if do_spatial_whitening:
        spatial_whitening  = load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = load_data(params, 'temporal_whitening')

    thresholds       = load_data(params, 'thresholds')
    data, data_shape = load_chunk(params, 0, chunk_size*N_total, padding=padding, chunk_size=chunk_size, nodes=nodes)
    
    peaks            = {}
    indices          = inv_nodes[edges[nodes[n_elec]]]
    
    if do_spatial_whitening:
        data = numpy.dot(data, spatial_whitening)
    if do_temporal_whitening: 
        data = scipy.ndimage.filters.convolve1d(data, temporal_whitening, axis=0, mode='constant')
    
    for i in xrange(N_e):
        peaks[i]   = algo.detect_peaks(data[:, i], thresholds[i], valley=True, mpd=0)


    pylab.figure()

    for count, i in enumerate(indices):
        
        pylab.plot(count*5 + data[:, i], '0.25')
        #xmin, xmax = pylab.xlim()
        pylab.scatter(peaks[i], count*5 + data[peaks[i], i], s=10, c='r')

    for count, i in enumerate(peaks[n_elec]):
        pylab.axvspan(i - safety_time, i + safety_time, facecolor='r', alpha=0.5)

    pylab.ylim(-5, len(indices)*5 )
    pylab.xlabel('Time [ms]')
    pylab.ylabel('Electrode')
    pylab.tight_layout()
    pylab.setp(pylab.gca(), yticks=[])
    pylab.show()
    return peaks

def view_peaks(file_name, t_start=0, t_stop=1, n_elec=2, square=True, xzoom=None, yzoom=None):
    params          = load_parameters(file_name)
    N_e             = params.getint('data', 'N_e')
    N_total         = params.getint('data', 'N_total')
    sampling_rate   = params.getint('data', 'sampling_rate')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    spike_thresh     = params.getfloat('data', 'spike_thresh')
    file_out_suff    = params.get('data', 'file_out_suff')
    N_t              = params.getint('data', 'N_t')
    nodes, edges     = get_nodes_and_edges(params)
    chunk_size       = (t_stop - t_start)*sampling_rate
    padding          = (t_start*sampling_rate*N_total, t_start*sampling_rate*N_total)

    if do_spatial_whitening:
        spatial_whitening  = load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = load_data(params, 'temporal_whitening')

    thresholds       = load_data(params, 'thresholds')
    data, data_shape = load_chunk(params, 0, chunk_size*N_total, padding=padding, chunk_size=chunk_size, nodes=nodes)
       
    peaks      = {}
    
    if do_spatial_whitening:
        data = numpy.dot(data, spatial_whitening)
    if do_temporal_whitening: 
        data = scipy.ndimage.filters.convolve1d(data, temporal_whitening, axis=0, mode='constant')
    
    for i in xrange(N_e):
        peaks[i]   = algo.detect_peaks(data[:, i], thresholds[i], valley=True, mpd=0)

    if not numpy.iterable(n_elec):
        if square:
            idx = numpy.random.permutation(numpy.arange(N_e))[:n_elec**2]
        else:
            idx = numpy.random.permutation(numpy.arange(N_e))[:n_elec]
    else:
        idx    = n_elec
        n_elec = numpy.sqrt(len(idx))
    pylab.figure()
    for count, i in enumerate(idx):
        if square:
            pylab.subplot(n_elec, n_elec, count + 1)
            if (numpy.mod(count, n_elec) != 0):
                pylab.setp(pylab.gca(), yticks=[])
            else:
                pylab.ylabel('Signal')
            if (count < n_elec*(n_elec - 1)):
                pylab.setp(pylab.gca(), xticks=[])
            else:
                pylab.xlabel('Time [ms]')
        else:
            pylab.subplot(n_elec, 1, count + 1)
            if count != (n_elec - 1):
                pylab.setp(pylab.gca(), xticks=[])
            else:
                pylab.xlabel('Time [ms]')
        pylab.plot(data[:, i], '0.25')
        xmin, xmax = pylab.xlim()
        pylab.scatter(peaks[i], data[peaks[i], i], s=10, c='r')
        pylab.xlim(xmin, xmax)
        pylab.plot([xmin, xmax], [-thresholds[i], -thresholds[i]], 'k--')
        pylab.plot([xmin, xmax], [thresholds[i], thresholds[i]], 'k--')
        pylab.title('Electrode %d' %i)
        if xzoom:
            pylab.xlim(xzoom[0], xzoom[1])
        pylab.ylim(-2*thresholds[i], 2*thresholds[i])
        if yzoom:
            pylab.ylim(yzoom[0], yzoom[1])
    pylab.tight_layout()
    pylab.show()
    return peaks


def raster_plot(file_name):

    result               = get_results(file_name)
    times                = []
    templates            = []
    for key in result['spiketimes'].keys():
        template    = int(key.split('_')[1])
        times     += result['spiketimes'][key].tolist()
        templates += [template]*len(result['spiketimes'][key])
    return numpy.array(times), numpy.array(templates)


def view_norms(file_name, save=True):
    """
    Sanity plot of the norms of the templates.
    
    Parameters
    ----------
    file_name : string
    
    """

    # Retrieve the key parameters.
    params = load_parameters(file_name)
    norms = load_data(params, 'norm-templates')
    N_tm = norms.shape[0] / 2
    y_margin = 0.1

    # Plot the figure.
    fig, ax = pylab.subplots(2, sharex=True)
    x = numpy.arange(0, N_tm, 1)
    y_cen = norms[0:N_tm]
    y_ort = norms[N_tm:2*N_tm]
    x_min = -1
    x_max = N_tm
    y_cen_dif = numpy.amax(y_cen) - numpy.amin(y_cen)
    y_cen_min = numpy.amin(y_cen) - y_margin * y_cen_dif
    y_cen_max = numpy.amax(y_cen) + y_margin * y_cen_dif
    y_ort_dif = numpy.amax(y_ort) - numpy.amin(y_ort)
    y_ort_min = numpy.amin(y_ort) - y_margin * y_ort_dif
    y_ort_max = numpy.amax(y_ort) + y_margin * y_ort_dif
    ax[0].plot(x, y_cen, 'o')
    ax[0].set_xlim([x_min, x_max])
    ax[0].set_ylim([y_cen_min, y_cen_max])
    ax[0].grid()
    ax[0].set_title("Norms of the %d templates in %s" %(N_tm, file_name))
    ax[0].set_xlabel("template (central component)")
    ax[0].set_ylabel("norm")
    ax[1].plot(x, y_ort, 'o')
    ax[1].set_ylim([y_ort_min, y_ort_max])
    ax[1].grid()
    ax[1].set_xlabel("template (orthogonal component)")
    ax[1].set_ylabel("norm")

    # Display figure.
    if save:
        fig.savefig("/tmp/norms-templates.pdf")
        pylab.close(fig)
    else:
        fig.show()

    return

def view_triggers(file_name, save=True):
    """
    Sanity plot of the triggers of a given dataset.
    
    Parameters
    ----------
    file_name : string
    save : boolean
    
    """
    # Retrieve the key parameters.
    params = load_parameters(file_name)
    spikes = load_data(params, 'triggers')

    fig, ax = pylab.subplots(4, 4)
    fig.suptitle("Spikes")
    for k in xrange(16):
        i = k / 4
        j = k % 4
        ax[i, j].imshow(spikes[:, :, i])

    # Display figure.
    if save:
        fig.savefig("/tmp/spikes.pdf")
        pylab.close(fig)
    else:
        fig.show()
    
    return
