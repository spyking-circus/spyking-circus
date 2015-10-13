import numpy, hdf5storage, pylab, cPickle

file_name       = '/home/pierre/synthetic/fake_2'
data            = cPickle.load(open(file_name + '.pic'))
n_cells         = data['cells'] 
nb_insert       = len(n_cells)
amplitude       = data['amplitudes']
sampling        = data['sampling']
probe_file      = data['probe']
sim_templates   = 0.8

inj_templates   = hdf5storage.loadmat(file_name + '/injected/templates.mat')['templates']
templates       = hdf5storage.loadmat(file_name + '/' + file_name.split('/')[-1] + '.templates.mat')['templates']
amplitudes      = hdf5storage.loadmat(file_name + '/' + file_name.split('/')[-1] + '.limits.mat')['limits']
clusters        = hdf5storage.loadmat(file_name + '/' + file_name.split('/')[-1] + '.clusters.mat')
real_amps       = hdf5storage.loadmat(file_name + '/injected/real_amps.mat')
n_tm            = inj_templates.shape[2]/2
res             = numpy.zeros(len(n_cells))
res2            = numpy.zeros(len(n_cells))
res3            = numpy.zeros(len(n_cells))

for gcount, temp_id in enumerate(xrange(n_tm - len(n_cells), n_tm)):
    source_temp = inj_templates[:, :, temp_id]
    similarity  = []
    temp_match  = None
    dmax        = 0
    for i in xrange(templates.shape[2]/2):
        d = numpy.corrcoef(templates[:, :, i].flatten(), source_temp.flatten())[0, 1]
        similarity += [d]
        if d > dmax:
            temp_match = i
            dmax       = d
    res[gcount]  = numpy.max(similarity)
    res2[gcount] = numpy.sum(numpy.array(similarity) > sim_templates)
    res3[gcount] = temp_match

pylab.figure()

pylab.subplot(221)
pylab.imshow(res.reshape(n_point, n_point), aspect='auto', interpolation='nearest', origin='lower')
cb = pylab.colorbar()
cb.set_label('Correlation')
pylab.yticks(numpy.linspace(0.5, n_point-0.5, 5), numpy.round(numpy.linspace(0.5, 50, 5), 1))
pylab.xticks(numpy.linspace(0.5, n_point-0.5, 5), numpy.round(numpy.linspace(0.5, 5, 5), 1))
pylab.ylabel('Rate [Hz]')
pylab.xlabel('Relative Amplitude')
pylab.xlim(-0.5, n_point-0.5)
pylab.ylim(-0.5, n_point-0.5)


pylab.subplot(222)
pylab.imshow(res2.reshape(n_point, n_point).astype(numpy.int32), aspect='auto', interpolation='nearest', origin='lower')
cb = pylab.colorbar()
cb.set_label('Number of templates')
pylab.yticks(numpy.linspace(0.5, n_point-0.5, 5), numpy.round(numpy.linspace(0.5, 50, 5), 1))
pylab.xticks(numpy.linspace(0.5, n_point-0.5, 5), numpy.round(numpy.linspace(0.5, 5, 5), 1))
pylab.ylabel('Rate [Hz]')
pylab.xlabel('Relative Amplitude')
pylab.xlim(-0.5, n_point-0.5)
pylab.ylim(-0.5, n_point-0.5)


pylab.subplot(223)
pylab.imshow(amplitudes[-len(n_cells):][:,0].reshape(n_point, n_point), aspect='auto', interpolation='nearest', origin='lower')
cb = pylab.colorbar()
cb.set_label('Min amplitude')
pylab.yticks(numpy.linspace(0.5, n_point-0.5, 5), numpy.round(numpy.linspace(0.5, 50, 5), 1))
pylab.xticks(numpy.linspace(0.5, n_point-0.5, 5), numpy.round(numpy.linspace(0.5, 5, 5), 1))
pylab.ylabel('Rate [Hz]')
pylab.xlabel('Relative Amplitude')
pylab.xlim(-0.5, n_point-0.5)
pylab.ylim(-0.5, n_point-0.5)


pylab.subplot(224)
pylab.imshow(amplitudes[-len(n_cells):][:,1].reshape(n_point, n_point), aspect='auto', interpolation='nearest', origin='lower')
cb = pylab.colorbar()
cb.set_label('Max amplitude')
pylab.yticks(numpy.linspace(0.5, n_point-0.5, 5), numpy.round(numpy.linspace(0.5, 50, 5), 1))
pylab.xticks(numpy.linspace(0.5, n_point-0.5, 5), numpy.round(numpy.linspace(0.5, 5, 5), 1))
pylab.ylabel('Rate [Hz]')
pylab.xlabel('Relative Amplitude')
pylab.xlim(-0.5, n_point-0.5)
pylab.ylim(-0.5, n_point-0.5)


pylab.tight_layout()

output = 'plots/test_clustering.pdf'
pylab.savefig(output)
