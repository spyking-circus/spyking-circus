import numpy, hdf5storage, pylab, cPickle

file_name       = '/home/pierre/synthetic/fake_2'
data            = cPickle.load(open(file_name + '.pic'))
n_cells         = data['cells']
nb_insert       = len(n_cells)
amplitude       = data['amplitudes']
sampling        = data['sampling']
probe_file      = data['probe']
thresh          = int(sampling*2*1e-3)
sim_templates   = 0.8

inj_templates   = hdf5storage.loadmat(file_name + '/injected/templates.mat')['templates']
templates       = hdf5storage.loadmat(file_name + '/' + file_name.split('/')[-1] + '.templates.mat')['templates']
amplitudes      = hdf5storage.loadmat(file_name + '/' + file_name.split('/')[-1] + '.limits.mat')['limits']
clusters        = hdf5storage.loadmat(file_name + '/' + file_name.split('/')[-1] + '.clusters.mat')
fitted_spikes   = hdf5storage.loadmat(file_name + '/' + file_name.split('/')[-1] + '.spiketimes.mat')
fitted_amps     = hdf5storage.loadmat(file_name + '/' + file_name.split('/')[-1] + '.amplitudes.mat')
spikes          = hdf5storage.loadmat(file_name + '/injected/spiketimes.mat')
n_tm            = inj_templates.shape[2]/2
res             = numpy.zeros(len(n_cells))
res2            = numpy.zeros(len(n_cells))
res3            = numpy.zeros((len(n_cells), 2))

for gcount, temp_id in enumerate(xrange(n_tm - len(n_cells), n_tm)):
    source_temp = inj_templates[:, :, temp_id]
    similarity  = []
    temp_match  = []
    dmax        = 0
    for i in xrange(templates.shape[2]/2):
        d = numpy.corrcoef(templates[:, :, i].flatten(), source_temp.flatten())[0, 1]
        similarity += [d]
        if d > dmax:
            temp_match += [i]
            dmax       = d
    res[gcount]  = numpy.max(similarity)
    res2[gcount] = numpy.sum(numpy.array(similarity) > sim_templates)
    if res2[gcount] > 0:

        all_fitted_spikes = []
        for tmp in temp_match:
            all_fitted_spikes += fitted_spikes['temp_' + str(tmp)].tolist()
        all_fitted_spikes = numpy.array(all_fitted_spikes, dtype=numpy.int32)

        key1   = 'temp_' + str(temp_id)
        count = 0
        for spike in spikes[key1]:
            idx = numpy.where(numpy.abs(all_fitted_spikes - spike) < thresh)[0]
            if len(idx) > 0:
                count += 1
        if len(spikes[key1]) > 0:
            res3[gcount, 0] = count/float(len(spikes[key1]))

        count = 0
        for lcount, spike in enumerate(all_fitted_spikes):
            idx = numpy.where(numpy.abs(spikes[key1] - spike) < thresh)[0]
            if len(idx) > 0:
                count += 1
        if len(all_fitted_spikes) > 0:
            res3[gcount, 1]  = count/(float(len(all_fitted_spikes)))

pylab.figure()
pylab.subplot(121)
pylab.plot(amplitude, 100*(1 - res3[:,0]), '.')
pylab.xlabel('Relative Amplitude')
pylab.ylabel('Error Rate')
pylab.title('False Negative')

pylab.subplot(122)
pylab.plot(amplitude, 100*(1 - res3[:,1]), '.')
pylab.xlabel('Relative Amplitude')
pylab.ylabel('Error Rate')
pylab.title('False Positive')

pylab.savefig('plots/test_all.pdf')


