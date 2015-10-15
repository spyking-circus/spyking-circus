import numpy, hdf5storage, pylab, cPickle

file_name       = '/home/pierre/synthetic/fake_1'
data            = cPickle.load(open(file_name + '.pic'))
n_cells         = data['cells'] 
nb_insert       = len(n_cells)
amplitude       = data['amplitudes']
sampling        = data['sampling']
thresh          = int(sampling*2*1e-3)
truncate        = False

fitted_spikes   = hdf5storage.loadmat(file_name + '/' + file_name.split('/')[-1] + '.spiketimes.mat')
fitted_amps     = hdf5storage.loadmat(file_name + '/' + file_name.split('/')[-1] + '.amplitudes.mat')
spikes          = hdf5storage.loadmat(file_name + '/injected/spiketimes.mat')
templates       = hdf5storage.loadmat(file_name + '/' + file_name.split('/')[-1] + '.templates.mat')['templates']
real_amps       = hdf5storage.loadmat(file_name + '/injected/real_amps.mat')
voltages        = hdf5storage.loadmat(file_name + '/injected/voltages.mat')
n_tm            = templates.shape[2]/2
res             = numpy.zeros((len(n_cells), 2))
res2            = numpy.zeros((len(n_cells), 2))
real_amplitudes = []
p_deltas        = {}
n_deltas        = {}
p_amps          = {}
n_amps          = {}

if truncate:
    max_spike = 0
    for temp_id in xrange(n_tm - len(n_cells), n_tm):
        key = 'temp_' + str(temp_id)
        if len(fitted_spikes[key] > 0):
            max_spike = max(max_spike, fitted_spikes[key].max())
    for temp_id in xrange(n_tm - len(n_cells), n_tm):
        key = 'temp_' + str(temp_id)
        spikes[key] = spikes[key][spikes[key] < max_spike]

for gcount, temp_id in enumerate(xrange(n_tm - len(n_cells), n_tm)):
    key = 'temp_' + str(temp_id)
    count = 0
    p_deltas[key] = []
    n_deltas[key] = []
    p_amps[key]   = []
    n_amps[key]   = []
    for spike in spikes[key]:
        idx = numpy.where(numpy.abs(fitted_spikes[key] - spike) < thresh)[0]
        if len(idx) > 0:
            count += 1
            p_deltas[key] += [numpy.abs(fitted_spikes[key][idx] - spike)[0]]
            p_amps[key]   += [fitted_amps[key][idx][:, 0][0]]
    if len(spikes[key]) > 0:
        res[gcount, 0] = count/float(len(spikes[key]))

    count = 0
    for lcount, spike in enumerate(fitted_spikes[key]):
        idx = numpy.where(numpy.abs(spikes[key] - spike) < thresh)[0]
        if len(idx) > 0:
            count += 1
            n_deltas[key] += [numpy.abs(spikes[key][idx] - spike)[0]]
            n_amps[key]   += [fitted_amps[key][lcount]]
    if len(fitted_spikes[key]) > 0:
        res[gcount, 1]  = count/(float(len(fitted_spikes[key])))
    
    res2[gcount, 0] = numpy.mean(fitted_amps[key][:, 0])
    res2[gcount, 1] = numpy.var(fitted_amps[key][:, 0])
    
    real_amplitudes += [numpy.mean(real_amps[key])]

    print key, len(spikes[key]), len(fitted_spikes[key]), res[gcount]

pylab.figure()
ax = pylab.subplot(211)
pylab.plot(amplitude, 100*(1 - res[:, 0]))
pylab.plot(amplitude, 100*(1 - res[:, 1]))
ax.set_yscale('log')
pylab.xlim(amplitude[0], amplitude[-1])
pylab.setp(pylab.gca(), xticks=[])
pylab.legend(('False negative', 'False positive'))
pylab.ylabel('Errors [%]')

pylab.subplot(212)
pylab.errorbar(amplitude*real_amplitudes, amplitude*res2[:,0], yerr=res2[:,1])
pylab.xlabel('Relative Amplitude')
pylab.ylabel('Fitted amplitude')
pylab.xlim(amplitude[0], amplitude[-1])
pylab.show()

pylab.tight_layout()
output = 'plots/test_fitting.pdf'
pylab.savefig(output)