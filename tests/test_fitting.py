import numpy, hdf5storage, pylab, cPickle
import unittest
from . import mpi_launch
from circus.shared.utils import *

def get_performance(file_name, name):

    file_name       = ".".join(file_name.split('.')[:-1])
    pic_name        = file_name + '.pic'
    data            = cPickle.load(open(pic_name))
    n_cells         = data['cells'] 
    nb_insert       = len(n_cells)
    amplitude       = data['amplitudes']
    sampling        = data['sampling']
    thresh          = int(sampling*2*1e-3)
    truncate        = True

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
    if not os.path.exists('plots/fitting'):
        os.makedirs('plots/fitting')
    output = 'plots/fitting/%s.pdf' %name
    pylab.savefig(output)
    return res

class TestFitting(unittest.TestCase):
    
    def setUp(self):
        self.all_spikes     = None
        self.max_chunk      = '20'
        self.file_name      = 'synthetic/fitting.raw'
        self.source_dataset = '/home/pierre/gpu/data/Dan/silico_0.dat'
        if not os.path.exists(self.file_name):
            mpi_launch('benchmarking', self.source_dataset, 2, 0, 'False', self.file_name, 'fitting')

    def test_fitting_one_CPU(self):
        io.change_flag(self.file_name, 'max_chunk', self.max_chunk)
        mpi_launch('fitting', self.file_name, 1, 0, 'False')
        io.change_flag(self.file_name, 'max_chunk', 'inf')
        res = get_performance(self.file_name, 'one_CPU')
        if self.all_spikes is None:
            self.all_spikes = res
        assert numpy.all(self.all_spikes == res)

    def test_fitting_two_CPUs(self):
        io.change_flag(self.file_name, 'max_chunk', self.max_chunk)
        mpi_launch('fitting', self.file_name, 2, 0, 'False')
        io.change_flag(self.file_name, 'max_chunk', 'inf')
        res = get_performance(self.file_name, 'two_CPU')
        if self.all_spikes is None:
            self.all_spikes = res
        assert numpy.all(self.all_spikes == res)
    
    def test_fitting_one_GPU(self):
        HAVE_CUDA = False
        try:
            import cudamat
            HAVE_CUDA = True
        except ImportError:
            pass
        if HAVE_CUDA:
            io.change_flag(self.file_name, 'max_chunk', self.max_chunk)
            mpi_launch('fitting', self.file_name, 1, 0, 'False')
            io.change_flag(self.file_name, 'max_chunk', 'inf')
            res = get_performance(self.file_name, 'one_GPU')
            if self.all_spikes is None:
                self.all_spikes = res
            assert numpy.all(self.all_spikes == res)

    def test_fitting_large_chunks(self):
        io.change_flag(self.file_name, 'chunk', '1', avoid_flag='max_chunk')
        io.change_flag(self.file_name, 'max_chunk', str(int(self.max_chunk)/2))
        mpi_launch('fitting', self.file_name, 2, 0, 'False')
        io.change_flag(self.file_name, 'max_chunk', 'inf')
        io.change_flag(self.file_name, 'chunk', '0.5', avoid_flag='max_chunk')
        res = get_performance(self.file_name, 'large_chunks')
        if self.all_spikes is None:
            self.all_spikes = res
        assert numpy.all(self.all_spikes == res)

    def test_fitting_refractory(self):
        io.change_flag(self.file_name, 'max_chunk', self.max_chunk)
        io.change_flag(self.file_name, 'refractory', '5')
        mpi_launch('fitting', self.file_name, 2, 0, 'False')
        io.change_flag(self.file_name, 'refractory', '0')
        io.change_flag(self.file_name, 'max_chunk', 'inf')
        res = get_performance(self.file_name, 'refractory')
        if self.all_spikes is None:
            self.all_spikes = res
        assert numpy.all(self.all_spikes == res)

    def test_fitting_spike_range(self):
        io.change_flag(self.file_name, 'max_chunk', self.max_chunk)
        io.change_flag(self.file_name, 'spike_range', '1')
        mpi_launch('fitting', self.file_name, 2, 0, 'False')
        io.change_flag(self.file_name, 'spike_range', '0')
        io.change_flag(self.file_name, 'max_chunk', 'inf')
        res = get_performance(self.file_name, 'spike_range')
        if self.all_spikes is None:
            self.all_spikes = res
        assert numpy.all(self.all_spikes == res)
