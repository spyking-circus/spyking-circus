import numpy, hdf5storage, pylab, cPickle
import unittest
from . import mpi_launch
from circus.shared.utils import *

def get_performance(file_name, n_cpu):

    file_name       = ".".join(file_name.split('.')[:-1])
    pic_name        = file_name + '.pic'
    data            = cPickle.load(open(pic_name))
    n_cells         = data['cells'] 
    n_point         = numpy.sqrt(len(n_cells))
    amplitude       = data['amplitudes'][0:n_point]
    rate            = data['rates'][::n_point]
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
    pylab.yticks(numpy.linspace(0.5, n_point-0.5, 5), numpy.round(rate, 1))
    pylab.xticks(numpy.linspace(0.5, n_point-0.5, 5), numpy.round(amplitude, 1))
    pylab.ylabel('Rate [Hz]')
    pylab.xlabel('Relative Amplitude')
    pylab.xlim(-0.5, n_point-0.5)
    pylab.ylim(-0.5, n_point-0.5)

    pylab.subplot(222)
    pylab.imshow(res2.reshape(n_point, n_point).astype(numpy.int32), aspect='auto', interpolation='nearest', origin='lower')
    cb = pylab.colorbar()
    cb.set_label('Number of templates')
    pylab.yticks(numpy.linspace(0.5, n_point-0.5, 5), numpy.round(rate, 1))
    pylab.xticks(numpy.linspace(0.5, n_point-0.5, 5), numpy.round(amplitude, 1))
    pylab.ylabel('Rate [Hz]')
    pylab.xlabel('Relative Amplitude')
    pylab.xlim(-0.5, n_point-0.5)
    pylab.ylim(-0.5, n_point-0.5)

    pylab.subplot(223)
    pylab.imshow(amplitudes[-len(n_cells):][:,0].reshape(n_point, n_point), aspect='auto', interpolation='nearest', origin='lower')
    cb = pylab.colorbar()
    cb.set_label('Min amplitude')
    pylab.yticks(numpy.linspace(0.5, n_point-0.5, 5), numpy.round(rate, 1))
    pylab.xticks(numpy.linspace(0.5, n_point-0.5, 5), numpy.round(amplitude, 1))
    pylab.ylabel('Rate [Hz]')
    pylab.xlabel('Relative Amplitude')
    pylab.xlim(-0.5, n_point-0.5)
    pylab.ylim(-0.5, n_point-0.5)

    pylab.subplot(224)
    pylab.imshow(amplitudes[-len(n_cells):][:,1].reshape(n_point, n_point), aspect='auto', interpolation='nearest', origin='lower')
    cb = pylab.colorbar()
    cb.set_label('Max amplitude')
    pylab.yticks(numpy.linspace(0.5, n_point-0.5, 5), numpy.round(rate, 1))
    pylab.xticks(numpy.linspace(0.5, n_point-0.5, 5), numpy.round(amplitude, 1))
    pylab.ylabel('Rate [Hz]')
    pylab.xlabel('Relative Amplitude')
    pylab.xlim(-0.5, n_point-0.5)
    pylab.ylim(-0.5, n_point-0.5)

    pylab.tight_layout()

    output = 'plots/test_clustering_%d.pdf' %n_cpu
    pylab.savefig(output)
    return templates, res2


class TestClustering(unittest.TestCase):


    def setUp(self):
        self.all_matches    = None
        self.all_templates  = None
        self.file_name      = 'synthetic/clustering.raw'
        self.source_dataset = '/home/pierre/gpu/data/Dan/silico_0.dat'
        if not os.path.exists(self.file_name):
            mpi_launch('benchmarking', self.source_dataset, 2, 0, 'False', self.file_name, 'clustering')

    def test_clustering_one_CPU(self):
        mpi_launch('clustering', self.file_name, 1, 0, 'False')
        res = get_performance(self.file_name, 1, 0)
        if self.all_templates is None:
            self.all_templates = res[0]
            self.all_matches   = res[1]
        assert numpy.all(self.all_templates == templates)
        
    def test_clustering_two_CPU(self):
        mpi_launch('clustering', self.file_name, 2, 0, 'False')
        res = get_performance(self.file_name, 2, 0)
        if self.all_templates is None:
            self.all_templates = res[0]
            self.all_matches   = res[1]
        assert numpy.all(self.all_templates == templates)

    def test_clustering_smart_search(self):
        io.change_flag(self.file_name, 'smart_search', '0')
        mpi_launch('clustering', self.file_name, 2, 0, 'False')
        io.change_flag(self.file_name, 'smart_search', '3')
        res = get_performance(self.file_name, 2, 0)
        if self.all_templates is None:
            self.all_templates = res[0]
            self.all_matches   = res[1]
        assert numpy.all(self.all_templates == templates)

    def test_clustering_nb_passes(self):
        io.change_flag(self.file_name, 'nb_repeats', '1')
        mpi_launch('clustering', self.file_name, 2, 0, 'False')
        io.change_flag(self.file_name, 'nb_repeats', '3')
        res = get_performance(self.file_name, 2, 0)
        if self.all_templates is None:
            self.all_templates = res[0]
            self.all_matches   = res[1]
        assert numpy.all(self.all_templates == templates)

    def test_clustering_sim_same_elec(self):
        io.change_flag(self.file_name, 'sim_same_elec', '5')
        mpi_launch('clustering', self.file_name, 2, 0, 'False')
        io.change_flag(self.file_name, 'sim_same_elec', '3')
        res = get_performance(self.file_name, 2, 0)
        if self.all_templates is None:
            self.all_templates = res[0]
            self.all_matches   = res[1]
        assert numpy.sum(res[1]) <= nump.sum(self.all_matches)

    def test_clustering_cc_merge(self):
        io.change_flag(self.file_name, 'cc_merge', '0.5')
        mpi_launch('clustering', self.file_name, 2, 0, 'False')
        io.change_flag(self.file_name, 'cc_merge', '0.975')
        res = get_performance(self.file_name, 2, 0)
        if self.all_templates is None:
            self.all_templates = res[0]
            self.all_matches   = res[1]
        assert res[0].shape[2]/2 <= self.all_templates.shape[2]/2
