import numpy, hdf5storage, pylab, cPickle
import unittest
from . import mpi_launch, get_dataset
from circus.shared.utils import *

def get_performance(file_name, name):

    a, b            = os.path.splitext(os.path.basename(file_name))
    file_name, ext  = os.path.splitext(file_name)
    file_out        = os.path.join(os.path.abspath(file_name), a)
    result_name     = os.path.join(file_name, 'injected')

    pic_name        = file_name + '.pic'
    data            = cPickle.load(open(pic_name))
    n_cells         = data['cells']
    nb_insert       = len(n_cells)
    amplitude       = data['amplitudes']
    sampling        = data['sampling']
    thresh          = int(sampling*2*1e-3)
    sim_templates   = 0.8

    inj_templates   = hdf5storage.loadmat(os.path.join(result_name, 'templates.mat'))['templates']
    templates       = hdf5storage.loadmat(file_out + '.templates.mat')['templates']
    amplitudes      = hdf5storage.loadmat(file_out + '.limits.mat')['limits']
    clusters        = hdf5storage.loadmat(file_out + '.clusters.mat')
    fitted_spikes   = hdf5storage.loadmat(file_out + '.spiketimes.mat')
    fitted_amps     = hdf5storage.loadmat(file_out + '.amplitudes.mat')
    spikes          = hdf5storage.loadmat(os.path.join(result_name, 'spiketimes.mat'))
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

    pylab.tight_layout()
    plot_path = os.path.join('plots', 'complete')
    if not plot_path:
        os.makedirs(plot_path)
    output = os.path.join(plot_path, '%s.pdf' %name)
    pylab.savefig(output)
    pylab.savefig(output)

class TestCompleteWorkflow(unittest.TestCase):

    def setUp(self):
        self.path           = os.path.join(dirname, 'synthetic')
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.file_name      = os.path.join(self.path, 'complete.raw')
        self.source_dataset = get_dataset(self)
        if not os.path.exists(self.file_name):
            mpi_launch('benchmarking', self.source_dataset, 2, 0, 'False', self.file_name, 'clustering')

    #def tearDown(self):
    #    data_path = '.'.join(self.file_name.split('.')[:-1])
    #    shutil.rmtree(data_path)


    def test_all_two_CPU(self):
        mpi_launch('clustering', self.file_name, 2, 0, 'False')
        mpi_launch('fitting', self.file_name, 2, 0, 'False')
        res = get_performance(self.file_name, 'test')

