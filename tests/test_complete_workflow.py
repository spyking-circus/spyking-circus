import numpy, h5py, pylab, cPickle
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

    print os.path.join(result_name, '.templates.hdf5')
    inj_templates   = h5py.File(os.path.join(result_name, '%s.templates.hdf5' %a)).get('templates')[:]
    templates       = h5py.File(file_out + '.templates.hdf5').get('templates')[:]
    
    result          = h5py.File(file_out + '.result.hdf5')
    fitted_spikes   = {}
    fitted_amps     = {}
    for key in result.get('spiketimes').keys():
        fitted_spikes[key] = result.get('spiketimes/%s' %key)[:]
    for key in result.get('amplitudes').keys():
        fitted_amps[key]   = result.get('amplitudes/%s' %key)[:]

    spikes          = {}
    real_amps       = {}
    result          = h5py.File(os.path.join(result_name, '%s.result.hdf5' %a))
    for key in result.get('spiketimes').keys():
        spikes[key] = result.get('spiketimes/%s' %key)[:]
    for key in result.get('real_amps').keys():
        real_amps[key]   = result.get('real_amps/%s' %key)[:]
    
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
    plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    plot_path = os.path.join(plot_path, 'plots')
    plot_path = os.path.join(plot_path, 'complete')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    output = os.path.join(plot_path, '%s.pdf' %name)
    pylab.savefig(output)
    pylab.savefig(output)


class TestCompleteWorkflow(unittest.TestCase):

    def setUp(self):
        dirname             = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
        self.path           = os.path.join(dirname, 'synthetic')
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.file_name      = os.path.join(self.path, 'complete.raw')
        self.source_dataset = get_dataset(self)
        if not os.path.exists(self.file_name):
            mpi_launch('benchmarking', self.source_dataset, 2, 0, 'False', self.file_name, 'clustering')

    def test_all_two_CPU(self):
        mpi_launch('clustering', self.file_name, 2, 0, 'False')
        mpi_launch('fitting', self.file_name, 2, 0, 'False')
        res = get_performance(self.file_name, 'test')

