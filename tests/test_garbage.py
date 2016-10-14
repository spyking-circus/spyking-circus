import numpy, h5py, pylab, cPickle
import unittest
from . import mpi_launch, get_dataset
from circus.shared.utils import *
from circus.shared.parser import CircusParser

def get_performance(file_name):

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
    truncate        = True

    result          = h5py.File(file_out + '.result.hdf5')
    fitted_spikes   = {}
    fitted_amps     = {}
    garbage         = {}
    cgarbage        = 0
    cspikes         = 0
    for key in result.get('spiketimes').keys():
        fitted_spikes[key] = result.get('spiketimes/%s' %key)[:]
        cspikes += len(fitted_spikes[key])
    for key in result.get('amplitudes').keys():
        fitted_amps[key]   = result.get('amplitudes/%s' %key)[:]
    for key in result.get('gspikes').keys():
        garbage[key] = result.get('gspikes/%s' %key)[:]
        cgarbage += len(garbage[key])

    templates       = h5py.File(file_out + '.templates.hdf5').get('temp_shape')[:]
    
    spikes          = {}
    real_amps       = {}
    ctruth          = 0
    result          = h5py.File(os.path.join(result_name, '%s.result.hdf5' %a))
    for key in result.get('spiketimes').keys():
        spikes[key] = result.get('spiketimes/%s' %key)[:]
        ctruth     += len(spikes[key])
    for key in result.get('real_amps').keys():
        real_amps[key]   = result.get('real_amps/%s' %key)[:]

    n_tm            = templates[2]//2
    res             = numpy.zeros((len(n_cells), 2))
    res2            = numpy.zeros((len(n_cells), 2))
    real_amplitudes = []
    p_deltas        = {}
    n_deltas        = {}
    p_amps          = {}
    n_amps          = {}
    
    return ctruth, cspikes, cgarbage


class TestGarbage(unittest.TestCase):
    
    def setUp(self):
        self.all_spikes     = None
        self.max_chunk      = '100'
        dirname             = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
        self.path           = os.path.join(dirname, 'synthetic')
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.file_name      = os.path.join(self.path, 'fitting.dat')
        self.source_dataset = get_dataset(self)
        if not os.path.exists(self.file_name):
            mpi_launch('benchmarking', self.source_dataset, 2, 0, 'False', self.file_name, 'fitting')
            mpi_launch('whitening', self.file_name, 2, 0, 'False')
        self.parser = CircusParser(self.file_name)


    def test_collect_all(self):
        self.parser.write('fitting', 'max_chunk', self.max_chunk)
        self.parser.write('fitting', 'collect_all', 'True')
        mpi_launch('fitting', self.file_name, 1, 0, 'False')
        self.parser.write('fitting', 'max_chunk', 'inf')
        self.parser.write('fitting', 'collect_all', 'False')
        ctruth, cspikes, cgarbage = get_performance(self.file_name)
        assert cgarbage < cspikes