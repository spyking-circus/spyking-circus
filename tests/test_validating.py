import numpy, h5py, pylab, cPickle
import unittest
from . import mpi_launch, get_dataset
from circus.shared.utils import *
from circus.shared.parser import CircusParser

class TestValidating(unittest.TestCase):
    
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
        self.length = self.parser.get_data_file().duration



    def test_validating(self):
        #mpi_launch('fitting', self.file_name, 2, 0, 'False')
        

        a, b            = os.path.splitext(os.path.basename(self.file_name))
        file_name, ext  = os.path.splitext(self.file_name)
        file_out        = os.path.join(os.path.abspath(file_name), a)
        result_name     = os.path.join(file_name, 'injected')
        spikes          = {}
        result          = h5py.File(os.path.join(result_name, '%s.result.hdf5' %a))
        for key in result.get('spiketimes').keys():
            spikes[key] = result.get('spiketimes/%s' %key)[:]

        juxta_file = file_out + '.juxta.dat'

        f = numpy.memmap(juxta_file, shape=(self.length,1), dtype=self.parser.get('validating', 'juxta_dtype'), mode='w+')
        f[spikes['temp_9']] = 100
        del f
        print self.length, len(spikes['temp_9'])

        mpi_launch('validating', self.file_name, 2, 0, 'False')