import numpy, h5py, pylab, pickle
import unittest
from . import mpi_launch, get_dataset
from circus.shared.utils import *
from circus.shared.parser import CircusParser

class TestConverting(unittest.TestCase):
    
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
            mpi_launch('benchmarking', self.source_dataset, 2, 0, 'False', self.file_name, 'fitting', 1)
            mpi_launch('whitening', self.file_name, 2, 0, 'False')
            self.parser = CircusParser(self.file_name)
            self.parser.write('fitting', 'max_chunk', '10')
            mpi_launch('fitting', self.file_name, 2, 0, 'False')
        else:
            self.parser = CircusParser(self.file_name)

    def test_converting_some(self):
        self.parser.write('converting', 'export_pcs', 'some')
        mpi_launch('converting', self.file_name, 1, 0, 'False')
        self.parser.write('converting', 'export_pcs', 'prompt')

    def test_converting_all(self):
        self.parser.write('converting', 'export_pcs', 'all')
        mpi_launch('converting', self.file_name, 2, 0, 'False')
        self.parser.write('converting', 'export_pcs', 'prompt')
