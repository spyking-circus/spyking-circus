import numpy, pylab, pickle
import unittest
from . import mpi_launch
from circus.shared.utils import *

class TestMPICUDA(unittest.TestCase):

    def test_MPI(self):
        HAVE_MPI = False
        try:
            import mpi4py
            HAVE_MPI = True
        except ImportError:
            pass
        assert HAVE_MPI == True 

    def test_CUDA(self):
        HAVE_CUDA = False
        try:
            import cudamat
            HAVE_CUDA = True
        except ImportError:
            pass
        assert HAVE_CUDA == True
