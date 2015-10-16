import numpy, hdf5storage, pylab, cPickle
import unittest
from . import mpi_launch
from circus.shared.utils import *

def test_MPI():
    HAVE_MPI = False
    try:
        import mpi4py
        HAVE_MPI = True
    except ImportError:
        pass
    assert HAVE_MPI == True 

def test_CUDA():
    HAVE_CUDA = False
    try:
        import cudamat
        HAVE_CUDA = True
    except ImportError:
        pass
    assert HAVE_CUDA == True