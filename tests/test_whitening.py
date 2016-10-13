import numpy, h5py, pylab, cPickle
import unittest
from . import mpi_launch, get_dataset
from circus.shared.utils import *
from circus.shared.parser import CircusParser

def get_performance(file_name, name):

    a, b            = os.path.splitext(os.path.basename(file_name))
    file_name, ext  = os.path.splitext(file_name)
    file_out        = os.path.join(os.path.abspath(file_name), a)
    data            = {}
    result          = h5py.File(file_out + '.basis.hdf5')
    data['spatial']  = result.get('spatial')[:]
    data['temporal'] = numpy.zeros(61) #result.get('temporal')[:]

    pylab.figure()
    pylab.subplot(121)
    pylab.imshow(data['spatial'], interpolation='nearest')
    pylab.title('Spatial')
    pylab.xlabel('# Electrode')
    pylab.ylabel('# Electrode')
    pylab.colorbar()
    pylab.subplot(122)
    pylab.title('Temporal')
    pylab.plot(data['temporal'])
    pylab.xlabel('Time [ms]')
    x, y = pylab.xticks()
    pylab.xticks(x, (x-x[-1]//2)//10)
    pylab.tight_layout()
    plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    plot_path = os.path.join(plot_path, 'plots')
    plot_path = os.path.join(plot_path, 'whitening')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    output = os.path.join(plot_path, '%s.pdf' %name)
    pylab.savefig(output)

    return data

class TestWhitening(unittest.TestCase):

    def setUp(self):
        dirname             = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
        self.path           = os.path.join(dirname, 'synthetic')
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.file_name      = os.path.join(self.path, 'whitening.dat')
        self.source_dataset = get_dataset(self)     
        self.whitening      = None
        if not os.path.exists(self.file_name):
            mpi_launch('benchmarking', self.source_dataset, 2, 0, 'False', self.file_name, 'fitting', 1) 
        self.params = CircusParser(self.file_name)
        self.params.write('clustering', 'max_elts', '1000')
        self.params.write('whitening', 'spatial', 'True')
        self.params.write('clustering', 'temporal', 'False')


    def test_whitening_one_CPU(self):
        mpi_launch('whitening', self.file_name, 1, 0, 'False')
        res = get_performance(self.file_name, 'one_CPU')
        if self.whitening is None:
            self.whitening = res
        assert ((res['spatial'] - self.whitening['spatial'])**2).mean() < 0.1

    def test_whitening_two_CPU(self):
        mpi_launch('whitening', self.file_name, 2, 0, 'False')
        res = get_performance(self.file_name, 'two_CPU')
        if self.whitening is None:
            self.whitening = res
        assert ((res['spatial'] - self.whitening['spatial'])**2).mean() < 0.1

    def test_whitening_safety_time(self):
        self.params.write('clustering', 'safety_time', '5')
        mpi_launch('whitening', self.file_name, 1, 0, 'False')
        self.params.write('clustering', 'safety_time', 'auto')
        res = get_performance(self.file_name, 'safety_time')
        if self.whitening is None:
            self.whitening = res
        assert ((res['spatial'] - self.whitening['spatial'])**2).mean() < 0.1