import numpy, hdf5storage, pylab, cPickle
import unittest
from . import mpi_launch
from circus.shared.utils import *

def get_performance(file_name, name):

    file_name = ".".join(file_name.split('.')[:-1])
    data      = hdf5storage.loadmat(file_name + '/' + file_name.split('/')[-1] + '.whitening.mat')

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
    pylab.xticks(x, (x-x[-1]/2)/10)
    pylab.tight_layout()
    if not os.path.exists('plots/whitening'):
        os.makedirs('plots/whitening')
    output = 'plots/whitening/%s.pdf' %name
    pylab.savefig(output)

    return data

class TestWhitening(unittest.TestCase):

    def setUp(self):
        self.file_name      = 'synthetic/whitening.raw'
        self.source_dataset = '/home/pierre/gpu/data/Dan/silico_0.dat'     
        self.whitening      = None
        if not os.path.exists(self.file_name):
            mpi_launch('benchmarking', self.source_dataset, 2, 0, 'False', self.file_name, 'fitting')   
        io.change_flag(self.file_name, 'max_elts', '1000', avoid_flag='Fraction')

    def test_whitening_one_CPU(self):
        mpi_launch('whitening', self.file_name, 1, 0, 'False')
        res = get_performance(self.file_name, 'one_CPU')
        if self.whitening is None:
            self.whitening = res
        assert (((res['spatial'] - self.whitening['spatial'])**2).mean() < 0.1) and (((res['temporal'] - self.whitening['temporal'])**2).mean() < 0.1)

    def test_whitening_two_CPU(self):
        mpi_launch('whitening', self.file_name, 2, 0, 'False')
        res = get_performance(self.file_name, 'two_CPU')
        if self.whitening is None:
            self.whitening = res
        assert (((res['spatial'] - self.whitening['spatial'])**2).mean() < 0.1) and (((res['temporal'] - self.whitening['temporal'])**2).mean() < 0.1)

    def test_whitening_safety_time(self):
        io.change_flag(self.file_name, 'safety_time', '5')
        mpi_launch('whitening', self.file_name, 1, 0, 'False')
        io.change_flag(self.file_name, 'safety_time', '0.5')
        res = get_performance(self.file_name, 'safety_time')
        if self.whitening is None:
            self.whitening = res
        assert (((res['spatial'] - self.whitening['spatial'])**2).mean() < 0.1) and (((res['temporal'] - self.whitening['temporal'])**2).mean() < 0.1)
