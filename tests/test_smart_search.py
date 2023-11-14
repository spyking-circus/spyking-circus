import numpy, h5py, pylab, pickle
import unittest
from . import mpi_launch, get_dataset
from circus.shared.utils import *
from circus.shared.parser import CircusParser

def get_performance(file_name, name):

    a, b            = os.path.splitext(os.path.basename(file_name))
    file_name, ext  = os.path.splitext(file_name)
    file_out        = os.path.join(os.path.abspath(file_name), a)
    result_name     = os.path.join(file_name, 'injected')

    pic_name        = file_name + '.pic'
    data            = pickle.load(open(pic_name))
    n_cells         = data['cells'] 
    n_point         = len(n_cells)
    amplitude       = data['amplitudes'][0:n_point]
    rate            = data['rates'][:n_point]
    sampling        = data['sampling']
    probe_file      = data['probe']
    sim_templates   = 0.8

    temp_file       = file_out + '.templates.hdf5'
    temp_x          = h5py.File(temp_file).get('temp_x')[:]
    temp_y          = h5py.File(temp_file).get('temp_y')[:]
    temp_data       = h5py.File(temp_file).get('temp_data')[:]
    temp_shape      = h5py.File(temp_file).get('temp_shape')[:]
    templates       = scipy.sparse.csc_matrix((temp_data, (temp_x, temp_y)), shape=(temp_shape[0]*temp_shape[1], temp_shape[2]))

    temp_file       = os.path.join(result_name, '%s.templates.hdf5' %a)
    temp_x          = h5py.File(temp_file).get('temp_x')[:]
    temp_y          = h5py.File(temp_file).get('temp_y')[:]
    temp_data       = h5py.File(temp_file).get('temp_data')[:]
    temp_shape      = h5py.File(temp_file).get('temp_shape')[:]
    inj_templates   = scipy.sparse.csc_matrix((temp_data, (temp_x, temp_y)), shape=(temp_shape[0]*temp_shape[1], temp_shape[2]))

    amplitudes      = h5py.File(file_out + '.templates.hdf5').get('limits')[:]

    n_tm            = inj_templates.shape[1]//2
    res             = numpy.zeros(len(n_cells))
    res2            = numpy.zeros(len(n_cells))
    res3            = numpy.zeros(len(n_cells))

    for gcount, temp_id in enumerate(range(n_tm - len(n_cells), n_tm)):
        source_temp = inj_templates[:, temp_id].toarray().flatten()
        similarity  = []
        temp_match  = None
        dmax        = 0
        for i in range(templates.shape[1]//2):
            d = numpy.corrcoef(templates[:, i].toarray().flatten(), source_temp)[0, 1]
            similarity += [d]
            if d > dmax:
                temp_match = i
                dmax       = d
        res[gcount]  = numpy.max(similarity)
        res2[gcount] = numpy.sum(numpy.array(similarity) > sim_templates)
        res3[gcount] = temp_match


    pylab.figure()

    pylab.subplot(121)
    pylab.plot(rate, res, '.')
    pylab.xlabel('Rate [Hz]')
    pylab.ylabel('Correlation')
    pylab.xlim(rate.min()-0.5, rate.max()+0.5)
    pylab.ylim(0, 1.1)

    pylab.subplot(122)
    pylab.plot(rate, res2, '.')
    pylab.ylabel('Number of templates')
    pylab.xlabel('Rate [Hz]')
    pylab.xlim(rate.min()-0.5, rate.max()+0.5)

    pylab.tight_layout()

    plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    plot_path = os.path.join(plot_path, 'plots')
    plot_path = os.path.join(plot_path, 'smart-search')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    output = os.path.join(plot_path, '%s.pdf' %name)
    pylab.savefig(output)
    return templates, res2

class TestSmartSearch(unittest.TestCase):

    def setUp(self):
        self.all_matches    = None
        self.all_templates  = None
        dirname             = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
        self.path           = os.path.join(dirname, 'synthetic')
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.file_name      = os.path.join(self.path, 'smart_search.dat')
        self.source_dataset = get_dataset(self)
        if not os.path.exists(self.file_name):
            mpi_launch('benchmarking', self.source_dataset, 2, 0, 'False', self.file_name, 'smart-search', 1)
            mpi_launch('whitening', self.file_name, 2, 0, 'False')
        self.parser = CircusParser(self.file_name)
        self.parser.write('clustering', 'max_elts', '2000')

    #def tearDown(self):
    #    data_path = '.'.join(self.file_name.split('.')[:-1])
    #    shutil.rmtree(data_path)

    def test_smart_search_on(self):
        self.parser.write('clustering', 'smart_search', 'True')
        mpi_launch('clustering', self.file_name, 2, 0, 'False')
        self.parser.write('clustering', 'smart_search', 'False')
        res = get_performance(self.file_name, 'smart_search_on')

    def test_smart_search_off(self):
        mpi_launch('clustering', self.file_name, 2, 0, 'False')
        res = get_performance(self.file_name, 'smart_search_off')
