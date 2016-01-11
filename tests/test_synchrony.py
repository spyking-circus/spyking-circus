import numpy, h5py, pylab, cPickle
import unittest
from . import mpi_launch, get_dataset
from circus.shared.utils import *

def get_performance(file_name, t_stop, name):

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
    probe_file      = data['probe']
    thresh          = int(sampling*2*1e-3)
    truncate        = True
    bin_cc          = 10
    t_stop          = t_stop*1000

    result          = h5py.File(file_out + '.result.hdf5')
    fitted_spikes   = {}
    fitted_amps     = {}
    for key in result.get('spiketimes').keys():
        fitted_spikes[key] = result.get('spiketimes/%s' %key)[:]
    for key in result.get('amplitudes').keys():
        fitted_amps[key]   = result.get('amplitudes/%s' %key)[:]

    templates       = h5py.File(file_out + '.templates.hdf5').get('templates')[:]

    spikes          = {}
    real_amps       = {}
    result          = h5py.File(os.path.join(result_name, '%s.result.hdf5' %a))
    for key in result.get('spiketimes').keys():
        spikes[key] = result.get('spiketimes/%s' %key)[:]
    for key in result.get('real_amps').keys():
        real_amps[key]   = result.get('real_amps/%s' %key)[:]

    clusters        = h5py.File(file_out + '.clusters.hdf5').get('electrodes')[:]

    N_t             = templates.shape[1]
    n_tm            = templates.shape[2]/2
    res             = numpy.zeros((len(n_cells), 2))
    res2            = numpy.zeros((len(n_cells), 2))
    real_amplitudes = []

    if truncate:
        max_spike = 0
        for temp_id in xrange(n_tm - len(n_cells), n_tm):
            key = 'temp_' + str(temp_id)
            if len(fitted_spikes[key] > 0):
                max_spike = max(max_spike, fitted_spikes[key].max())
        for temp_id in xrange(n_tm - len(n_cells), n_tm):
            key = 'temp_' + str(temp_id)
            spikes[key] = spikes[key][spikes[key] < max_spike]

    def fast_cc(t1, t2, bin=20, max_delay=10):
        
        t1b = numpy.floor(t1/bin)
        t2b = numpy.floor(t2/bin)

        x1  = numpy.zeros(t_stop/bin)
        x2  = numpy.zeros(t_stop/bin)

        for i in t1b:
            x1[i] += 1
        for j in t2b:
            x2[j] += 1
        
        return numpy.corrcoef(x1, x2)[0, 1]

    res = []
    src = []
    ids = []

    for gcount, temp_id in enumerate(xrange(n_tm - len(n_cells), n_tm)):
        key    = 'temp_' + str(temp_id)
        res   += [fitted_spikes[key]/(float(sampling)/1000.)]
        src   += [spikes[key]/(float(sampling)/1000.)]
        ids   += [temp_id]

    cc = numpy.zeros((nb_insert, nb_insert))
    cd = numpy.zeros((nb_insert, nb_insert))

    for i in xrange(nb_insert):
        for j in xrange(nb_insert):
            cc[i,j] = fast_cc(res[i], res[j])
            cd[i,j] = fast_cc(src[i], src[j])


    pylab.figure()
    pylab.subplot(221)
    probe            = {}
    probetext        = file(probe_file, 'r')
    try:
        exec probetext in probe
    except Exception:
        io.print_error(["Something wrong with the probe file!"])
    probetext.close()

    positions = {}
    for i in probe['channel_groups'].keys():
        positions.update(probe['channel_groups'][i]['geometry'])
    xmin = 0
    xmax = 0
    ymin = 0
    ymax = 0
    N_total          = probe['total_nb_channels']
    N_e              = len(probe['channel_groups'][1]['geometry'].keys())
    nodes            = numpy.arange(N_e)
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)
    scaling = 10*numpy.max(numpy.abs(templates[:,:,temp_id]))
    for i in xrange(N_e):
        if positions[i][0] < xmin:
            xmin = positions[i][0]
        if positions[i][0] > xmax:
            xmax = positions[i][0]
        if positions[i][1] < ymin:
            ymin = positions[i][0]
        if positions[i][1] > ymax:
            ymax = positions[i][1]
        
    colors = ['r', 'b', 'g', 'k', 'c']
    for gcount, temp_id in enumerate(xrange(n_tm - len(n_cells), n_tm)):
        best_elec = clusters[gcount]
        for count, i in enumerate(xrange(N_e)):
            x, y     = positions[i]
            xpadding = ((x - xmin)/float(xmax - xmin))*(2*N_t)
            ypadding = ((y - ymin)/float(ymax - ymin))*scaling
            if i == best_elec and gcount == 0:
                pylab.axvspan(xpadding, xpadding+N_t, 0.8, 1, color='0.5', alpha=0.5)
            pylab.plot(xpadding + numpy.arange(0, N_t), ypadding + templates[i, :, temp_id], color=colors[gcount])


    pylab.setp(pylab.gca(), xticks=[], yticks=[])
    pylab.xlim(xmin, 3*N_t)

    pylab.subplot(222)
    cc_nodiag = []
    cd_nodiag = []
    for i in xrange(cc.shape[0]):
        for j in xrange(i+1, cc.shape[1]):
            cd_nodiag += [cd[i, j]]
            cc_nodiag += [cc[i, j]]
    pylab.plot(cd_nodiag, cc_nodiag, '.')
    pylab.ylabel(r'Injected $<CC(0)>$')
    pylab.xlabel(r'Recovered $<CC(0)>$')
    pylab.xlim(0, 1)
    pylab.ylim(0, 1)
    pylab.plot([0, 1], [0, 1], 'k--')

    pylab.subplot(223)
    for count, i in enumerate(ids):
        pylab.scatter(fitted_spikes['temp_%d' %i], count + numpy.ones(len(fitted_spikes['temp_%d' %i])))
        pylab.xlim(0, sampling)
        x, y = pylab.xticks()
        pylab.xticks(x, numpy.round(numpy.linspace(0, 1, len(x)), 1))
    pylab.title('Recovered')
    pylab.xlabel('Time [s]')
    pylab.ylabel('Neuron')

    pylab.subplot(224)
    for count, i in enumerate(ids):
        pylab.scatter(spikes['temp_%d' %i], count + numpy.ones(len(spikes['temp_%d' %i])))
        pylab.xlim(0, sampling)
        x, y = pylab.xticks()
        pylab.xticks(x, numpy.round(numpy.linspace(0, 1, len(x)), 1))
    pylab.title('Injected')
    pylab.xlabel('Time [s]')

    pylab.tight_layout()
    plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    plot_path = os.path.join(plot_path, 'plots')
    plot_path = os.path.join(plot_path, 'synchrony')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    output = os.path.join(plot_path, '%s.pdf' %name)
    pylab.savefig(output)
    return numpy.mean(cd_nodiag)/numpy.mean(cc_nodiag)

class TestSynchrony(unittest.TestCase):

    def setUp(self):
        self.all_matches    = None
        self.all_templates  = None
        self.max_chunk      = '20'
        dirname             = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
        self.path           = os.path.join(dirname, 'synthetic')
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.file_name      = os.path.join(self.path, 'synchrony.raw')
        self.source_dataset = get_dataset(self)
        if not os.path.exists(self.file_name):
            mpi_launch('benchmarking', self.source_dataset, 2, 0, 'False', self.file_name, 'synchrony')

    #def tearDown(self):
    #    data_path = '.'.join(self.file_name.split('.')[:-1])
    #    shutil.rmtree(data_path)

    def test_synchrony(self):
        io.change_flag(self.file_name, 'max_chunk', self.max_chunk)
        mpi_launch('fitting', self.file_name, 2, 0, 'False')
        io.change_flag(self.file_name, 'max_chunk', 'inf')
        res = get_performance(self.file_name, 100*0.5, 'test')
        assert (numpy.abs(res - 1) < 0.75), "Synchrony not properly resolved %g" %res