from .shared.utils import *


def main(filename, params, nb_cpu, use_gpu):

    templates     = io.load_data(params, 'templates')
    thresholds    = io.load_data(params, 'thresholds')
    results       = io.load_data(params, 'results')
    file_out_suff = params.get('data', 'file_out_suff')
    n_templates   = templates.shape[2]/2

    r             = []
    n_total       = 0
    n_removed     = 0

    for i in xrange(n_templates):
        gmin       = templates[:,:,i].min()
        data       = numpy.where(templates[:,:,i] == gmin)
        idx        = numpy.where(results['amplitudes']['temp_'+str(i)][:, 0]*gmin < -0.5*thresholds[data[0][0]])[0]
        n_removed += len(idx)
        n_total   += len(results['amplitudes']['temp_'+str(i)])
        results['spiketimes']['temp_' + str(i)] = results['spiketimes']['temp_' + str(i)][idx]
        results['amplitudes']['temp_' + str(i)] = results['amplitudes']['temp_' + str(i)][idx]

    keys = ['spiketimes', 'amplitudes']
    for key in keys:
        if os.path.exists(file_out_suff + '.%s-merged.mat' %key):
            io.purge(file_out_suff, '.%s-merged.mat' %key)
        hdf5storage.savemat(file_out_suff + '.%s-merged' %key, results[key])

    print n_removed/float(n_total)
