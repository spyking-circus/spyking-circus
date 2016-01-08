from .shared.utils import *
from shared import gui
import pylab


def main(filename, params, nb_cpu, nb_gpu, use_gpu):

    import h5py
    parallel_hdf5 = h5py.get_config().mpi

    #################################################################
    sampling_rate  = params.getint('data', 'sampling_rate')
    N_e            = params.getint('data', 'N_e')
    N_t            = params.getint('data', 'N_t')
    N_total        = params.getint('data', 'N_total')
    file_out_suff  = params.get('data', 'file_out_suff')
    file_out       = params.get('data', 'file_out')
    cc_gap         = params.getfloat('merging', 'cc_gap')
    cc_overlap     = params.getfloat('merging', 'cc_overlap')
    cc_bin         = params.getfloat('merging', 'cc_bin')
    cc_average     = params.getfloat('merging', 'cc_average')
    make_plots     = params.getboolean('merging', 'make_plots')
    plot_path      = os.path.join(params.get('data', 'data_file_noext'), 'plots')
    
    bin_size       = int(cc_bin * sampling_rate * 1e-3)
    delay_average  = int(cc_average/cc_bin)
    max_delay      = max(50, cc_average)

    if comm.rank == 0:

        io.purge(file_out_suff, '-merged')
        if make_plots:
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            io.purge(plot_path, 'merging')
    
        pylab.switch_backend('QT4Agg')
        pylab.style.use('ggplot')

        mygui = gui.MergeGUI(comm, params)
        mng   = pylab.get_current_fig_manager()
        mng.window.showMaximized()
        pylab.show()