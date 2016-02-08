from .shared.utils import *
from shared import gui
import pylab


def main(filename, params, nb_cpu, nb_gpu, use_gpu):

    file_out_suff  = params.get('data', 'file_out_suff')
    
    if comm.rank == 0:

        io.purge(file_out_suff, '-merged')
        pylab.switch_backend('QT4Agg')
        try:
            pylab.style.use('ggplot')
        except Exception:
            pass

    mygui = gui.MergeGUI(comm, params)
    mng   = pylab.get_current_fig_manager()
    mng.window.showMaximized()
    pylab.show()