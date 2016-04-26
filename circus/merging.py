from .shared.utils import *
from shared import gui
import pylab
from matplotlib.backends import qt_compat

use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtGui, QtCore, uic
else:
    from PyQt4 import QtGui, QtCore, uic

def main(filename, params, nb_cpu, nb_gpu, use_gpu, extension):


    file_out_suff = params.get('data', 'file_out_suff')
    
    #if comm.rank == 0:
    
        #if os.path.exists(file_out_suff, '*.results%d.hdf5' %extension)

        #io.purge(file_out_suff, '-merged')


    comm.Barrier()

    app = QtGui.QApplication([])
    try:
        pylab.style.use('ggplot')
    except Exception:
        pass
    mygui = gui.MergeWindow(comm, params, app, extension)
    sys.exit(app.exec_())
