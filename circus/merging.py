from .shared.utils import *
from shared import gui
import pylab
from matplotlib.backends import qt_compat

use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtGui, QtCore, uic
else:
    from PyQt4 import QtGui, QtCore, uic

def main(filename, params, nb_cpu, nb_gpu, use_gpu):

    file_out_suff  = params.get('data', 'file_out_suff')
    
    if comm.rank == 0:

        io.purge(file_out_suff, '-merged')

    app = QtGui.QApplication([])
    pylab.style.use('ggplot')
    mygui = gui.MergeWindow(comm, params)
    sys.exit(app.exec_())
