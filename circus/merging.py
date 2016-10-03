from .shared.utils import *
from shared import gui
from shared.messages import init_logging, print_and_log
import pylab
from matplotlib.backends import qt_compat

use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtGui, QtCore, uic
else:
    from PyQt4 import QtGui, QtCore, uic

def main(params, nb_cpu, nb_gpu, use_gpu, extension):

    logger        = init_logging(params.logfile)
    logger        = logging.getLogger('circus.merging')
    file_out_suff = params.get('data', 'file_out_suff')
    extension_in  = extension
    extension_out = '-merged'
    
    if comm.rank == 0:
    
    	if (extension != '') and (os.path.exists(file_out_suff + '.result%s.hdf5' %extension_out)):
            key = ''
            while key not in ['y', 'n']:
                print("Export already made! Do you want to erase everything? (y)es / (n)o ")
                key = raw_input('')
                if key =='y':
                    purge(file_out_suff, extension)
                    extension_in = ''

    comm.Barrier()

    if comm.rank == 0:
        app = QtGui.QApplication([])
        try:
            pylab.style.use('ggplot')
        except Exception:
            pass
    else:
        app = None

    print_and_log(['Launching the merging GUI...'], 'debug', logger)
    mygui = gui.MergeWindow(params, app, extension_in, extension_out)
    sys.exit(app.exec_())
