from .shared.utils import *
from shared import gui
from shared.messages import init_logging, print_and_log
from circus.shared.utils import query_yes_no
import pylab

try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from matplotlib.backends import qt_compat
    use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE

    if use_pyside:
        from PySide.QtGui import QApplication
    else:
        from PyQt4.QtGui import QApplication


def main(params, nb_cpu, nb_gpu, use_gpu, extension):

    logger = init_logging(params.logfile)
    logger = logging.getLogger('circus.merging')
    file_out_suff = params.get('data', 'file_out_suff')
    extension_in = extension
    extension_out = '-merged'

    # Erase previous results (if user agrees).
    if comm.rank == 0:
        existing_file_paths = [
            file_path
            for file_path in [
                file_out_suff + ".%s%s.hdf5" % (file_id, extension_out)
                for file_id in ['templates', 'clusters', 'result']
            ]
            if os.path.isfile(file_path)
        ]
        existing_directory_path = [
            directory_path
            for directory_path in [
                file_out_suff + "%s.GUI" % extension_out
            ]
            if os.path.isdir(directory_path)
        ]
        if len(existing_file_paths) > 0 or len(existing_directory_path) > 0:
            erase = query_yes_no("Merging already done! Do you want to erase previous merging results?", default=None)
            if erase:
                for path in existing_file_paths:
                    os.remove(path)
                    if comm.rank == 0:
                        print_and_log(["Removed file %s" % path], 'debug', logger)
                for path in existing_directory_path:
                    shutil.rmtree(path)
                    if comm.rank == 0:
                        print_and_log(["Removed directory %s" % path], 'debug', logger)

    comm.Barrier()

    if comm.rank == 0 and params.getfloat('merging', 'auto_mode') == 0:
        app = QApplication([])
        try:
            pylab.style.use('ggplot')
        except Exception:
            pass
    else:
        app = None

    if comm.rank == 0:
        print_and_log(['Launching the merging GUI...'], 'debug', logger)

    _ = gui.MergeWindow(params, app, extension_in, extension_out)
    sys.exit(app.exec_())
