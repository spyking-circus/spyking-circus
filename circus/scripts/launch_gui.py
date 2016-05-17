import os
import sys
import subprocess
from threading  import Thread
from Queue import Queue, Empty

import pkg_resources

try:
    from PySide import QtGui, QtCore, uic
    from PySide.QtCore import Qt
    from PySide.QtGui import QApplication, QCursor, QFileDialog, QCheckBox, QWidget, QTextCursor
except ImportError:
    from PyQt4 import QtGui, QtCore, uic
    from PyQt4.QtCore import Qt
    from PyQt4.QtGui import QApplication, QCursor, QFileDialog, QCheckBox, QWidget, QTextCursor

ON_POSIX = 'posix' in sys.builtin_module_names

def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()


class LaunchGUI(QtGui.QDialog):
    def __init__(self, app):
        super(LaunchGUI, self).__init__()
        self.app = app
        self.init_gui_layout()

    def init_gui_layout(self):
        gui_fname = pkg_resources.resource_filename('circus',
                                                    os.path.join('shared',
                                                                 'qt_launcher.ui'))
        self.ui = uic.loadUi(gui_fname)
        self.task_comboboxes = [cb for cb in self.ui.grp_tasks.children()
                                if isinstance(cb, QCheckBox)]
        self.ui.btn_run.clicked.connect(self.run)
        self.ui.btn_stop.clicked.connect(self.stop)
        self.ui.btn_file.clicked.connect(self.update_data_file)
        self.ui.btn_output.clicked.connect(self.update_output_file)
        self.ui.btn_hostfile.clicked.connect(self.update_host_file)
        self.ui.cb_batch.toggled.connect(self.update_batch_mode)
        self.ui.cb_preview.toggled.connect(self.update_preview_mode)
        self.last_mode = None
        self.ui.cb_benchmarking.toggled.connect(self.update_benchmarking)
        self.ui.cb_merging.toggled.connect(self.update_extension)
        self.ui.cb_converting.toggled.connect(self.update_extension)
        self.update_benchmarking()
        self.update_extension()
        for cb in self.task_comboboxes:
            cb.toggled.connect(self.store_tasks)
        self.ui.edit_file.textChanged.connect(self.update_command)
        self.ui.edit_output.textChanged.connect(self.update_command)
        self.ui.edit_hostfile.textChanged.connect(self.update_command)
        self.ui.spin_cpus.valueChanged.connect(self.update_command)
        self.ui.spin_gpus.valueChanged.connect(self.update_command)
        self.store_tasks()
        self.process = None
        self.ui.show()

    def store_tasks(self):
        self.stored_tasks = [cb.isChecked() for cb in self.task_comboboxes]

    def restore_tasks(self):
        for cb, prev_state in zip(self.task_comboboxes,
                                  self.stored_tasks):
            cb.setEnabled(True)
            cb.setChecked(prev_state)

    def update_batch_mode(self):
        batch_mode = self.ui.cb_batch.isChecked()
        self.ui.spin_cpus.setEnabled(not batch_mode)
        self.ui.spin_gpus.setEnabled(not batch_mode)
        self.ui.edit_hostfile.setEnabled(not batch_mode)
        self.ui.btn_hostfile.setEnabled(not batch_mode)
        self.update_tasks()
        self.update_extension()
        self.update_benchmarking()
        if batch_mode:
            self.ui.cb_preview.setChecked(False)
            self.ui.lbl_file.setText('Command file')
        else:
            self.last_mode = None
            self.ui.lbl_file.setText('Data file')
        self.update_command()

    def update_preview_mode(self):
        preview_mode = self.ui.cb_preview.isChecked()

        self.update_tasks()

        if preview_mode:
            self.ui.cb_batch.setChecked(False)
        self.update_command()


    def update_extension(self):
        batch_mode = self.ui.cb_batch.isChecked()
        if (not batch_mode and (self.ui.cb_merging.isChecked() or
                                    self.ui.cb_converting.isChecked())):
            self.ui.edit_extension.setEnabled(True)
        else:
            self.ui.edit_extension.setEnabled(False)

    def update_benchmarking(self):
        batch_mode = self.ui.cb_batch.isChecked()
        enable = not batch_mode and self.ui.cb_benchmarking.isChecked()
        self.ui.edit_output.setEnabled(enable)
        self.ui.btn_output.setEnabled(enable)
        self.ui.cmb_type.setEnabled(enable)
        self.update_command()

    def update_tasks(self):
        batch_mode = self.ui.cb_batch.isChecked()
        preview_mode = self.ui.cb_preview.isChecked()
        if batch_mode:
            self.restore_tasks()
            for cb in self.task_comboboxes:
                cb.setEnabled(False)
        elif preview_mode:
            prev_stored_tasks = self.stored_tasks
            for cb in self.task_comboboxes:
                cb.setEnabled(False)
                cb.setChecked(False)
            self.ui.cb_filtering.setChecked(True)
            self.ui.cb_whitening.setChecked(True)
            self.stored_tasks = prev_stored_tasks
        else:  # We come back from batch or preview mode
            self.restore_tasks()
        self.update_command()

    def update_data_file(self):
        if self.ui.cb_batch.isChecked():
            title = 'Select file with list of commands'
        else:
            title = 'Select data file'
        fname = QFileDialog.getOpenFileName(self, title,
                                            self.ui.edit_file.text())
        if fname:
                self.ui.edit_file.setText(fname)

    def update_host_file(self):
        fname = QFileDialog.getOpenFileName(self, 'Select MPI host file',
                                            self.ui.edit_hostfile.text())
        if fname:
                self.ui.edit_hostfile.setText(fname)

    def update_output_file(self):
        fname = QFileDialog.getSaveFileName(self, 'Output file name',
                                            self.ui.edit_output.text())
        if fname:
            self.ui.edit_output.setText(fname)

    def command_line_args(self):
        batch_mode = self.ui.cb_batch.isChecked()
        preview_mode = self.ui.cb_preview.isChecked()

        args = ['spyking-circus', str(self.ui.edit_file.text())]

        if batch_mode:
            args.append('--batch')
        elif preview_mode:
            args.append('--preview')
        else:
            tasks = []
            results = False
            for cb in self.task_comboboxes:
                if cb.isChecked():
                    label = str(cb.text()).lower()
                    if label == 'results':
                        results = True
                    else:
                        tasks.append(label)
            args.extend(['--method', ','.join(tasks)])
            if results:
                args.append('--result')
            args.extend(['--cpu', str(self.ui.spin_cpus.value())])
            args.extend(['--gpu', str(self.ui.spin_gpus.value())])
            if 'merging' in tasks or 'converting' in tasks:
                extension = str(self.ui.edit_extension.text()).strip()
                if extension:
                    args.extend(['--extension', extension])
            if 'benchmarking' in tasks:
                args.extend(['--output', str(self.ui.edit_output.text())])
                args.extend(['--type', str(self.ui.cmb_type.currentText())])

        return args

    def update_command(self, text=None):
        args = ' '.join(self.command_line_args())
        self.ui.edit_command.setPlainText(args)

    def run(self):
        args = self.command_line_args()
        # Disable everything except for the stop button and the output area
        previous_state = {}
        all_children = self.ui.findChildren(QWidget)
        previous_state = {obj: obj.isEnabled() for obj in all_children}
        for obj in all_children:
            obj.setEnabled(False)
            obj.repaint()
        for widget in [self.ui.btn_stop, self.ui.edit_stdout]:
            widget.setEnabled(True)
            widget.parent().setEnabled(True)
            widget.parent().repaint()
        self.ui.repaint()

        # Start process
        self.app.setOverrideCursor(Qt.WaitCursor)

        try:
            self.ui.edit_stdout.setPlainText('')
            self.process = subprocess.Popen(args, stdout=subprocess.PIPE,
                                            bufsize=1, close_fds=ON_POSIX)
            q = Queue()
            t = Thread(target=enqueue_output, args=(self.process.stdout, q))
            t.start()

            while t.isAlive():
                try:
                    line = q.get(timeout=0.1)
                except Empty:
                    pass
                else:
                    self.ui.edit_stdout.append(line.rstrip())
                    # new_text = self.ui.edit_stdout.toPlainText() + line
                    # self.ui.edit_stdout.setPlainText(new_text)
                    # text_cursor = self.ui.edit_stdout.textCursor()
                    # text_cursor.movePosition(QTextCursor.End)
                    # self.ui.edit_stdout.setTextCursor(text_cursor)

                self.app.processEvents()
        except Exception:
            import traceback
            self.ui.edit_stdout.append('<pre style="color: red">'+traceback.format_exc()+'</pre>')
        finally:
            # Done
            for obj, state in previous_state.iteritems():
                obj.setEnabled(state)
            self.app.restoreOverrideCursor()
            self.process = None

    def stop(self):
        print 'stopping'
        if self.process is not None:
            self.process.kill()
            new_text = self.ui.edit_stdout.toHtml()
            new_text += '<pre style="color: red">Interrupted by the user</pre>'
            self.ui.edit_stdout.setHtml(new_text)


def main():
    app = QtGui.QApplication([])
    gui = LaunchGUI(app)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()