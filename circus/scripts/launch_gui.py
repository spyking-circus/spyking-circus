import datetime
import os
import psutil
import shutil
import sys
import textwrap

import pkg_resources

try:
    from PySide import QtGui, QtCore, uic
    from PySide.QtCore import Qt, QUrl, QProcess, SIGNAL
    from PySide.QtGui import (QApplication, QCursor, QFileDialog, QCheckBox,
                              QPushButton, QLineEdit, QPixmap,
                              QWidget, QTextCursor, QMessageBox, QPalette, QBrush,
                              QDesktopServices, QFont)
except ImportError:
    from PyQt4 import QtGui, QtCore, uic
    from PyQt4.QtCore import Qt, QUrl, QProcess, SIGNAL
    from PyQt4.QtGui import (QApplication, QCursor, QFileDialog, QCheckBox,
                             QPushButton, QLineEdit, QPalette, QBrush,
                             QWidget, QTextCursor, QMessageBox, QPixmap,
                             QDesktopServices, QFont)


def overwrite_text(cursor, text):
    text_length = len(text)
    cursor.clearSelection()
    # Select the text after the current position (if any)
    current_position = cursor.position()
    cursor.movePosition(QTextCursor.Right,
                        mode=QTextCursor.MoveAnchor,
                        n=text_length)
    cursor.movePosition(QTextCursor.Left,
                        mode=QTextCursor.KeepAnchor,
                        n=cursor.position()-current_position)
    # Insert the text (will overwrite the selected text)
    cursor.insertText(text)


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

        #palette = QPalette()
        #palette.setBrush(QPalette.Background, QBrush(QPixmap("/home/pierre/github/spyking-circus/circus/icons/logo.jpg")))
        self.ui.btn_run.clicked.connect(self.run)
        self.ui.btn_stop.clicked.connect(self.stop)
        self.ui.btn_file.clicked.connect(self.update_data_file)
        self.ui.btn_about.clicked.connect(self.show_about)
        self.ui.btn_output.clicked.connect(self.update_output_file)
        self.ui.btn_hostfile.clicked.connect(self.update_host_file)
        self.ui.btn_log.clicked.connect(self.open_log_file)
        self.ui.cb_batch.toggled.connect(self.update_batch_mode)
        self.ui.cb_preview.toggled.connect(self.update_preview_mode)
        self.ui.cb_results.toggled.connect(self.update_results_mode)
        self.ui.cb_benchmarking.toggled.connect(self.update_benchmarking)
        self.ui.cb_merging.toggled.connect(self.update_extension)
        self.ui.cb_converting.toggled.connect(self.update_extension)
        self.update_benchmarking()
        self.update_extension()
        for cb in self.task_comboboxes:
            cb.toggled.connect(self.store_tasks)
            cb.toggled.connect(self.update_command)
        self.ui.edit_file.textChanged.connect(self.update_command)
        self.ui.edit_output.textChanged.connect(self.update_command)
        self.ui.edit_hostfile.textChanged.connect(self.update_command)
        self.ui.edit_extension.textChanged.connect(self.update_command)
        self.ui.spin_cpus.valueChanged.connect(self.update_command)
        self.ui.spin_gpus.valueChanged.connect(self.update_command)
        self.store_tasks()
        self.process = None
        self.ui.closeEvent = self.closeEvent
        self.last_log_file = None
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
            self.ui.spin_cpus.setEnabled(not batch_mode)
            self.ui.spin_gpus.setEnabled(not batch_mode)
            self.ui.edit_hostfile.setEnabled(not batch_mode)
            self.ui.btn_hostfile.setEnabled(not batch_mode)
            self.update_tasks()
            self.update_extension()
            self.update_benchmarking()
            self.update_command()
            self.ui.cb_preview.setChecked(False)
            self.ui.cb_results.setChecked(False)
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
            self.ui.cb_results.setChecked(False)

        self.update_command()

    def update_results_mode(self):
        results_mode = self.ui.cb_results.isChecked()
        self.ui.spin_cpus.setEnabled(not results_mode)
        self.ui.spin_gpus.setEnabled(not results_mode)
        self.ui.edit_hostfile.setEnabled(not results_mode)
        self.ui.btn_hostfile.setEnabled(not results_mode)
        self.update_tasks()
        self.update_extension()
        self.update_benchmarking()
        self.update_command()
        if results_mode:
            self.ui.cb_batch.setChecked(False)
            self.ui.cb_preview.setChecked(False)

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
        results_mode = self.ui.cb_results.isChecked()
        if batch_mode or results_mode:
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
        
        if str(self.ui.edit_file.text()) != '':
            self.ui.btn_run.setEnabled(True)
        else:
            self.ui.btn_run.setEnabled(False)

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

    def open_log_file(self):
        assert self.last_log_file is not None
        QDesktopServices.openUrl(QUrl(self.last_log_file))

    def command_line_args(self):
        batch_mode = self.ui.cb_batch.isChecked()
        preview_mode = self.ui.cb_preview.isChecked()
        results_mode = self.ui.cb_results.isChecked()

        args = ['spyking-circus']
        fname = str(self.ui.edit_file.text()).strip()
        if fname:
            args.append(fname)

        if batch_mode:
            args.append('--batch')
        elif preview_mode:
            args.append('--preview')
        elif results_mode:
            args.append('--result')
        else:
            tasks = []
            for cb in self.task_comboboxes:
                if cb.isChecked():
                    label = str(cb.text()).lower()
                    tasks.append(label)
            args.extend(['--method', ','.join(tasks)])
            args.extend(['--cpu', str(self.ui.spin_cpus.value())])
            args.extend(['--gpu', str(self.ui.spin_gpus.value())])
            hostfile = str(self.ui.edit_hostfile.text()).strip()
            if hostfile:
                args.extend(['--hostfile', hostfile])
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
        if self.ui.cb_batch.isChecked():
            self.last_log_file = None
        else:
            f_next, _ = os.path.splitext(str(self.ui.edit_file.text()))        
            f_params = f_next + '.params'
            if not os.path.exists(f_params):
                self.create_params_file(f_params)
                return
            self.last_log_file = f_next + '.log'
        args = self.command_line_args()

        

        # # Start process
        self.ui.edit_stdout.clear()
        format = self.ui.edit_stdout.currentCharFormat()
        format.setFontWeight(QFont.Normal)
        format.setForeground(QtCore.Qt.blue)
        self.ui.edit_stdout.setCurrentCharFormat(format)
        time_str = datetime.datetime.now().ctime()
        start_msg = '''\
                       Starting spyking circus at {time_str}.

                       Command line call:
                       {call}
                    '''.format(time_str=time_str, call=' '.join(args))
        self.ui.edit_stdout.appendPlainText(textwrap.dedent(start_msg))
        format.setForeground(QtCore.Qt.black)
        self.ui.edit_stdout.setCurrentCharFormat(format)
        self.ui.edit_stdout.appendPlainText('\n')

        self.process = QProcess(self)
        self.connect(self.process, SIGNAL('readyReadStandardOutput()'), self.append_output)
        self.connect(self.process, SIGNAL('readyReadStandardError()'),
                     self.append_error)
        self.connect(self.process, SIGNAL('started()'),
                     self.process_started)
        self.connect(self.process, SIGNAL('finished(int)'),
                     self.process_finished)
        self.connect(self.process, SIGNAL('error()'),
                     self.process_errored)
        self._interrupted = False
        self.process.start(args[0], args[1:])

    def restore_gui(self):
        for widget, previous_state in self._previous_state:
            widget.setEnabled(previous_state)
        self.app.restoreOverrideCursor()

    def process_started(self):
        # Disable everything except for the stop button and the output area
        all_children = [obj for obj in self.ui.findChildren(QWidget)
                        if isinstance(obj, (QCheckBox, QPushButton, QLineEdit))]
        self._previous_state = [(obj, obj.isEnabled()) for obj in all_children]
        for obj in all_children:
            obj.setEnabled(False)
        self.ui.btn_stop.setEnabled(True)
        # If we let the user interact, this messes with the cursor we use to
        # support the progress bar display
        self.ui.edit_stdout.setTextInteractionFlags(Qt.NoTextInteraction)
        self.app.setOverrideCursor(Qt.WaitCursor)

    def process_finished(self, exit_code):
        format = self.ui.edit_stdout.currentCharFormat()
        format.setFontWeight(QFont.Bold)
        if exit_code == 0:
            if self._interrupted:
                color = QtCore.Qt.red
                msg = ('Process interrupted by user')
            else:
                color = QtCore.Qt.green
                msg = ('Process exited normally')
        else:
            color = QtCore.Qt.red
            msg = ('Process exited with exit code %d' % exit_code)
        format.setForeground(color)
        self.ui.edit_stdout.setCurrentCharFormat(format)
        self.ui.edit_stdout.appendPlainText(msg)
        self.restore_gui()
        self.ui.edit_stdout.setTextInteractionFlags(Qt.TextSelectableByMouse |
                                                    Qt.TextSelectableByKeyboard)
        self.process = None
        self.ui.btn_log.setEnabled(self.last_log_file is not None and
                                   os.path.isfile(self.last_log_file))

    def process_errored(self):
        exit_code = self.process.exitCode()
        format = self.ui.edit_stdout.currentCharFormat()
        format.setFontWeight(QFont.Bold)
        format.setForeground(QtCore.Qt.red)
        self.ui.edit_stdout.setCurrentCharFormat(format)
        self.ui.edit_stdout.appendPlainText('Process exited with exit code' % exit_code)
        self.restore_gui()
        self.ui.edit_stdout.setTextInteractionFlags(Qt.TextSelectableByMouse |
                                                    Qt.TextSelectableByKeyboard)
        self.process = None

    def add_output_lines(self, lines):
        '''
        Add the output line by line to the text area, jumping back to the
        beginning of the line when we encounter a carriage return (to
        correctly display progress bars)
        '''
        cursor = self.ui.edit_stdout.textCursor()
        cursor.clearSelection()
        splitted_lines = lines.split('\n')
        for idx, line in enumerate(splitted_lines):
            if '\r' in line:
                chunks = line.split('\r')
                overwrite_text(cursor, chunks[0])  # No going back for the first chunk
                for chunk in chunks[1:]:  # Go back to start of line for each \r
                    cursor.movePosition(QTextCursor.StartOfLine)
                    overwrite_text(cursor, chunk)
            else:
                overwrite_text(cursor, line)

            # Take care to not introduce new newlines
            if '\n' in lines and (idx == 0 or
                                          idx != len(splitted_lines) - 1):
                cursor.movePosition(QTextCursor.EndOfLine)
                cursor.insertText('\n')
        self.ui.edit_stdout.setTextCursor(cursor)

    def append_output(self):
        if self.process is None:  # Can happen when manually interrupting
            return
        lines = str(self.process.readAllStandardOutput())
        self.add_output_lines(lines)

    def append_error(self):
        if self.process is None:  # Can happen when manually interrupting
            return
        lines = str(self.process.readAllStandardError())
        self.add_output_lines(lines)

    def stop(self, force=False):
        if self.process is not None:

            if not force:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle('Confirm process termination')
                msg.setText(
                    'This will terminate the running process. Are you sure '
                    'you want to do this?')
                msg.setInformativeText(
                    'Interrupting the process may leave partly '
                    'created files that cannot be used for '
                    'further analysis.')
                msg.addButton('Stop process', QMessageBox.YesRole)
                cancel_button = msg.addButton('Cancel', QMessageBox.NoRole)
                msg.setDefaultButton(cancel_button)
                msg.exec_()
                if msg.clickedButton() == cancel_button:
                    # Continue to run
                    return

            self._interrupted = True
            # Terminate child processes as well
            pid = self.process.pid()
            if isinstance(pid, int) and pid != 0:
                process = psutil.Process(pid)
                for proc in process.children(recursive=True):
                    proc.terminate()
            self.process.terminate()
            self.process = None

    def create_params_file(self, fname):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText("Parameter file %r not found, do you want SpyKING CIRCUS to "
                    "create it for you?" % fname)
        msg.setWindowTitle("Generate parameter file?")
        msg.setInformativeText("This will create a parameter file from a "
                               "template file and open it in your system's "
                               "standard text editor. Fill properly before "
                               "launching the code. See the documentation "
                               "for details")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        answer = msg.exec_()
        if answer == QMessageBox.Yes:
            user_path = os.path.join(os.path.expanduser('~'), 'spyking-circus')
            if os.path.exists(user_path + 'config.params'):
                config_file = os.path.abspath(user_path + 'config.params')
            else:
                config_file = os.path.abspath(
                    pkg_resources.resource_filename('circus', 'config.params'))
            shutil.copyfile(config_file, fname)
            QDesktopServices.openUrl(QUrl(fname))

    def show_about(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText("SpyKING CIRCUS v0.4 "
                    "written by P. Yger and O. Marre")
        msg.setWindowTitle("About")
        msg.setInformativeText("Documentation can be found at\n"
                                "http://spyking-circus.rtfd.org\n"
                                "\n"
                                "Open a browser to see the online help?"
            )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        answer = msg.exec_()
        if answer == QMessageBox.Yes:
            QDesktopServices.openUrl(QUrl("http://spyking-circus.rtfd.org"))

    def closeEvent(self, event):
        if self.process is not None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle('Confirm process interruption')
            msg.setText('Closing the window will terminate the running process. '
                        'Do you really want to exit?')
            msg.setInformativeText('Interrupting the process may leave partly '
                                   'created files that cannot be used for '
                                   'further analysis.')
            close_button = msg.addButton("Stop and close", QMessageBox.YesRole)
            cancel_button = msg.addButton("Cancel", QMessageBox.NoRole)
            msg.setDefaultButton(cancel_button)
            msg.exec_()
            if msg.clickedButton() == close_button:
                self.stop(force=True)
                super(LaunchGUI, self).closeEvent(event)
            else:
                event.ignore()


def main():
    app = QtGui.QApplication([])
    gui = LaunchGUI(app)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()