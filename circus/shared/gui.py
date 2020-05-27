# -*- coding: utf-8 -*-
from __future__ import division

import six
import pkg_resources
import logging
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

import numpy as np

try:
    from PyQt5 import uic
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QCursor
    from PyQt5.QtWidgets import QMainWindow
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.backends import qt_compat
    use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE
    if use_pyside:
        from PySide import uic
        from PySide.QtCore import Qt
        from PySide.QtGui import QCursor, QMainWindow
    else:
        from PyQt4 import uic
        from PyQt4.QtCore import Qt
        from PyQt4.QtGui import QCursor, QMainWindow

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.colors import colorConverter

from distutils.version import LooseVersion, StrictVersion
MPL_VERSION = StrictVersion(mpl.__version__) > StrictVersion("2.2.1")

from circus.shared.utils import *
import circus.shared.files as io
from circus.shared.algorithms import slice_templates, slice_clusters
from circus.shared.mpi import comm
from circus.shared.probes import get_nodes_and_edges
from circus.shared.messages import print_and_log
from circus.shared.utils import apply_patch_for_similarities, get_shared_memory_flag, bhatta_dist, test_if_support


logger = logging.getLogger(__name__)


class SymmetricVCursor(widgets.AxesWidget):
    """Variant of matplotlib.widgets.Cursor, drawing two symmetric vertical lines at -x and x"""
    def __init__(self, ax, useblit=False, **lineprops):
        """
        Add a cursor to *ax*.  If ``useblit=True``, use the backend-
        dependent blitting features for faster updates (GTKAgg
        only for now).  *lineprops* is a dictionary of line properties.
        """
        # TODO: Is the GTKAgg limitation still true?
        widgets.AxesWidget.__init__(self, ax)

        self.connect_event('motion_notify_event', self.onmove)
        self.connect_event('draw_event', self.clear)

        self.visible = True
        self.useblit = useblit and self.canvas.supports_blit

        if self.useblit:
            lineprops['animated'] = True
        self.linev1 = ax.axvline(ax.get_ybound()[0], visible=False, **lineprops)
        self.linev2 = ax.axvline(ax.get_xbound()[0], visible=False, **lineprops)

        self.background = None
        self.needclear = False

    def clear(self, event):
        """clear the cursor"""
        if self.ignore(event):
            return
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.linev1.set_visible(False)
        self.linev2.set_visible(False)

    def onmove(self, event):
        """on mouse motion draw the cursor if visible"""
        if self.ignore(event):
            return
        if not self.canvas.widgetlock.available(self):
            return
        if event.inaxes != self.ax:
            self.linev1.set_visible(False)
            self.linev2.set_visible(False)

            if self.needclear:
                self.canvas.draw()
                self.needclear = False
            return
        self.needclear = True
        if not self.visible:
            return
        self.linev1.set_xdata((event.xdata, event.xdata))
        self.linev2.set_xdata((-event.xdata, -event.xdata))

        self.linev1.set_visible(self.visible)
        self.linev2.set_visible(self.visible)

        self._update()

    def _update(self):

        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.linev1)
            self.ax.draw_artist(self.linev2)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

        return False


class MergeWindow(QMainWindow):

    def __init__(self, params, app, extension_in='', extension_out='-merged'):

        if app is not None:
            if comm.rank == 0:
                super(MergeWindow, self).__init__()

        if comm.rank == 0 and app is not None:
            print_and_log(["Loading GUI with %d CPUs..." % comm.size], 'default', logger)

        apply_patch_for_similarities(params, extension_in)

        self.app = app
        self.params = params
        self.ext_in = extension_in
        self.ext_out = extension_out
        data_file = params.data_file
        self.N_e = params.getint('data', 'N_e')
        self.nb_templates = int(io.load_data(params, 'nb_templates') // 2)
        self.N_t = params.getint('detection', 'N_t')
        self.N_total = params.nb_channels
        self.sampling_rate = params.rate
        self.correct_lag = params.getboolean('merging', 'correct_lag')
        self.file_out_suff = params.get('data', 'file_out_suff')
        self.cc_overlap = params.getfloat('merging', 'cc_overlap')
        self.cc_bin = params.getfloat('merging', 'cc_bin')
        self.auto_mode = params.getfloat('merging', 'auto_mode')
        self.merge_drifts = params.getboolean('merging', 'merge_drifts')
        self.drift_limit = params.getfloat('merging', 'drift_limit')
        self.default_lag = params.getfloat('merging', 'default_lag')
        self.remove_noise = params.getboolean('merging', 'remove_noise')
        self.noise_limit = params.getfloat('merging', 'noise_limit')
        self.sparsity_limit = params.getfloat('merging', 'sparsity_limit')
        self.min_spikes = params.getint('merging', 'min_spikes')
        self.adapted_cc = params.getboolean('clustering', 'adapted_cc')
        self.adapted_thr = params.getint('clustering', 'adapted_thr')

        self.duration = io.load_data(params, 'duration')
        self.nb_bhatta_bins = 100
        self.bhattas_level = ((self.nb_bhatta_bins - 1)/self.duration)/self.nb_bhatta_bins
        self.has_support = test_if_support(params, self.ext_in)
        if self.has_support:
            self.supports = io.load_data(params, 'supports', self.ext_in)
            self.mean_channels = numpy.mean(numpy.sum(self.supports, 1))
            if self.mean_channels < 3:
                self.low_channel_count = True
                if comm.rank == 0:
                    print_and_log(["Templates on few channels only, taking norm into account"], 'debug', logger)
            else:
                self.low_channel_count = False
        self.bin_size = int(self.cc_bin * self.sampling_rate * 1e-3)
        self.max_delay = 50
        self.time_rpv = params.getfloat('merging', 'time_rpv')
        self.rpv_threshold = params.getfloat('merging', 'rpv_threshold')
        self.result = io.load_data(params, 'results', self.ext_in)

        self.nb_bins = int(self.duration/(self.cc_bin*1e-3))

        self.overlap = h5py.File(self.file_out_suff + '.templates%s.hdf5' % self.ext_in, libver='earliest', mode='r').get('maxoverlap')[:]
        try:
            self.lag = h5py.File(self.file_out_suff + '.templates%s.hdf5' % self.ext_in, libver='earliest', mode='r').get('maxlag')[:]
        except Exception:
            self.lag = numpy.zeros(self.overlap.shape, dtype=numpy.int32)
        self.shape = h5py.File(self.file_out_suff + '.templates%s.hdf5' % self.ext_in, libver='earliest', mode='r').get('temp_shape')[:]
        self.electrodes = io.load_data(params, 'electrodes', self.ext_in)

        self.SHARED_MEMORY = get_shared_memory_flag(params)

        if self.SHARED_MEMORY:
            self.templates, self.mpi_memory_1 = io.load_data_memshared(params, 'templates', extension=self.ext_in)
            self.clusters, self.mpi_memory_2 = io.load_data_memshared(params, 'clusters-light', extension=self.ext_in)
        else:
            self.templates = io.load_data(params, 'templates', self.ext_in)
            self.clusters = io.load_data(params, 'clusters-light', self.ext_in)

        self.thresholds = io.load_data(params, 'thresholds')
        self.indices = numpy.arange(self.shape[2] // 2)
        nodes, edges = get_nodes_and_edges(params)
        self.inv_nodes = numpy.zeros(self.N_total, dtype=numpy.int32)
        self.inv_nodes[nodes] = numpy.arange(len(nodes))
        self.nodes = nodes
        self.edges = edges
        self.sparsities = numpy.zeros(self.shape[2] // 2, dtype=numpy.float32)

        self.norms = numpy.zeros(len(self.indices), dtype=numpy.float32)
        self.rates = numpy.zeros(len(self.indices), dtype=numpy.float32)
        self.to_delete = numpy.zeros(0, dtype=numpy.int32)

        self.sign_peaks = params.get('detection', 'peaks')
        for idx in self.indices:
            tmp = self.templates[:, idx]
            tmp = tmp.toarray().reshape(self.N_e, self.N_t)
            self.rates[idx] = len(self.result['spiketimes']['temp_' + str(idx)]) / self.duration
            if self.sign_peaks == 'negative':
                elec = numpy.argmin(numpy.min(tmp, 1))
                thr = self.thresholds[elec]
                if thr > 0:
                    self.norms[idx] = -tmp.min()/thr
            elif self.sign_peaks == 'positive':
                elec = numpy.argmax(numpy.max(tmp, 1))
                thr = self.thresholds[elec]
                if thr > 0:
                    self.norms[idx] = tmp.max()/thr
            elif self.sign_peaks == 'both':
                elec = numpy.argmax(numpy.max(numpy.abs(tmp), 1))
                thr = self.thresholds[elec]
                if thr > 0:
                    self.norms[idx] = numpy.abs(tmp).max()/thr

            if self.has_support:
                support = numpy.where(self.supports[idx])[0]
                nb_channels = len(support)
            else:
                best_elec = self.nodes[self.electrodes[idx]]
                nb_channels = len(numpy.where(tmp.sum(1) != 0)[0])
                max_nb_channels = len(self.edges[best_elec])
            if self.N_e == 1:
                self.sparsities[idx] = 1.0
            else:
                self.sparsities[idx] = 1.0 - float(nb_channels - 1) / float(self.N_e - 1)

        self.overlap /= self.shape[0] * self.shape[1]
        self.all_merges = numpy.zeros((0, 2), dtype=numpy.int32)
        self.mpi_wait = numpy.array([0], dtype=numpy.int32)

        if self.adapted_cc:
            common_supports = io.load_data(params, 'common-supports')
            self.exponents = numpy.exp(-common_supports/self.adapted_thr)

        if comm.rank > 0:
            self.listen()

        if self.app is not None:
            self.init_gui_layout()

        self.probe = self.params.probe
        self.x_position = []
        self.y_position = []
        self.label = []
        self.order = []
        for key in self.probe['channel_groups'].keys():
            for item in self.probe['channel_groups'][key]['geometry'].keys():
                if item in self.probe['channel_groups'][key]['channels']:
                    self.x_position += [self.probe['channel_groups'][key]['geometry'][item][0]]
                    self.y_position += [self.probe['channel_groups'][key]['geometry'][item][1]]
                    if len(self.probe['channel_groups'].keys()) == 1:
                        self.label += ["%d" % item]
                    else:
                        self.label += ["%s-%d" % (key, item)]

        if self.auto_mode > 0:
            print_and_log(['Automatic merging with a threshold of %g' % self.auto_mode], 'info', logger)

        self.generate_data()

        if len(self.raw_data) == 0:
            print_and_log(['No pairs were found given the parameters'], 'info', logger)
            self.finalize(None)

        self.selected_points = set()
        self.selected_templates = set()
        self.inspect_templates = []
        self.inspect_colors_templates = []
        self.inspect_points = []
        self.inspect_colors = []
        self.lasso_selector = None

        self.suggest_value_template = self.noise_limit
        self.suggest_value_bhatta = self.drift_limit

        if self.app is not None:
            self.suggest_value = self.get_suggest_value.value()
        else:
            self.suggest_value = self.auto_mode

        if self.app is not None:
            # Update suggested values visible in the GUI.
            self.get_suggest_value_template.setValue(self.suggest_value_template)
            self.get_suggest_value_bhatta.setValue(self.suggest_value_bhatta)
            self.get_suggest_value.setValue(self.suggest_value)
            self.get_sparsity_limit.setValue(self.sparsity_limit)

            self.rect_selectors = [
                widgets.RectangleSelector(ax, onselect=self.callback_rect, button=1, drawtype='box', spancoords='data')
                for ax in [self.score_ax1, self.score_ax2, self.score_ax3]
            ]
            for selector in self.rect_selectors:
                selector.set_active(False)

            self.lag_selector = SymmetricVCursor(self.data_ax, color='blue')
    
            self.lag_selector.active = False
            self.line_lag1 = self.data_ax.axvline(self.data_ax.get_ybound()[0],
                                                  color='black')
            self.line_lag2 = self.data_ax.axvline(self.data_ax.get_ybound()[0],
                                                  color='black')

            self.update_lag(self.default_lag)
            
            self.plot_data()
            self.plot_scores()

            # # Connect matplotlib events
            for fig in [self.ui.score_1, self.ui.score_2, self.ui.score_3,
                        self.ui.detail, self.ui.data_overview, self.ui.waveforms, self.ui.detail_2]:
                fig.mpl_connect('scroll_event', self.zoom)
                fig.mpl_connect('button_press_event', self.on_mouse_press)

            # self.fig.canvas.mpl_connect('close_event', self.handle_close)
            self.ui.btn_rectangle.clicked.connect(self.update_rect_selector)
            self.ui.btn_lasso.clicked.connect(self.update_rect_selector)
            self.ui.btn_picker.clicked.connect(self.update_rect_selector)
            self.ui.btn_select.clicked.connect(self.add_to_selection)
            self.ui.btn_unselect.clicked.connect(self.remove_selection)
            self.ui.btn_delete.clicked.connect(self.remove_templates)
            self.ui.btn_suggest_templates.clicked.connect(self.suggest_templates)
            self.ui.btn_suggest_pairs.clicked.connect(self.suggest_pairs)
            self.ui.btn_suggest_drifts.clicked.connect(self.suggest_drifts)
            self.ui.get_suggest_value.valueChanged.connect(self.update_suggest_value)
            self.ui.get_suggest_value_template.valueChanged.connect(self.update_suggest_value_template)
            self.ui.get_sparsity_limit.valueChanged.connect(self.update_sparsity_limit)
            self.ui.get_suggest_value_bhatta.valueChanged.connect(self.update_suggest_value_bhatta)
            self.ui.btn_unselect_template.clicked.connect(self.remove_selection_templates)

            self.ui.cmb_sorting.currentIndexChanged.connect(self.update_data_sort_order)
            self.ui.btn_merge.clicked.connect(self.do_merge)
            self.ui.btn_finalize.clicked.connect(self.finalize)
            self.ui.btn_merge_and_finalize.clicked.connect(self.merge_and_finalize)
            self.ui.btn_set_lag.clicked.connect(lambda event: setattr(self.lag_selector,
                                                                      'active', True))
            self.ui.show_peaks.clicked.connect(self.update_waveforms)
            self.ui.show_labels.clicked.connect(self.update_waveforms)
            self.ui.show_thresholds.clicked.connect(self.update_waveforms)

            # TODO: Tooltips
            # self.electrode_ax.format_coord = lambda x, y: 'template similarity: %.2f  cross-correlation metric %.2f' % (x, y)
            # self.score_ax2.format_coord = lambda x, y: 'normalized cross-correlation metric: %.2f  cross-correlation metric %.2f' % (x, y)
            # self.score_ax3.format_coord = lambda x, y: 'template similarity: %.2f  normalized cross-correlation metric %.2f' % (x, y)
            # self.data_ax.format_coord = self.data_tooltip
            # Select the best point at start
            idx = np.argmax(self.score_y)
            self.update_inspect({idx})

        if self.remove_noise:
            self.update_lag(self.default_lag)
            self.suggest_templates(None)
            self.remove_templates(None)

        if self.auto_mode > 0:
            perform_merges = True
            self.update_lag(self.default_lag)
            self.current_order = 0

            self.suggest_pairs(None)
            self.add_to_selection(None)
            self.do_merge(None)

            if self.merge_drifts:
                self.suggest_drifts(None)
                self.add_to_selection(None)

                self.do_merge(None)

            self.finalize(None)

    def listen(self):

        while self.mpi_wait[0] == 0:
            self.generate_data()

        if self.mpi_wait[0] == 1:
            self.finalize(None)
        elif self.mpi_wait[0] == 2:
            
            if self.SHARED_MEMORY:
                for memory in self.mpi_memory_1 + self.mpi_memory_2:
                    memory.Free()
            sys.exit(0)


    def update_suggest_value(self):
        self.suggest_value = self.get_suggest_value.value()
        if self.app is not None:
            self.decision_boundary.set_xdata([0, 1])
            self.decision_boundary.set_ydata([self.suggest_value, self.suggest_value])
            self.ui.score_1.draw_idle()

    def update_suggest_value_template(self):
        self.suggest_value_template = self.get_suggest_value_template.value()
        if self.app is not None:
            self.decision_boundary2.set_xdata([self.suggest_value_template, self.suggest_value_template])
            self.decision_boundary2.set_ydata([0, self.rates[self.to_consider].max()])
            self.ui.score_2.draw_idle()

    def update_sparsity_limit(self):
        self.sparsity_limit = self.get_sparsity_limit.value()

    def update_suggest_value_bhatta(self):
        self.suggest_value_bhatta = self.get_suggest_value_bhatta.value()
        if self.app is not None:
            self.decision_boundary3.set_xdata([0, 1])
            self.decision_boundary3.set_ydata([self.suggest_value_bhatta, self.suggest_value_bhatta])
            self.ui.score_3.draw_idle()

    def closeEvent(self, event):
        if comm.rank == 0:
            self.mpi_wait = comm.bcast(numpy.array([2], dtype=numpy.int32), root=0)
            super(MergeWindow, self).closeEvent(event)

    def init_gui_layout(self):
        gui_fname = pkg_resources.resource_filename('circus',
                                                    os.path.join('qt_GUI',
                                                                 'qt_merge.ui'))
        if comm.rank == 0:
            self.ui = uic.loadUi(gui_fname, self)
            # print dir(self.ui)
            self.score_ax1 = self.ui.score_1.axes
            self.score_ax2 = self.ui.score_2.axes
            self.score_ax3 = self.ui.score_3.axes
            self.waveforms_ax = self.ui.waveforms.axes
            self.detail_ax = self.ui.detail.axes
            self.detail_2_ax = self.ui.detail_2.axes
            self.data_ax = self.ui.data_overview.axes
            self.current_order = self.ui.cmb_sorting.currentIndex()
            self.mpl_toolbar = NavigationToolbar(self.ui.waveforms, None)
            self.mpl_toolbar.pan()
            self.ui.show()
        else:
            self.ui = None

    def generate_data(self):

        def reversed_corr(spike_1, spike_2, max_delay):

            size = 2 * max_delay + 1
            x_cc = numpy.zeros(size, dtype=numpy.float32)
            control = 0
            overlap = False

            if (len(spike_1) > 0) and (len(spike_2) > 0):

                t1b = numpy.unique(numpy.round(spike_1 / self.bin_size))
                t2b = numpy.unique(numpy.round(spike_2 / self.bin_size))

                for d in range(size):
                    x_cc[d] += len(numpy.intersect1d(t1b, t2b + d - max_delay, assume_unique=True))

                x_cc /= self.nb_bins

                nspike_1 = spike_1 / self.sampling_rate
                nspike_2 = spike_2 / self.sampling_rate
                xaxis = bins = numpy.arange(0, self.duration, 1)

                is_active_1, _ = numpy.histogram(nspike_1, xaxis)
                is_active_2, _ = numpy.histogram(nspike_2, xaxis)

                is_active = is_active_1 + is_active_2
                duration = len(numpy.where(is_active > 0)[0])
                r1 = len(spike_1) / duration
                r2 = len(spike_2) / duration
                control = len(spike_1) * len(spike_2) / float(self.nb_bins ** 2)

                is_overlapping = is_active_1 * is_active_2
                duration = numpy.where(is_overlapping == 0)[0]
                is_active_1[duration] = 0
                is_active_2[duration] = 0
                overlap = (is_active_1.sum() > self.min_spikes) and (is_active_2.sum() > self.min_spikes)

            return x_cc*1e6, control*1e6, overlap

        def get_rpv(spike_1, spike_2, threshold):

            all_spikes = numpy.sort(numpy.concatenate((spike_1, spike_2))/self.sampling_rate)
            isi = numpy.diff(all_spikes)
            nb_rpv = numpy.where(isi < threshold*1e-3)
            if len(all_spikes) > 0:
                return len(nb_rpv)/len(all_spikes)
            else:
                return 1.

        def largest_nonzero_interval(vec):
            '''
            Find islands of non-zeros in the vector vec
            '''

            edges, = np.nonzero(np.diff((vec==0)*1))
            edge_vec = [edges+1]
            if vec[0] != 0:
                edge_vec.insert(0, [0])
            if vec[-1] != 0:
                edge_vec.append([len(vec)])
            edges = np.concatenate(edge_vec)

            max_size = 0
            for i, j in  zip(edges[::2], edges[1::2]):
                if j - i > max_size:
                    max_size = (j - i)
            return max_size

        self.raw_lags = numpy.linspace(-self.max_delay * self.cc_bin, self.max_delay * self.cc_bin, 2 * self.max_delay + 1)

        self.mpi_wait = comm.bcast(self.mpi_wait, root=0)

        if self.mpi_wait[0] > 0:
            return

        self.indices = comm.bcast(self.indices, root=0)
        self.to_delete = comm.bcast(self.to_delete, root=0)

        to_consider = sorted(set(self.indices) - set(self.to_delete))
        self.to_consider = numpy.array(list(to_consider), dtype=numpy.int32)
        real_indices = self.to_consider[comm.rank::comm.size]

        n_size = 2 * self.max_delay + 1

        self.raw_data = numpy.zeros((0, n_size), dtype=numpy.float32)
        self.raw_control = numpy.zeros(0, dtype=numpy.float32)
        self.pairs = numpy.zeros((0, 2), dtype=numpy.int32)
        self.bhattas = numpy.zeros(0, dtype=numpy.float32)
        self.rpvs = numpy.zeros(0, dtype=numpy.float32)
        self.overlapping = numpy.zeros(0, dtype=numpy.int32)

        to_explore = range(comm.rank, len(self.to_consider), comm.size)

        bins = numpy.linspace(0, self.duration, self.nb_bhatta_bins)

        if comm.rank == 0:
            print_and_log(['Updating the data...'], 'default', logger)
            to_explore = get_tqdm_progressbar(self.params, to_explore)
            if self.app is not None:
                self.ui.label_nb_templates.setText("# templates: %d" % len(self.to_consider))

        for count, temp_id1 in enumerate(to_explore):

            temp_id1 = self.to_consider[temp_id1]
            best_matches = self.to_consider[numpy.argsort(self.overlap[temp_id1, self.to_consider])[::-1]]
            if not self.adapted_cc:
                candidates = best_matches[self.overlap[temp_id1, best_matches] >= self.cc_overlap]
            else:
                candidates = best_matches[self.overlap[temp_id1, best_matches]**self.exponents[temp_id1, best_matches] >= self.cc_overlap]

            for temp_id2 in candidates:

                spikes1 = self.result['spiketimes']['temp_' + str(temp_id1)].astype('int64')
                spikes2 = self.result['spiketimes']['temp_' + str(temp_id2)].copy().astype('int64')
                if self.correct_lag:
                    spikes2 += self.lag[temp_id1, temp_id2]
                a, b, overlap = reversed_corr(spikes1, spikes2, self.max_delay)
                self.raw_data = numpy.vstack((self.raw_data, a))
                self.raw_control = numpy.concatenate((self.raw_control, numpy.array([b], dtype=numpy.float32)))
                self.pairs = numpy.vstack((self.pairs, numpy.array([temp_id1, temp_id2], dtype=numpy.int32)))

                x1, y1 = numpy.histogram(spikes1/self.sampling_rate, bins=bins, density=True)
                x2, y2 = numpy.histogram(spikes2/self.sampling_rate, bins=bins, density=True)

                max_size = largest_nonzero_interval(x1*x2)
                enough_spikes = (len(spikes1) > self.min_spikes) and (len(spikes2) > self.min_spikes)

                if (max_size > 5) and enough_spikes:
                    dist = bhatta_dist(spikes1/self.sampling_rate, spikes2/self.sampling_rate, bounds=(0, self.duration), n_steps=self.nb_bhatta_bins)
                else:
                    dist = 0
                self.bhattas = numpy.concatenate((self.bhattas, numpy.array([dist], dtype=numpy.float32)))
                self.rpvs = numpy.concatenate((self.rpvs, numpy.array([get_rpv(spikes1, spikes2, self.time_rpv)], dtype=numpy.float32)))
                self.overlapping = numpy.concatenate((self.overlapping, numpy.array([overlap], dtype=numpy.int32)))

        sys.stderr.flush()
        self.pairs = gather_array(self.pairs, comm, 0, 1, dtype='int32')
        self.overlapping = gather_array(self.overlapping, comm, 0, 1, dtype='int32')
        self.raw_control = gather_array(self.raw_control, comm)
        self.raw_data = gather_array(self.raw_data, comm, 0, 1)
        self.bhattas = gather_array(self.bhattas, comm, 0, 1)
        self.rpvs = gather_array(self.rpvs, comm, 0, 1)
        self.sort_idcs = numpy.arange(len(self.pairs))
        comm.Barrier()

    def calc_scores(self, lag):
        data = self.raw_data[:, abs(self.raw_lags) <= lag]
        control = self.raw_control
        score = self.overlap[self.pairs[:, 0], self.pairs[:, 1]]
        score2 = numpy.nan_to_num(numpy.maximum(1 - data.mean(axis=1) / control, 0))
        score3 = self.bhattas
        return score, score2, score3

    def plot_scores(self):
        # Left: Scores
        if self.app is not None:
            if not getattr(self, 'collections', None):
                # It is important to set one facecolor per point so that we can change
                # it later
                self.collections = []
                self.collections_inspected = []  # used to visualize inspected points in front of all the points
                for ax, x, y in [(self.score_ax1, self.score_x, self.score_y),
                                 (self.score_ax2, self.norms[self.to_consider], self.rates[self.to_consider]),
                                 (self.score_ax3, self.score_x, self.score_z)]:
                    self.collections.append(
                        ax.scatter(x, y, facecolor=['black' for _ in x])
                    )
                    self.collections_inspected.append(
                        ax.scatter(x, y, s=[0.0 for _ in x], facecolor=['black' for _ in x])
                    )
                # self.score_ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                self.decision_boundary = self.score_ax1.plot([0, 1], [self.suggest_value, self.suggest_value], 'r--', alpha=0.5)[0]
                self.decision_boundary2 = self.score_ax2.plot([self.suggest_value_template, self.suggest_value_template], [0, self.rates[self.to_consider].max()], 'r--', alpha=0.5)[0]
                self.decision_boundary3 = self.score_ax3.plot([0, 1], [self.suggest_value_bhatta, self.suggest_value_bhatta], 'r--', alpha=0.5)[0]
                self.score_ax1.set_ylabel('CC metric')
                self.score_ax1.set_xlabel('Template similarity')
                self.score_ax2.set_xlabel('Template Norm')
                self.score_ax2.set_ylabel('Rate [Hz]')
                self.score_ax3.set_xlabel('Template similarity')
                self.score_ax3.set_ylabel('Bhatta distance')
                self.waveforms_ax.set_xticks([])
                self.waveforms_ax.set_yticks([])
                # self.waveforms_ax.set_xlabel('Time [ms]')
                # self.waveforms_ax.set_ylabel('Amplitude')
            else:
                xys = [
                    (self.score_x, self.score_y),
                    (self.norms[self.to_consider], self.rates[self.to_consider]),
                    (self.score_x, self.score_z)
                ]
                for collection, (x, y) in zip(self.collections, xys):
                    collection.set_offsets(np.hstack([x[np.newaxis, :].T, y[np.newaxis, :].T]))
                for collection_inspected, (x, y) in zip(self.collections_inspected, xys):
                    collection_inspected.set_offsets(np.hstack([x[np.newaxis, :].T, y[np.newaxis, :].T]))

            for ax, score_y, score_x in [(self.score_ax1, self.score_y, self.score_x),
                                         (self.score_ax2, self.rates[self.to_consider], self.norms[self.to_consider]),
                                         (self.score_ax3, self.score_z, self.score_x)]:
                if len(score_y) > 0:
                    ymin, ymax = min(score_y), max(score_y)
                else:
                    ymin, ymax = 0, 1
                yrange = (ymax - ymin)*0.5 * 1.05  # stretch everything a bit
                ax.set_ylim((ymax + ymin)*0.5 - yrange, (ymax + ymin)*0.5 + yrange)

                if len(score_x) > 0:
                    xmin, xmax = min(score_x), max(score_x)
                else:
                    xmin, xmax = 0, 1
                range = (xmax - xmin)*0.5 * 1.05  # stretch everything a bit
                ax.set_xlim((xmax + xmin)*0.5 - range, (xmax + xmin)*0.5 + range)

            for fig in [self.ui.score_1, self.ui.score_2, self.ui.score_3, self.ui.waveforms]:
                fig.draw_idle()

    def plot_data(self):
        # Right: raw data
        if self.app is not None:
            all_raw_data = self.raw_data
            all_raw_data /= (1 + all_raw_data.mean(1)[:, np.newaxis])
            if len(all_raw_data) > 0:
                cmax = 0.5*all_raw_data.max()
                cmin = 0.5*all_raw_data.min()
            else:
                cmin = 0
                cmax = 1
            self.update_sort_idcs()
            all_raw_data = all_raw_data[self.sort_idcs, :]

            self.data_image = self.data_ax.imshow(all_raw_data,
                                                  interpolation='nearest', cmap='coolwarm',
                                                  extent=(self.raw_lags[0], self.raw_lags[-1],
                                                          0, len(self.sort_idcs)), origin='lower')
            self.data_ax.set_aspect('auto')
            self.data_ax.spines['right'].set_visible(False)
            self.data_ax.spines['left'].set_visible(False)
            self.data_ax.spines['top'].set_visible(False)
            self.data_image.set_clim(cmin, cmax)
            self.inspect_markers = self.data_ax.scatter([], [], marker='<',
                                                        clip_on=False, s=40)
            self.data_selection = mpl.patches.Rectangle((self.raw_lags[0], 0),
                                                        width=self.raw_lags[-1] - self.raw_lags[0],
                                                        height=0,
                                                        color='white', alpha=0.75)
            self.data_ax.add_patch(self.data_selection)
            self.data_ax.set_xlim(self.raw_lags[0], self.raw_lags[-1])
            self.data_ax.set_ylim(0, len(self.sort_idcs)+1)
            self.data_ax.set_yticks([])
            self.ui.data_overview.draw()

    def data_tooltip(self, x, y):
        row = int(y)
        if row >= 0 and row < len(self.raw_data):
            all_raw_data = self.raw_data
            data_idx = self.sort_idcs[row]
            lag_diff = np.abs(x - self.raw_lags)
            nearest_lag_idx = np.argmin(lag_diff)
            nearest_lag = self.raw_lags[nearest_lag_idx]
            value = all_raw_data[data_idx, nearest_lag_idx]
            return ('%.2f - lag: %.2fms (template similarity: %.2f  '
                    'CC metric %.2f)') % (value, nearest_lag, self.score_x[data_idx], self.score_y[data_idx])
        else:
            return ''

    def callback_lasso(self, verts):
        p = mpl.path.Path(verts)
        in_selection = p.contains_points(self.lasso_selector.points)
        indices = np.nonzero(in_selection)[0]
        if len(self.lasso_selector.points) != len(self.points[1]):
            self.update_inspect(indices, self.lasso_selector.add_or_remove)
        else:
            self.update_inspect_template(indices, self.lasso_selector.add_or_remove)

    def callback_rect(self, eclick, erelease):
        xmin, xmax, ymin, ymax = eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin

        self.score_ax = eclick.inaxes

        if self.score_ax == self.score_ax1:
            score_x, score_y = self.score_x, self.score_y
        elif self.score_ax == self.score_ax2:
            score_x, score_y = self.norms[self.to_consider], self.rates[self.to_consider]
        elif self.score_ax == self.score_ax3:
            score_x, score_y = self.score_x, self.score_z

        in_selection = ((score_x >= xmin) &
                        (score_x <= xmax) &
                        (score_y >= ymin) &
                        (score_y <= ymax))
        indices = np.nonzero(in_selection)[0]
        add_or_remove = None
        if erelease.key == 'shift':
            add_or_remove = 'add'
        elif erelease.key == 'control':
            add_or_remove = 'remove'

        if self.score_ax != self.score_ax2:
            self.update_inspect(indices, add_or_remove)
        else:
            self.update_inspect_template(indices, add_or_remove)

    def zoom(self, event):
        if event.inaxes == self.score_ax1:
            x = self.score_x
            y = self.score_y
            link_with_x = self.score_ax1.set_xlim
            link_with_y = None
        elif event.inaxes == self.score_ax3:
            x = self.score_x
            y = self.score_z
            link_with_x = self.score_ax3.set_xlim
            link_with_y = None
        elif event.inaxes == self.score_ax2:
            x = self.norms[self.to_consider]
            y = self.rates[self.to_consider]
            link_with_x = None
            link_with_y = None
        elif event.inaxes == self.waveforms_ax:
            x = self.x_position
            y = self.y_position
            link_with_x = None
            link_with_y = None
        else:
            return

        score_ax = event.inaxes
        # get the current x and y limits
        cur_xlim = score_ax.get_xlim()
        cur_ylim = score_ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1 / 2.0
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = 2.0
        else:
            # deal with something that should never happen
            scale_factor = 1
        # set new limits
        newxmin = np.clip(xdata - cur_xrange * scale_factor, np.min(x), np.max(x))
        newxmax = np.clip(xdata + cur_xrange * scale_factor, np.min(x), np.max(x))
        new_xrange = (newxmax - newxmin) * 0.5 * 1.05  # stretch everything a bit
        newxmin = (newxmax + newxmin) * 0.5 - new_xrange
        newxmax = (newxmax + newxmin) * 0.5 + new_xrange
        newymin = np.clip(ydata - cur_yrange*scale_factor, np.min(y), np.max(y))
        newymax = np.clip(ydata + cur_yrange*scale_factor, np.min(y), np.max(y))
        new_yrange = (newymax - newymin) * 0.5 * 1.05  # stretch everything a bit
        newymin = (newymax + newymin) * 0.5 - new_yrange
        newymax = (newymax + newymin) * 0.5 + new_yrange
        score_ax.set_xlim(newxmin, newxmax)
        score_ax.set_ylim(newymin, newymax)
        # Update the linked axes in the other plots as well
        if link_with_x is not None:
            link_with_x(newxmin, newxmax)
        if link_with_y is not None:
            link_with_y(newymin, newymax)

        for fig in [self.ui.score_1, self.ui.score_3, self.ui.score_2, self.ui.waveforms]:
            fig.draw_idle()

    def update_lag(self, lag):
        actual_lag = self.raw_lags[np.argmin(np.abs(self.raw_lags - lag))]
        self.use_lag = actual_lag
        self.score_x, self.score_y, self.score_z = self.calc_scores(lag=self.use_lag)
        if self.app is not None:
            self.points = [zip(self.score_x, self.score_y),
                           zip(self.norms[self.to_consider], self.rates[self.to_consider]),
                           zip(self.score_x, self.score_z)]
            self.line_lag1.set_xdata((lag, lag))
            self.line_lag2.set_xdata((-lag, -lag))
            self.data_ax.set_xlabel('lag (ms) -- cutoff: %.2fms' % self.use_lag)

    def update_rect_selector(self, event):
        for selector in self.rect_selectors:
            selector.set_active(self.ui.btn_rectangle.isChecked())

    def update_detail_plot(self):
        indices = self.inspect_points
        all_raw_data = self.raw_data
        all_raw_control = self.raw_control.ravel()

        if self.app is not None:
            self.detail_ax.clear()
            for count, idx in enumerate(indices):
                data_line, = self.detail_ax.plot(self.raw_lags,
                                                 all_raw_data[idx, :].T, lw=2, color=self.inspect_colors[count])
                self.detail_ax.plot(self.raw_lags, all_raw_control[idx]*numpy.ones(all_raw_data.shape[1]), ':',
                                    color=self.inspect_colors[count], lw=2)
            ymin, ymax = self.detail_ax.get_ylim()
            self.detail_ax.plot([-self.use_lag, -self.use_lag], [0, ymax], 'k')
            self.detail_ax.plot([self.use_lag, self.use_lag], [0, ymax], 'k')
            self.detail_ax.set_xticks(self.data_ax.get_xticks())
            self.detail_ax.set_xticklabels([])
            self.ui.detail.draw_idle()

    def update_detail_2_plot(self):
        indices = self.inspect_points
        temp_indices = self.to_consider[list(self.inspect_templates)]
        bins = numpy.linspace(0, self.duration, self.nb_bhatta_bins)

        if self.app is not None:
            self.detail_2_ax.clear()
            for count, idx in enumerate(indices):

                n1, n2 = self.pairs[idx]
                spikes1 = self.result['spiketimes']['temp_' + str(n1)].astype('int64')
                spikes2 = self.result['spiketimes']['temp_' + str(n2)].astype('int64')
                if self.correct_lag:
                    spikes2 += self.lag[n1, n2]

                spikes1 = spikes1 / self.sampling_rate
                spikes2 = spikes2 / self.sampling_rate

                try:
                    x1, y = numpy.histogram(spikes1, bins=bins, density=True)
                    cidx = numpy.where(temp_indices == n1)[0][0]
                    data_line, = self.detail_2_ax.plot(y[1:], x1, lw=2, color=self.inspect_colors_templates[cidx])
                    x2, y = numpy.histogram(spikes2, bins=bins, density=True)
                    cidx = numpy.where(temp_indices == n2)[0][0]
                    data_line, = self.detail_2_ax.plot(y[1:], x2, lw=2, color=self.inspect_colors_templates[cidx])
                    #data_line, = self.detail_2_ax.plot(y[1:], (x1+x2)/2, lw=1, color='k')
                except Exception:
                    pass


            data_line, = self.detail_2_ax.plot(bins, self.bhattas_level*numpy.ones(self.nb_bhatta_bins), lw=1, color='k')
            self.detail_2_ax.set_xticks(numpy.linspace(0, self.duration, 5))
            self.detail_2_ax.set_xlabel('Time [s]')
            self.detail_2_ax.set_ylabel('Spike density')
            self.detail_2_ax.set_xlim(0, self.duration)
            self.ui.detail_2.draw_idle()

    def update_sort_idcs(self):
        # The selected points are sorted before all the other points -- an easy
        # way to achieve this is to add the maximum score to their score
        
        if self.current_order == 0:
            score = self.score_x
        elif self.current_order == 1:
            score = self.score_y
        elif self.current_order == 2:
            score = self.score_z
        else:
            raise AssertionError(self.current_order)
        
        score = score.copy()
        if len(self.selected_points):
            score[np.array(sorted(self.selected_points))] += score.max()

        self.sort_idcs = np.argsort(score)

    def update_data_plot(self):
        reverse_sort = np.argsort(self.sort_idcs)

        if self.app is not None:
            if len(self.inspect_points):
                inspect = reverse_sort[np.array(sorted(self.inspect_points))]
                data = numpy.vstack((np.ones(len(inspect))*(2*self.raw_lags[-1]-self.raw_lags[-2]), inspect+0.5)).T
                self.inspect_markers.set_offsets(data)
                self.inspect_markers.set_color(self.inspect_colors)
            else:
                self.inspect_markers.set_offsets([])
                self.inspect_markers.set_color([])

            self.ui.data_overview.draw_idle()

    def update_data_sort_order(self, new_sort_order=None):
        if new_sort_order is not None:
            self.current_order = new_sort_order
        self.update_sort_idcs()

        if self.app is not None:
            self.data_image.set_extent((self.raw_lags[0], self.raw_lags[-1], 0, len(self.sort_idcs)))
            self.data_ax.set_ylim(0, len(self.sort_idcs))
            all_raw_data = self.raw_data.copy()
            all_raw_data /= (1 + all_raw_data.mean(1)[:, np.newaxis])
            if len(all_raw_data) > 0:
                cmax = 0.5 * all_raw_data.max()
                cmin = 0.5 * all_raw_data.min()
                all_raw_data = all_raw_data[self.sort_idcs, :]
            else:
                cmin = 0
                cmax = 1
            self.data_image.set_data(all_raw_data)
            self.data_image.set_clim(cmin, cmax)
            self.data_selection.set_y(len(self.sort_idcs)-len(self.selected_points))
            self.data_selection.set_height(len(self.selected_points))
            self.update_data_plot()

    def update_waveforms(self):

        if self.app is not None:
            self.waveforms_ax.clear()

            all_channels = []

            for idx, p in enumerate(self.to_consider[list(self.inspect_templates)]):
                tmp = self.templates[:, p]
                tmp = tmp.toarray().reshape(self.N_e, self.N_t)
                elec = numpy.argmin(numpy.min(tmp, 1))
                thr = self.thresholds[elec]

                if self.ui.show_peaks.isChecked():
                    indices = [self.inv_nodes[self.nodes[elec]]]
                    all_channels += indices
                else:
                    if self.has_support:
                        indices = numpy.where(self.supports[p])[0]
                    else:
                        indices = self.inv_nodes[self.edges[self.nodes[elec]]]
                    all_channels += list(indices)

                for sidx in indices:
                    xaxis = numpy.linspace(self.x_position[sidx], self.x_position[sidx] + (self.N_t/(self.sampling_rate*1e-3)), self.N_t)
                    self.waveforms_ax.plot(xaxis, self.y_position[sidx] + tmp[sidx], c=colorConverter.to_rgba(self.inspect_colors_templates[idx]))

            if self.ui.show_thresholds.isChecked():
                for sidx in numpy.unique(all_channels):
                    thr = self.thresholds[sidx]
                    xaxis = numpy.linspace(self.x_position[sidx], self.x_position[sidx] + (self.N_t/(self.sampling_rate*1e-3)), self.N_t)
                    if self.sign_peaks in ['negative', 'both']:
                        self.waveforms_ax.plot([xaxis[0], xaxis[-1]], [self.y_position[sidx]-thr, self.y_position[sidx]-thr], c='k', linestyle='--')
                    if self.sign_peaks in ['positive', 'both']:
                        self.waveforms_ax.plot([xaxis[0], xaxis[-1]], [self.y_position[sidx]+thr, self.y_position[sidx]+thr], c='k', linestyle='--')
                    self.waveforms_ax.plot([xaxis[0], xaxis[-1]], [self.y_position[sidx], self.y_position[sidx]], c='0.5', linestyle='--')

            if self.ui.show_labels.isChecked():
                for sidx in numpy.unique(all_channels):
                    self.waveforms_ax.text(self.x_position[sidx], self.y_position[sidx], self.label[sidx])

            self.waveforms_ax.set_xlabel('Probe Space')
            self.waveforms_ax.set_ylabel('Probe Space')

            for fig in [self.ui.waveforms]:
                fig.draw_idle()

    def update_score_plot(self):
        if self.app is not None:
            for collection in self.collections:
                if collection.axes == self.score_ax2:  # i.e. template rate v.s. template norm
                    fcolors = collection.get_facecolors()
                    colorin = colorConverter.to_rgba('black', alpha=0.5)
                    colorout = colorConverter.to_rgba('black')
                    fcolors[:] = colorout
                    for p in self.selected_templates:
                        fcolors[p] = colorin
                    for idx, p in enumerate(self.inspect_templates):
                        fcolors[p] = colorConverter.to_rgba(self.inspect_colors_templates[idx])
                else:
                    fcolors = collection.get_facecolors()
                    colorin = colorConverter.to_rgba('black', alpha=0.5)
                    colorout = colorConverter.to_rgba('black')
                    fcolors[:] = colorout
                    for p in self.selected_points:
                        fcolors[p] = colorin
                    for idx, p in enumerate(self.inspect_points):
                        fcolors[p] = colorConverter.to_rgba(self.inspect_colors[idx])
            for collection_inspected in self.collections_inspected:
                if collection_inspected.axes == self.score_ax2:  # i.e. template rate v.s. template norm
                    # Change colors of selected templates.
                    fcolors = collection_inspected.get_facecolors()
                    colorin = colorConverter.to_rgba('black', alpha=0.5)
                    colorout = colorConverter.to_rgba('black')
                    fcolors[:] = colorout
                    for p in self.selected_templates:
                        fcolors[p] = colorin
                    for idx, p in enumerate(self.inspect_templates):
                        fcolors[p] = colorConverter.to_rgba(self.inspect_colors_templates[idx])
                    # Make unselected templates not visible.
                    sizes = np.copy(collection_inspected.get_sizes())
                    sizes[:] = 0
                    sizes[numpy.array(list(self.inspect_templates), dtype='int32')] = mpl.rcParams['lines.markersize'] ** 2
                    collection_inspected.set_sizes(sizes)
                else:
                    # Change colors to selected points.
                    fcolors = collection_inspected.get_facecolors()
                    colorin = colorConverter.to_rgba('black', alpha=0.5)
                    colorout = colorConverter.to_rgba('black')
                    fcolors[:] = colorout
                    for p in self.selected_points:
                        fcolors[p] = colorin
                    for idx, p in enumerate(self.inspect_points):
                        fcolors[p] = colorConverter.to_rgba(self.inspect_colors[idx])
                    #  Make unselected points not visible.
                    sizes = np.copy(collection_inspected.get_sizes())
                    sizes[:] = 0
                    sizes[numpy.array(list(self.inspect_points), dtype='int32')] = mpl.rcParams['lines.markersize'] ** 2
                    collection_inspected.set_sizes(sizes)

            for fig in [self.ui.score_1, self.ui.score_2, self.ui.score_3]:
                fig.draw_idle()

    def update_inspect_template(self, indices, add_or_remove=None, link=True):
        if MPL_VERSION:
            all_colors = mpl.colors.colorConverter.to_rgba_array([i['color'] for i in plt.rcParams['axes.prop_cycle'].__dict__['_left']])
        else:
            all_colors = colorConverter.to_rgba_array(plt.rcParams['axes.color_cycle'])

        indices = self.to_consider[list(indices)]

        for i in range(len(indices)):
            indices[i] -= numpy.sum(self.to_delete <= indices[i])

        if add_or_remove is 'add':
            indices = set(self.inspect_templates) | set(indices)
        elif add_or_remove is 'remove':
            indices = set(self.inspect_templates) - set(indices)

        self.inspect_templates = sorted(indices)

        # We use a deterministic mapping to colors, based on their index
        self.inspect_colors_templates = [
            all_colors[idx % len(all_colors)]
            for idx in self.inspect_templates
        ]

        is_selected_1 = numpy.where(numpy.in1d(self.pairs[:, 0], self.inspect_templates) == True)[0]
        is_selected_2 = numpy.where(numpy.in1d(self.pairs[:, 1], self.inspect_templates) == True)[0]
        is_selected = numpy.unique(numpy.concatenate((is_selected_1, is_selected_2)))
        if link:
            self.inspect_points = set()
            self.update_inspect(is_selected, 'add', False)

        if self.app is not None:
            self.ui.label_pairs_selected.setText("# pairs selected: %d" % len(self.inspect_points))
            self.ui.label_temp_selected.setText("# templates selected: %d" % len(self.inspect_templates))
            self.update_score_plot()
            self.update_detail_plot()
            self.update_detail_2_plot()
            self.update_data_plot()
            self.update_waveforms()

    def update_inspect(self, indices, add_or_remove=None, link=True):
        if MPL_VERSION:
            all_colors = mpl.colors.colorConverter.to_rgba_array([i['color'] for i in plt.rcParams['axes.prop_cycle'].__dict__['_left']])
        else:
            all_colors = colorConverter.to_rgba_array(plt.rcParams['axes.color_cycle'])

        if add_or_remove is 'add':
            indices = set(self.inspect_points) | set(indices)
        elif add_or_remove is 'remove':
            indices = set(self.inspect_points) - set(indices)

        self.inspect_points = sorted(indices)
        # We use a deterministic mapping to colors, based on their index
        self.inspect_colors = [all_colors[idx % len(all_colors)]
                               for idx in self.inspect_points]

        if link:
            self.inspect_templates = set()
            all_templates = numpy.unique(self.pairs[list(indices)].flatten())
            indices = []
            for i in all_templates:
                indices += [numpy.where(self.to_consider == i)[0][0]]
            self.update_inspect_template(indices, 'add', False)

        if self.app is not None:
            self.ui.label_pairs_selected.setText("# pairs selected: %d" % len(self.inspect_points))
            self.ui.label_temp_selected.setText("# templates selected: %d" % len(self.inspect_templates))
            self.update_score_plot()
            self.update_detail_plot()
            self.update_detail_2_plot()
            self.update_data_plot()
            self.update_waveforms()

    def update_selection(self, indices, add_or_remove=None):
        if add_or_remove is None:
            self.selected_points.clear()

        if add_or_remove == 'remove':
            self.selected_points.difference_update(set(indices))
        else:
            self.selected_points.update(set(indices))

        if self.app is not None:
            self.ui.label_pairs_selected.setText("# pairs selected: %d" % len(self.inspect_points))
            self.ui.label_temp_selected.setText("# templates selected: %d" % len(self.inspect_templates))
            self.update_score_plot()
            self.update_detail_plot()
            self.update_detail_2_plot()
            self.update_data_sort_order()

    def add_to_selection(self, event):
        _ = event  # (PyCharm code inspection)
        to_add = sorted(set(self.inspect_points))
        self.inspect_points = set()
        self.update_selection(to_add, add_or_remove='add')

    def remove_selection(self, event):
        _ = event  # (PyCharm code inspection)
        self.inspect_points = set()
        self.update_selection(self.selected_points, add_or_remove='remove')

    def remove_selection_templates(self, event):
        _ = event  # (PyCharm code inspection)
        self.inspect_templates = set()
        self.update_inspect_template(self.inspect_templates, add_or_remove='remove')

    def suggest_pairs(self, event):
        _ = event  # (PyCharm code inspection)
        self.inspect_points = set()

        all_indices = (self.score_y > self.suggest_value)

        if self.app is not None:
            test1 = self.ui.prevent_rpv.isChecked()
        else:
            test1 = True

        if test1:
            all_indices = all_indices & (self.rpvs < self.rpv_threshold)

        if self.app is not None:
            test2 = self.ui.prevent_min_spikes.isChecked()
        else:
            test2 = True

        if test2:
            all_indices = all_indices & (self.overlapping == 1)

        indices = numpy.where(all_indices)[0]

        if self.low_channel_count:
            ratios = self.norms[self.pairs[indices,0]]/self.norms[self.pairs[indices,1]]
            indices = indices[numpy.where(numpy.abs(ratios - 1) < 0.1)[0]]

        if self.app is not None:
            self.app.setOverrideCursor(QCursor(Qt.WaitCursor))

        self.update_inspect(indices, add_or_remove='add')
        
        if self.app is not None:
            self.app.restoreOverrideCursor()

    def suggest_drifts(self, event):
        _ = event  # (PyCharm code inspection)
        self.inspect_points = set()
        
        all_indices = (self.score_z > self.suggest_value_bhatta)

        if self.app is not None:
            test1 = self.ui.prevent_rpv.isChecked()
        else:
            test1 = True

        if test1:
            all_indices = all_indices & (self.rpvs < self.rpv_threshold)

        indices = numpy.where(all_indices)[0]

        if self.app is not None:
            self.app.setOverrideCursor(QCursor(Qt.WaitCursor))

        self.update_inspect(indices, add_or_remove='add')
        
        if self.app is not None:
            self.app.restoreOverrideCursor()

    def suggest_templates(self, event):
        _ = event  # (PyCharm code inspection)
        self.inspect_templates = set()
        indices = numpy.where((self.norms[self.to_consider] <= self.suggest_value_template) & (self.sparsities[self.to_consider] >= self.sparsity_limit))[0]
        self.update_inspect_template(indices, add_or_remove='add')

    def on_mouse_press(self, event):
        if event.inaxes in [self.score_ax1, self.score_ax2, self.score_ax3]:
            if self.ui.btn_lasso.isChecked():
                # Select multiple points
                self.start_lasso_select(event)
            elif self.ui.btn_rectangle.isChecked():
                pass  # handled already by rect selector
            elif self.ui.btn_picker.isChecked():
                # Select a single point for display
                # Find the closest point
                if event.inaxes == self.score_ax1:
                    x = self.score_x
                    y = self.score_y
                elif event.inaxes == self.score_ax2:
                    x = self.norms[self.to_consider]
                    y = self.rates[self.to_consider]
                elif event.inaxes == self.score_ax3:
                    x = self.score_x
                    y = self.score_z
                elif event.inaxes == self.waveforms_ax:
                    pass
                else:
                    raise AssertionError(str(event.inaxes))

                # Transform data coordinates to display coordinates
                data = event.inaxes.transData.transform(zip(x, y))

                distances = ((data[:, 0] - event.x)**2 +
                             (data[:, 1] - event.y)**2)
                min_idx, min_value = np.argmin(distances), np.min(distances)
                if min_value > 50:
                    # Don't select anything if the mouse cursor is more than
                    # 50 pixels away from a point
                    selection = {}
                else:
                    selection = {min_idx}
                add_or_remove = None
                if event.key == 'shift':
                    add_or_remove = 'add'
                elif event.key == 'control':
                    add_or_remove = 'remove'
                if event.inaxes == self.score_ax2:
                    self.update_inspect_template(selection, add_or_remove)
                else:
                    self.update_inspect(selection, add_or_remove)
            else:
                raise AssertionError('No tool active')
        elif event.inaxes == self.data_ax:
            if self.lag_selector.active:
                # Update lag
                self.update_lag(abs(event.xdata))
                self.update_detail_plot()
                self.lag_selector.active = False
                self.plot_scores()
                self.update_data_plot()
            else:  # select a line
                if event.ydata < 0 or event.ydata >= len(self.sort_idcs):
                    return
                index = self.sort_idcs[int(event.ydata)]
                if index in self.selected_points:
                    return
                if event.key == 'shift':
                    add_or_remove = 'add'
                elif event.key == 'control':
                    add_or_remove = 'remove'
                else:
                    add_or_remove = None
                self.update_inspect({index}, add_or_remove)
        else:
            return

    def start_lasso_select(self, event):
        self.lasso_selector = widgets.Lasso(event.inaxes,
                                            (event.xdata, event.ydata),
                                            self.callback_lasso)
        add_or_remove = None
        if event.key == 'shift':
            add_or_remove = 'add'
        elif event.key == 'control':
            add_or_remove = 'remove'
        self.lasso_selector.add_or_remove = add_or_remove
        if event.inaxes == self.score_ax1:
            self.lasso_selector.points = self.points[0]
        elif event.inaxes == self.score_ax2:
            self.lasso_selector.points = self.points[1]
        else:
            self.lasso_selector.points = self.points[2]

    def remove_templates(self, event):
        _ = event  # (PyCharm code inspection)
        print_and_log(['Deleting %d noisy templates' % len(self.inspect_templates)], 'default', logger)
        print_and_log(['Deleting noisy templates: %s' % str(sorted(self.inspect_templates))], 'debug', logger)

        if self.app is not None:
            self.app.setOverrideCursor(QCursor(Qt.WaitCursor))

        if len(self.inspect_templates) > 0:
            self.to_delete = numpy.concatenate((self.to_delete, self.to_consider[self.inspect_templates]))
            self.generate_data()
            self.update_lag(self.use_lag)
            self.collections = None
            self.collections_inspected = None
            self.selected_points = set()
            self.selected_templates = set()
            self.inspect_points = set()
            self.inspect_templates = set()
            if self.app is not None:
                self.score_ax1.clear()
                self.score_ax2.clear()
                self.score_ax3.clear()
                self.update_data_sort_order()
                self.update_detail_plot()
                self.update_detail_2_plot()
                self.update_waveforms()
                self.plot_scores()
            # do lengthy process
    
        if self.app is not None:
            self.app.restoreOverrideCursor()

    def do_merge(self, event, regenerate=True):
        # This simply removes the data points for now
        print_and_log(['Data indices to merge: %s' % str(sorted(self.selected_points))], 'debug', logger)

        if self.app is not None:
            self.app.setOverrideCursor(QCursor(Qt.WaitCursor))

        for pair in self.pairs[list(self.selected_points), :]:

            one_merge = [self.indices[pair[0]], self.indices[pair[1]]]

            elec_ic1 = self.electrodes[one_merge[0]]
            elec_ic2 = self.electrodes[one_merge[1]]
            nic1 = one_merge[0] - numpy.where(self.electrodes == elec_ic1)[0][0]
            nic2 = one_merge[1] - numpy.where(self.electrodes == elec_ic2)[0][0]
            mask1 = self.clusters['clusters_' + str(elec_ic1)] > -1
            mask2 = self.clusters['clusters_' + str(elec_ic2)] > -1
            tmp1 = numpy.unique(self.clusters['clusters_' + str(elec_ic1)][mask1])
            tmp2 = numpy.unique(self.clusters['clusters_' + str(elec_ic2)][mask2])
            elements1 = numpy.where(self.clusters['clusters_' + str(elec_ic1)] == tmp1[nic1])[0]
            elements2 = numpy.where(self.clusters['clusters_' + str(elec_ic2)] == tmp2[nic2])[0]

            if len(elements1) > len(elements2):
                to_remove = one_merge[1]
                to_keep = one_merge[0]
                elec = elec_ic2
                elements = elements2
            else:
                to_remove = one_merge[0]
                to_keep = one_merge[1]
                elec = elec_ic1
                elements = elements1

            if to_keep != to_remove:
                key = 'temp_' + str(to_keep)
                key2 = 'temp_' + str(to_remove)
                spikes = self.result['spiketimes'][key2]
                if self.correct_lag:
                    spikes = spikes.astype(numpy.int64)
                    spikes += self.lag[to_keep, to_remove]
                    spikes = spikes.astype(numpy.uint32)
                amplitudes = self.result['amplitudes'][key2]
                n1, n2 = len(self.result['amplitudes'][key2]), len(self.result['amplitudes'][key])
                self.result['amplitudes'][key] = numpy.vstack((self.result['amplitudes'][key].reshape(n2, 2), amplitudes.reshape(n1, 2)))
                self.result['spiketimes'][key] = numpy.concatenate((self.result['spiketimes'][key], spikes))
                idx = numpy.argsort(self.result['spiketimes'][key])
                self.result['spiketimes'][key] = self.result['spiketimes'][key][idx]
                self.result['amplitudes'][key] = self.result['amplitudes'][key][idx]
                self.result['spiketimes'].pop(key2)
                self.result['amplitudes'].pop(key2)

                self.all_merges = numpy.vstack((self.all_merges, [self.indices[to_keep], self.indices[to_remove]]))
                idx = numpy.where(self.indices == to_remove)[0]
                self.to_delete = numpy.concatenate((self.to_delete, [to_remove]))
                self.rates[to_keep] += self.rates[to_remove]
                self.indices[idx] = self.indices[to_keep]

        if regenerate:
            self.generate_data()
            self.update_lag(self.use_lag)
            self.collections = None
            self.collections_inspected = None
            self.selected_points = set()
            self.selected_templates = set()
            self.inspect_points = set()
            self.inspect_templates = set()
            if self.app is not None:
                self.score_ax1.clear()
                self.score_ax2.clear()
                self.score_ax3.clear()                
                self.update_data_sort_order()
                self.update_detail_plot()
                self.update_detail_2_plot()
                self.update_waveforms()
                self.plot_scores()

        # do lengthy process

        if self.app is not None:
            self.app.restoreOverrideCursor()

    def finalize(self, event):
        _ = event  # (PyCharm code inspection)

        if comm.rank == 0:
            if self.app is not None:
                self.app.setOverrideCursor(QCursor(Qt.WaitCursor))
            self.mpi_wait = comm.bcast(numpy.array([1], dtype=numpy.int32), root=0)

        comm.Barrier()
        self.all_merges = comm.bcast(self.all_merges, root=0)
        self.to_delete = comm.bcast(self.to_delete, root=0)

        slice_templates(self.params, to_merge=self.all_merges, to_remove=list(self.to_delete), extension=self.ext_out)
        slice_clusters(self.params, self.clusters, to_merge=self.all_merges, to_remove=list(self.to_delete), extension=self.ext_out, light=True)

        if comm.rank == 0:
            new_result = {'spiketimes': {}, 'amplitudes': {}}

            to_keep = sorted(set(numpy.unique(self.indices)) - set(self.to_delete))
            to_keep = numpy.array(list(to_keep))

            # nb_templates = self.nb_templates - len(self.to_delete) - len(self.all_merges) 
            print_and_log([
                'We kept %d templates out of %d after merging' % (len(to_keep), self.nb_templates),
                'As meta merging is still experimental, you need',
                'to use the extension option to see its results',
                '>> circus-gui-matlab mydata.dat -e merged',
                'or',
                '>> spyking-circus mydata.dat -m converting -e merged'
            ], 'info', logger)

            for count, temp_id in enumerate(to_keep):
                key_before = 'temp_' + str(temp_id)
                key_after = 'temp_' + str(count)
                new_result['spiketimes'][key_after] = self.result['spiketimes'].pop(key_before)
                new_result['amplitudes'][key_after] = self.result['amplitudes'].pop(key_before)

            keys = ['spiketimes', 'amplitudes']

            if self.params.getboolean('fitting', 'collect_all'):
                keys += ['gspikes']
                new_result['gspikes'] = io.get_garbage(self.params)['gspikes']

            mydata = h5py.File(self.file_out_suff + '.result%s.hdf5' % self.ext_out, 'w', libver='earliest')
            for key in keys:
                mydata.create_group(key)
                for temp in new_result[key].keys():
                    tmp_path = '%s/%s' %(key, temp)
                    mydata.create_dataset(tmp_path, data=new_result[key][temp])
            mydata.close()

            mydata = h5py.File(self.file_out_suff + '.templates%s.hdf5' % self.ext_out, 'r+', libver='earliest')
            version = mydata.create_dataset('version', data=numpy.array(circus.__version__.split('.'), dtype=numpy.int32))
            maxoverlaps = mydata.create_dataset('maxoverlap', shape=(len(to_keep), len(to_keep)), dtype=numpy.float32)
            maxlag = mydata.create_dataset('maxlag', shape=(len(to_keep), len(to_keep)), dtype=numpy.int32)
            for c, i in enumerate(to_keep):
                maxoverlaps[c, :] = self.overlap[i, to_keep]*self.shape[0] * self.shape[1]
                maxlag[c, :] = self.lag[i, to_keep]
            mydata.close()

            if self.app is not None:
                self.app.restoreOverrideCursor()

        if self.SHARED_MEMORY:
            for memory in self.mpi_memory_1 + self.mpi_memory_2:
                memory.Free()

        sys.exit(0)

    def merge_and_finalize(self, event):

        self.do_merge(event, regenerate=False)
        self.finalize(event)


class PreviewGUI(QMainWindow):

    def __init__(self, params, show_fit=False):
        super(PreviewGUI, self).__init__()

        self.show_fit = show_fit
        self.params = params
        self.data_file = params.data_file
        self.data_file.open()
        self.maxtime = self.data_file.t_stop/self.data_file.sampling_rate
        self.mintime = self.data_file.t_start/self.data_file.sampling_rate
        self.init_gui_layout()
        self.probe = self.params.probe
        self.N_e = self.params.getint('data', 'N_e')
        self.N_t = self.params.getint('detection', 'N_t')
        self.spike_thresh = self.params.getfloat('detection', 'spike_thresh')
        self.peaks_sign = self.params.get('detection', 'peaks')
        self.N_total = self.params.nb_channels
        self.sampling_rate = self.params.rate
        self.template_shift = self.params.getint('detection', 'template_shift')
        self.filename = self.params.get('data', 'data_file')

        name = os.path.basename(self.filename)
        r, f = os.path.splitext(name)
        local_path = os.path.join(r, 'tmp')
        self.filename = self.filename.replace(local_path, '')

        nodes, edges = get_nodes_and_edges(self.params)
        self.nodes = nodes
        self.edges = edges

        self.do_temporal_whitening = self.params.getboolean('whitening', 'temporal')
        self.do_spatial_whitening = self.params.getboolean('whitening', 'spatial')

        if self.do_spatial_whitening:
            self.spatial_whitening = io.load_data(self.params, 'spatial_whitening')
        if self.do_temporal_whitening:
            self.temporal_whitening = io.load_data(self.params, 'temporal_whitening')

        self.thresholds = io.load_data(self.params, 'thresholds')
        self.t_start = self.data_file.t_start/self.data_file.sampling_rate
        self.t_stop = self.t_start + 1
        self.has_mua = False

        if self.show_fit:
            try:
                self.templates = io.load_data(self.params, 'templates')
                self.result = io.load_data(self.params, 'results')
            except Exception:
                print_and_log(["No results found!"], 'info', logger)
                self.show_fit = False

            try:
                if self.params.getboolean('fitting', 'collect_all'):
                    self.has_garbage = True
                    self.garbage = io.load_data(self.params, 'garbage')
                else:
                    self.has_garbage = False
            except Exception:
                self.has_garbage = False

            try:
                self.mua = io.load_data(self.params, 'mua')
                count = 0
                self.sizes = []
                for key in self.probe['channel_groups'].keys():
                    for item in self.probe['channel_groups'][key]['geometry'].keys():
                        if item in self.probe['channel_groups'][key]['channels']:
                            self.sizes += [len(self.mua['spiketimes']['elec_%d' % count])]
                            count += 1
                self.sizes = numpy.array(self.sizes)
                s_min = self.sizes.min()
                s_max = self.sizes.max()
                if s_max == s_min:
                    self.sizes = 30 * numpy.ones(len(self.sizes))
                else:
                    self.sizes = 30 + 170 * (self.sizes - s_min) / (s_max - s_min)

                self.has_mua = True
            except Exception:
                print_and_log(["MUA results not found!"], 'info', logger)

        self.get_data()
        self.x_position = []
        self.y_position = []
        self.label = []
        self.order = []
        for key in self.probe['channel_groups'].keys():
            for item in self.probe['channel_groups'][key]['geometry'].keys():
                if item in self.probe['channel_groups'][key]['channels']:
                    self.x_position += [self.probe['channel_groups'][key]['geometry'][item][0]]
                    self.y_position += [self.probe['channel_groups'][key]['geometry'][item][1]]
                    if len(self.probe['channel_groups'].keys()) == 1:
                        self.label += ["%d" % item]
                    else:
                        self.label += ["%s-%d" % (key, item)]

        self.points = zip(self.x_position, self.y_position)

        self.selected_points = set()
        self.inspect_points = []
        self.inspect_colors = []
        self.lasso_selector = None
        self.rect_selector = widgets.RectangleSelector(
            self.electrode_ax, onselect=self.callback_rect, button=1, drawtype='box', spancoords='data'
        )
        self.rect_selector.set_active(False)
        self.plot_electrodes()

        # Connect events
        self.ui.electrodes.mpl_connect('button_press_event', self.on_mouse_press)
        self.ui.electrodes.mpl_connect('scroll_event', self.zoom)
        self.ui.electrodes.mpl_connect('motion_notify_event', self.update_statusbar)
        self.ui.electrodes.mpl_connect('figure_leave_event', lambda event: self.statusbar.clearMessage())
        self.ui.raw_data.mpl_connect('button_press_event', self.on_mouse_press)
        self.ui.raw_data.mpl_connect('scroll_event', self.zoom)
        self.ui.raw_data.mpl_connect('motion_notify_event', self.update_statusbar)
        self.ui.raw_data.mpl_connect('figure_leave_event', lambda event: self.statusbar.clearMessage())
        self.btn_rectangle.clicked.connect(self.update_rect_selector)
        self.btn_lasso.clicked.connect(self.update_rect_selector)
        self.btn_picker.clicked.connect(self.update_rect_selector)
        self.get_time.valueChanged.connect(self.update_time)
        self.get_threshold.valueChanged.connect(self.update_threshold)
        self.show_residuals.clicked.connect(self.update_data_plot)
        self.show_labels.clicked.connect(self.update_electrode_plot)
        self.show_mua.clicked.connect(self.update_electrode_plot)
        self.show_unfitted.clicked.connect(self.update_data_plot)
        self.btn_write_threshold.clicked.connect(self.write_threshold)
        if self.show_fit:
            self.time_box.setEnabled(True)
            self.show_residuals.setEnabled(True)
            self.show_unfitted.setEnabled(True)
        else:
            self.btn_write_threshold.setEnabled(True)
            self.threshold_box.setEnabled(True)
        self.get_time.setValue(self.t_start)
        self.get_threshold.setValue(self.spike_thresh)

        if self.has_mua:
            self.show_mua.setEnabled(True)

        # Select the most central point at start
        idx = np.argmin((self.x_position - np.mean(self.x_position)) ** 2 +
                        (self.y_position - np.mean(self.y_position)) ** 2)
        self.update_inspect({idx})

    def update_threshold(self):
        self.user_threshold = self.get_threshold.value()
        self.update_data_plot()

    def update_time(self):
        if self.show_fit:
            self.t_start = min(self.maxtime, self.get_time.value())
            self.t_stop = self.t_start + 1
            if self.t_stop > self.maxtime:
                self.t_stop = self.maxtime
            self.get_data()
            self.update_data_plot()

    def write_threshold(self):
        self.params.write('detection', 'spike_thresh', '%g' % self.get_threshold.value(), preview_path=True)

    def get_data(self):
        self.chunk_size = int(self.sampling_rate*(self.t_stop - self.t_start))
        t_start = int(self.t_start*self.sampling_rate)
        self.data = self.data_file.get_snippet(t_start, self.chunk_size, nodes=self.nodes)

        if self.do_spatial_whitening:
            self.data = numpy.dot(self.data, self.spatial_whitening)
        if self.do_temporal_whitening:
            self.data = scipy.ndimage.filters.convolve1d(self.data, self.temporal_whitening, axis=0, mode='constant')

        self.time = numpy.linspace(self.t_start, self.t_stop, self.data.shape[0])

        if self.show_fit:
            try:
                self.curve = numpy.zeros((self.N_e, self.chunk_size), dtype=numpy.float32)
                limit = self.sampling_rate-self.template_shift+1
                for key in self.result['spiketimes'].keys():
                    elec  = int(key.split('_')[1])
                    lims  = (self.t_start*self.sampling_rate + self.template_shift, self.t_stop*self.sampling_rate - self.template_shift-1)
                    idx   = numpy.where((self.result['spiketimes'][key] > lims[0]) & (self.result['spiketimes'][key] < lims[1]))
                    for spike, (amp1, amp2) in zip(self.result['spiketimes'][key][idx], self.result['amplitudes'][key][idx]):
                        spike -= self.t_start*self.sampling_rate
                        tmp1 = self.templates[:, elec].toarray().reshape(self.N_e, self.N_t)
                        tmp2 = self.templates[:, elec+self.templates.shape[1]//2].toarray().reshape(self.N_e, self.N_t)
                        self.curve[:, int(spike) - self.template_shift:int(spike) + self.template_shift + 1] += amp1 * tmp1 + amp2 * tmp2
            except Exception as exception:
                print_and_log(["Unable to show fit: {}".format(exception)], 'default', logger)
                self.curve = numpy.zeros((self.N_e, int(self.sampling_rate)), dtype=numpy.float32)

            if self.has_garbage:
                self.uncollected = {}
                for key in self.garbage['gspikes'].keys():
                    elec = int(key.split('_')[1])
                    lims = (self.t_start * self.sampling_rate + self.template_shift, self.t_stop * self.sampling_rate - self.template_shift - 1)
                    idx = numpy.where((self.garbage['gspikes'][key] > lims[0]) & (self.garbage['gspikes'][key] < lims[1]))
                    all_spikes = self.garbage['gspikes'][key][idx]
                    self.uncollected[elec] = all_spikes / float(self.sampling_rate)

    def init_gui_layout(self):
        gui_fname = pkg_resources.resource_filename('circus', os.path.join('qt_GUI', 'qt_preview.ui'))
        self.ui = uic.loadUi(gui_fname, self)
        self.electrode_ax = self.ui.electrodes.axes
        self.data_x = self.ui.raw_data.axes
        if self.show_fit:
            self.ui.btn_next.clicked.connect(self.increase_time)
            self.ui.btn_prev.clicked.connect(self.decrease_time)
        else:
            self.ui.btn_next.setVisible(False)
            self.ui.btn_prev.setVisible(False)
        # Toolbar will not be displayed
        self.mpl_toolbar = NavigationToolbar(self.ui.raw_data, None)
        self.mpl_toolbar.pan()
        self.ui.show()

    def increase_time(self, event):
        _ = event  # (PyCharm code inspection)
        if self.show_fit:
            if self.t_start < self.maxtime - 1:
                self.t_start += 1
                self.t_stop += 1
                if self.t_stop > self.maxtime:
                    self.t_stop = self.maxtime

            self.get_data()
            self.get_time.setValue(self.t_start)
            self.update_data_plot()

    def decrease_time(self, event):
        _ = event  # (PyCharm code inspection)
        if self.show_fit:
            if self.t_start > self.mintime + 1:
                self.t_start -= 1
                self.t_stop -= 1
                if self.t_start < self.mintime:
                    self.t_start = self.mintime
                    self.t_stop = self.t_start + 1
                self.get_data()
                self.get_time.setValue(self.t_start)
                self.update_data_plot()

    def plot_electrodes(self):
        if not getattr(self, 'collections', None):
            # It is important to set one facecolor per point so that we can change
            # it later
            self.electrode_collection = self.electrode_ax.scatter(
                self.x_position, self.y_position, facecolor=['black' for _ in self.x_position], s=30)
            self.electrode_ax.set_xlabel('Space [um]')
            self.electrode_ax.set_xticklabels([])
            self.electrode_ax.set_ylabel('Space [um]')
            self.electrode_ax.set_yticklabels([])
            self.electrode_labels = [
                self.electrode_ax.text(x, y, l)
                for (x, y, l) in zip(self.x_position, self.y_position, self.label)
            ]
        else:
            self.electrode_collection.set_offsets(np.hstack([
                self.x_position[np.newaxis, :].T,
                self.y_position[np.newaxis, :].T
            ]))

        ax, x, y = self.electrode_ax, self.y_position, self.x_position
        ymin, ymax = min(x), max(x)
        yrange = (ymax - ymin) * 0.5 * 1.05  # stretch everything a bit
        ax.set_ylim((ymax + ymin) * 0.5 - yrange, (ymax + ymin) * 0.5 + yrange)
        xmin, xmax = min(y), max(y)
        xrange = (xmax - xmin) * 0.5 * 1.05  # stretch everything a bit
        ax.set_xlim((xmax + xmin) * 0.5 - xrange, (xmax + xmin) * 0.5 + xrange)

        self.ui.raw_data.draw_idle()

    def start_lasso_select(self, event):
        self.lasso_selector = widgets.Lasso(event.inaxes, (event.xdata, event.ydata), self.callback_lasso)
        add_or_remove = None
        if event.key == 'shift':
            add_or_remove = 'add'
        elif event.key == 'control':
            add_or_remove = 'remove'
        self.lasso_selector.add_or_remove = add_or_remove
        if event.inaxes == self.electrode_ax:
            self.lasso_selector.points = self.points

    def on_mouse_press(self, event):
        if event.inaxes == self.electrode_ax:
            if self.ui.btn_lasso.isChecked():
                # Select multiple points
                self.start_lasso_select(event)
            elif self.ui.btn_rectangle.isChecked():
                pass  # handled already by rect selector
            elif self.ui.btn_picker.isChecked():
                # Select a single point for display
                # Transform data coordinates to display coordinates
                x = self.x_position
                y = self.y_position
                data = event.inaxes.transData.transform(zip(x, y))

                # Find the closest point
                distances = ((data[:, 0] - event.x)**2 +
                             (data[:, 1] - event.y)**2)
                min_idx, min_value = np.argmin(distances), np.min(distances)
                if min_value > 50:
                    # Don't select anything if the mouse cursor is more than
                    # 50 pixels away from a point
                    selection = {}
                else:
                    selection = {min_idx}
                add_or_remove = None
                if event.key == 'shift':
                    add_or_remove = 'add'
                elif event.key == 'control':
                    add_or_remove = 'remove'
                self.update_inspect(selection, add_or_remove)
            else:
                raise AssertionError('No tool active')
        else:
            return

    def callback_lasso(self, verts):
        p = mpl.path.Path(verts)
        in_selection = p.contains_points(self.lasso_selector.points)
        indices = np.nonzero(in_selection)[0]
        self.update_inspect(indices, self.lasso_selector.add_or_remove)

    def callback_rect(self, eclick, erelease):
        xmin, xmax, ymin, ymax = eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        x, y = self.x_position, self.y_position
        in_selection = ((x >= xmin) & (x <= xmax) &
                        (y >= ymin) & (y <= ymax))
        indices = np.nonzero(in_selection)[0]
        add_or_remove = None
        if erelease.key == 'shift':
            add_or_remove = 'add'
        elif erelease.key == 'control':
            add_or_remove = 'remove'
        self.update_inspect(indices, add_or_remove)

    def update_rect_selector(self, event):
        _ = event  # (PyCharm code inspection)
        self.rect_selector.set_active(self.btn_rectangle.isChecked())
        self.ui.electrodes.draw_idle()

    def update_inspect(self, indices, add_or_remove=None):

        if MPL_VERSION:
            all_colors = mpl.colors.colorConverter.to_rgba_array([i['color'] for i in plt.rcParams['axes.prop_cycle'].__dict__['_left']])
        else:
            all_colors = colorConverter.to_rgba_array(plt.rcParams['axes.color_cycle'])

        if add_or_remove is 'add':
            indices = set(self.inspect_points) | set(indices)
        elif add_or_remove is 'remove':
            indices = set(self.inspect_points) - set(indices)

        self.inspect_points = sorted(indices)
        # We use a deterministic mapping to colors, based on their index
        self.inspect_colors = [all_colors[idx % len(all_colors)]
                               for idx in self.inspect_points]

        self.update_electrode_plot()
        self.update_data_plot()

    def update_electrode_plot(self):
        collection = self.electrode_collection

        fcolors = collection.get_facecolors()
        colorin = colorConverter.to_rgba('black', alpha=0.25)
        colorout = colorConverter.to_rgba('black')

        fcolors[:] = colorout
        for p in self.selected_points:
            fcolors[p] = colorin
        for idx, p in enumerate(self.inspect_points):
            fcolors[p] = colorConverter.to_rgba(self.inspect_colors[idx])

        if self.ui.show_mua.isChecked():
            collection.set_sizes(self.sizes)
        else:
            collection.set_sizes(30 * numpy.ones(len(fcolors)))

        for l in self.electrode_labels:
            if self.ui.show_labels.isChecked():
                l.set_visible(True)
            else:
                l.set_visible(False)

        # if self.show_mua:
        #     collection

        self.ui.electrodes.draw_idle()

    def update_data_plot(self):

        self.data_x.clear()
        indices = self.inspect_points
        if len(indices) > 0:
            yspacing = numpy.max(np.abs(self.data[:, indices]))*1.01
        else:
            yspacing = 0

        if not self.show_fit:
            for count, idx in enumerate(indices):
                data_line, = self.data_x.plot(
                    self.time, count * yspacing + self.data[:, idx], lw=1, color=self.inspect_colors[count]
                )
                thr = self.thresholds[idx] * (self.user_threshold / self.spike_thresh)
                if self.peaks_sign in ['negative', 'both']:
                    self.data_x.plot(
                        [self.t_start, self.t_stop], [-thr + count * yspacing, -thr + count * yspacing],
                        '--', color=self.inspect_colors[count], lw=2
                    )
                if self.peaks_sign in ['positive', 'both']:
                    self.data_x.plot(
                        [self.t_start, self.t_stop], [thr + count * yspacing, thr + count * yspacing],
                        '--', color=self.inspect_colors[count], lw=2
                    )

        else:

            for count, idx in enumerate(indices):

                if self.ui.show_residuals.isChecked():
                    data_line, = self.data_x.plot(
                        self.time, count * yspacing + (self.data[:, idx] - self.curve[idx, :]),
                        lw=1, color='0.5', alpha=0.5
                    )
                data_line, = self.data_x.plot(
                    self.time, count * yspacing + self.data[:, idx], lw=1, color=self.inspect_colors[count]
                )
                data_line, = self.data_x.plot(
                    self.time, count * yspacing + self.curve[idx, :], lw=1, color='k'
                )

                thr = self.thresholds[idx]
                if self.peaks_sign in ['negative', 'both']:
                    self.data_x.plot(
                        [self.t_start, self.t_stop], [-thr + count * yspacing, -thr + count * yspacing],
                        '--', color=self.inspect_colors[count], lw=2
                    )
                if self.peaks_sign in ['positive', 'both']:
                    self.data_x.plot(
                        [self.t_start, self.t_stop], [thr + count * yspacing, thr + count * yspacing],
                        '--', color=self.inspect_colors[count], lw=2
                    )

                if self.ui.show_unfitted.isChecked() and self.has_garbage:
                    for i in self.uncollected[idx]:
                        self.data_x.add_patch(plt.Rectangle((i-0.001, -self.thresholds[idx] + count * yspacing), 0.002, 2 * self.thresholds[idx], alpha=0.5, color='k'))

        self.data_x.set_yticklabels([])
        self.data_x.set_xlabel('Time [s]')
        self.data_x.set_xlim(self.t_start, self.t_stop)

        self.ui.raw_data.draw_idle()

    def update_statusbar(self, event):
        # Update information about the mouse position to the status bar
        status_bar = self.statusbar
        if event.inaxes == self.electrode_ax:
            status_bar.showMessage(u'x: %.0fμm  y: %.0fμm' % (event.xdata, event.ydata))
        elif event.inaxes == self.data_x:
            yspacing = numpy.max(np.abs(self.data)) * 1.05
            if yspacing != 0:
                row = int((event.ydata + 0.5 * yspacing) / yspacing)
            else:
                row = int(event.ydata)
            if row < 0 or row >= len(self.inspect_points):
                status_bar.clearMessage()
            else:
                time_idx = np.argmin(np.abs(self.time - event.xdata))
                start_idx = np.argmin(np.abs(self.time - self.t_start))
                rel_time_idx = time_idx - start_idx
                electrode_idx = self.inspect_points[row]
                electrode_x, electrode_y = self.points[electrode_idx]
                data = self.data[rel_time_idx, electrode_idx]
                msg = '%.2f' % data
                if self.show_fit:
                    fit = self.curve[electrode_idx, rel_time_idx]
                    msg += ' (fit: %.2f)' % fit
                msg += '  t: %.2fs ' % self.time[time_idx]
                msg += u'(electrode %d at x: %.0fμm  y: %.0fμm)' % (electrode_idx, electrode_x, electrode_y)
                status_bar.showMessage(msg)

    def zoom(self, event):
        if event.inaxes == self.electrode_ax:
            x = self.x_position
            y = self.y_position
        elif event.inaxes == self.data_x:
            x = self.time
            y = self.data[:, self.inspect_points]
        else:
            return

        ax = event.inaxes
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0]) * 0.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0]) * 0.5
        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1 / 2.0
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = 2.0
        else:
            # deal with something that should never happen
            scale_factor = 1
        # set new limits
        newxmin = np.clip(xdata - cur_xrange * scale_factor, np.min(x), np.max(x))
        newxmax = np.clip(xdata + cur_xrange * scale_factor, np.min(x), np.max(x))
        new_xrange = (newxmax - newxmin) * 0.5 * 1.05  # stretch everything a bit
        newxmin = (newxmax + newxmin) * 0.5 - new_xrange
        newxmax = (newxmax + newxmin) * 0.5 + new_xrange
        ax.set_xlim(newxmin, newxmax)

        if event.inaxes == self.electrode_ax:
            newymin = np.clip(ydata - cur_yrange * scale_factor, np.min(y), np.max(y))
            newymax = np.clip(ydata + cur_yrange * scale_factor, np.min(y), np.max(y))
            new_yrange = (newymax - newymin) * 0.5 * 1.05  # stretch everything a bit
            newymin = (newymax + newymin)*0.5 - new_yrange
            newymax = (newymax + newymin)*0.5 + new_yrange
            ax.set_ylim(newymin, newymax)
        if event.inaxes == self.electrode_ax:
            self.ui.electrodes.draw_idle()
        else:
            self.ui.raw_data.draw_idle()
