import six, h5py, pkg_resources

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.colors import colorConverter
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors

from utils import *
from algorithms import slice_templates, slice_clusters

class ToggleButton(widgets.Button):

    def __init__(self, *args, **kwds):
        self.toggle_group = kwds.pop('toggle_group')
        super(ToggleButton, self).__init__(*args, **kwds)
        self.toggled = False
        [i.set_color('black') for i in self.ax.spines.itervalues()]
        self.ax.set_frame_on(False)

    def _release(self, event):
        if self.ignore(event):
            return
        if event.canvas.mouse_grabber != self.ax:
            return
        event.canvas.release_mouse(self.ax)
        if not self.eventson:
            return
        if event.inaxes != self.ax:
            return
        if self.toggled:
            return
        self.update_toggle(True)

        for cid, func in six.iteritems(self.observers):
            func(event)

    def update_toggle(self, new_toggle):
        self.toggled = new_toggle
        if self.toggled:
            c = self.hovercolor
            frame = True
            # unselect all other buttons
            for b in self.toggle_group:
                if b is not self:
                    b.update_toggle(False)
        else:
            c = self.color
            frame = False

        self.ax.set_axis_bgcolor(c)
        self.ax.set_frame_on(frame)
        if self.drawon:
            self.ax.figure.canvas.draw()

    def _motion(self, event):
        pass

class SymmetricVCursor(widgets.AxesWidget):
    '''Variant of matplotlib.widgets.Cursor, drawing two symmetric vertical
    lines at -x and x'''
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


class MergeGUI(object):

    def __init__(self, comm, params):

        self.comm       = comm
        self.params     = params
        sampling_rate   = params.getint('data', 'sampling_rate')
        self.file_out_suff = params.get('data', 'file_out_suff')
        self.cc_overlap = params.getfloat('merging', 'cc_overlap')
        self.cc_bin     = params.getfloat('merging', 'cc_bin')
        
        self.bin_size   = int(self.cc_bin * sampling_rate * 1e-3)
        self.max_delay  = 50

        templates       = io.load_data(params, 'templates')
        self.clusters   = io.load_data(params, 'clusters')
        self.result     = io.load_data(params, 'results')
        self.overlap    = h5py.File(self.file_out_suff + '.templates.hdf5', libver='latest').get('maxoverlap')[:]
        self.shape      = templates.shape
        self.indices    = numpy.arange(self.shape[2]/2)
        self.overlap   /= self.shape[0] * self.shape[1]
        self.all_merges = numpy.zeros((0, 2), dtype=numpy.int32)
        self.mpi_wait   = numpy.array([0], dtype=numpy.int32)

        if self.comm.rank > 0:
            self.listen()

        self.cmap = plt.get_cmap('winter')
        self.init_gui_layout()
        self.fig = self.score_ax1.figure
        # Remove all buttons from the standard toolbar
        toolbar = self.fig.canvas.toolbar
        for action in toolbar.actions():
            toolbar.removeAction(action)
        
        self.generate_data()
        self.selected_points = set()
        self.inspect_points = []
        self.inspect_colors = []
        self.lasso_selector = None
        self.rect_selectors = [widgets.RectangleSelector(ax,
                                                         onselect=self.callback_rect,
                                                         button=1,
                                                         drawtype='box',
                                                         spancoords='data')
                               for ax in [self.score_ax1, self.score_ax2, self.score_ax3]]
        for selector in self.rect_selectors:
            selector.set_active(False)
        self.pick_button.update_toggle(True)
        self.lag_selector = SymmetricVCursor(self.data_ax, color='blue')
        self.line_lag1 = self.data_ax.axvline(self.data_ax.get_ybound()[0],
                                              color='black')
        self.line_lag2 = self.data_ax.axvline(self.data_ax.get_ybound()[0],
                                              color='black')
        self.update_lag(5)
        self.plot_data()

        # Connect events
        self.fig.canvas.mpl_connect('scroll_event', self.zoom)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('close_event', self.handle_close)
        self.rect_button.on_clicked(self.update_rect_selector)
        self.lasso_button.on_clicked(self.update_rect_selector)
        self.pick_button.on_clicked(self.update_rect_selector)
        self.add_button.on_clicked(self.add_to_selection)
        self.remove_button.on_clicked(self.remove_selection)
        self.sort_order.on_clicked(self.update_data_sort_order)
        self.merge_button.on_clicked(self.do_merge)
        self.finalize_button.on_clicked(self.finalize)
        self.score_ax1.format_coord = lambda x, y: 'template similarity: %.2f  cross-correlation metric %.2f' % (x, y)
        self.score_ax2.format_coord = lambda x, y: 'normalized cross-correlation metric: %.2f  cross-correlation metric %.2f' % (x, y)
        self.score_ax3.format_coord = lambda x, y: 'template similarity: %.2f  normalized cross-correlation metric %.2f' % (x, y)
        self.data_ax.format_coord = self.data_tooltip
        # Select the best point at start
        idx = np.argmax(self.score_y)
        self.update_inspect({idx})
            

    def listen(self):

        while self.mpi_wait[0] == 0:
            self.generate_data()

        if self.mpi_wait[0] == 1:
            self.finalize(None)
        elif self.mpi_wait[0] == 2:
            sys.exit(0)

    def handle_close(self, event):
        if self.comm.rank == 0:
            self.mpi_wait = self.comm.bcast(numpy.array([2], dtype=numpy.int32), root=0)
        sys.exit(0)


    def init_gui_layout(self):
        gs = gridspec.GridSpec(15, 4, width_ratios=[2, 2, 1, 4])
        # TOOLBAR
        buttons_gs = gridspec.GridSpecFromSubplotSpec(1, 3,
                                                      subplot_spec=gs[0, 0])
        lasso_button_ax = plt.subplot(buttons_gs[0, 0])
        rect_button_ax = plt.subplot(buttons_gs[0, 1])
        pick_button_ax = plt.subplot(buttons_gs[0, 2])
        self.toggle_group = []
        pick_icon  = pkg_resources.resource_filename('circus', os.path.join('icons', 'gimp-tool-color-picker.png'))
        lasso_icon = pkg_resources.resource_filename('circus', os.path.join('icons', 'gimp-tool-free-select.png'))
        rect_icon  = pkg_resources.resource_filename('circus', os.path.join('icons', 'gimp-tool-rect-select.png'))
        self.lasso_button = ToggleButton(lasso_button_ax, '',
                                         image=mpl.image.imread(lasso_icon),
                                         toggle_group=self.toggle_group)
        self.rect_button = ToggleButton(rect_button_ax, '',
                                        image=mpl.image.imread(rect_icon),
                                        toggle_group=self.toggle_group)
        self.pick_button = ToggleButton(pick_button_ax, '',
                                        image=mpl.image.imread(pick_icon),
                                        toggle_group=self.toggle_group)
        self.toggle_group.extend([self.lasso_button,
                                  self.rect_button,
                                  self.pick_button])
        self.score_ax1 = plt.subplot(gs[1:8, 0])
        self.score_ax2 = plt.subplot(gs[1:8, 1])
        self.score_ax3 = plt.subplot(gs[8:, 0])
        self.detail_ax = plt.subplot(gs[1:5, 3])
        self.data_ax = plt.subplot(gs[5:13, 3])
        sort_order_ax = plt.subplot(gs[14, 3])
        sort_order_ax.set_axis_bgcolor('none')
        self.sort_order = widgets.RadioButtons(sort_order_ax, ('template similarity',
                                                               'cross-correlation',
                                                               'normalized cross-correlation'))
        self.current_order = 'template similarity'
        add_button_ax      = plt.subplot(gs[7, 2])
        self.add_button    = widgets.Button(add_button_ax, 'Select')
        remove_button_ax   = plt.subplot(gs[8, 2])
        self.remove_button = widgets.Button(remove_button_ax, 'Unselect')
        merge_button_ax    = plt.subplot(gs[9, 2])
        self.merge_button  = widgets.Button(merge_button_ax, 'Merge')
        finalize_button_ax = plt.subplot(gs[14, 2])
        self.finalize_button = widgets.Button(finalize_button_ax, 'Finalize')


    def generate_data(self):

        def reversed_corr(spike_1, spike_2, max_delay):

            size    = 2*max_delay+1
            x_cc    = numpy.zeros(size, dtype=numpy.float32)
            y_cc    = numpy.copy(x_cc)
            if (len(spike_1) > 0) and (len(spike_2) > 0):
                t1b     = numpy.unique(numpy.round(spike_1/self.bin_size))
                t2b     = numpy.unique(numpy.round(spike_2/self.bin_size))
                t2b_inv = t2b[-1] + t2b[0] - t2b
                for d in xrange(size):
                    x_cc[d] += len(numpy.intersect1d(t1b, t2b + d - max_delay, assume_unique=True))
                    y_cc[d] += len(numpy.intersect1d(t1b, t2b_inv + d - max_delay, assume_unique=True))
            return x_cc, y_cc

        self.raw_lags    = numpy.linspace(-self.max_delay*self.cc_bin, self.max_delay*self.cc_bin, 2*self.max_delay+1)
        
        self.mpi_wait    = self.comm.bcast(self.mpi_wait, root=0)
        
        if self.mpi_wait[0] > 0:
            return

        self.indices     = self.comm.bcast(self.indices, root=0)

        real_indices     = numpy.unique(self.indices)
        sub_real_indices = real_indices[numpy.arange(comm.rank, len(real_indices), comm.size)]
        
        n_pairs          = len(sub_real_indices)*(len(real_indices) - 1)/2.
        n_size           = 2*self.max_delay + 1

        self.raw_data    = numpy.zeros((0, n_size), dtype=numpy.float32)
        self.raw_control = numpy.zeros((0, n_size), dtype=numpy.float32)
        self.pairs       = numpy.zeros((0, 2), dtype=numpy.int32)

        for temp_id1 in sub_real_indices:
            for temp_id2 in real_indices[real_indices > temp_id1]:
                if self.overlap[temp_id1, temp_id2] >= self.cc_overlap:
                    spikes1 = self.result['spiketimes']['temp_' + str(temp_id1)]
                    spikes2 = self.result['spiketimes']['temp_' + str(temp_id2)]
                    a, b    = reversed_corr(spikes1, spikes2, self.max_delay)
                    self.raw_data    = numpy.vstack((self.raw_data, a))
                    self.raw_control = numpy.vstack((self.raw_control, b))
                    self.pairs       = numpy.vstack((self.pairs, numpy.array([temp_id1, temp_id2], dtype=numpy.int32)))
        
        self.pairs       = gather_array(self.pairs, self.comm, 0, 1, dtype='int32')
        self.raw_control = gather_array(self.raw_control, self.comm, 0, 1)
        self.raw_data    = gather_array(self.raw_data, self.comm, 0, 1)
        self.sort_idcs   = numpy.arange(len(self.pairs))
        
    def calc_scores(self, lag):
        data    = self.raw_data[:, abs(self.raw_lags) <= lag]
        control = self.raw_control[:, abs(self.raw_lags) <= lag]
        norm_factor = (control.mean(1) + data.mean(1) + 1)[:, np.newaxis]
        score  = ((control - data)/norm_factor).mean(axis=1)
        score2 = (control - data).mean(axis=1)
        score3 = self.overlap[self.pairs[:, 0], self.pairs[:, 1]]
        return score3, score, score2

    def plot_scores(self):
        # Left: Scores
        if not getattr(self, 'collections', None):
            # It is important to set one facecolor per point so that we can change
            # it later
            self.collections = []
            for ax, x, y in [(self.score_ax1, self.score_x, self.score_y),
                             (self.score_ax2, self.score_z, self.score_y),
                             (self.score_ax3, self.score_x, self.score_z)]:
                self.collections.append(ax.scatter(x, y,
                                                   facecolor=['black' for _ in x]))
            self.score_ax1.set_ylabel('cross-correlation metric')
            self.score_ax1.set_xticklabels([])
            self.score_ax2.set_xlabel('normalized cross-correlation metric')
            self.score_ax2.set_yticklabels([])
            self.score_ax3.set_xlabel('template similarity')
            self.score_ax3.set_ylabel('normalized cross-correlation metric')
        else:
            for collection, (x, y) in zip(self.collections, [(self.score_x, self.score_y),
                                                                 (self.score_z, self.score_y),
                                                                 (self.score_x, self.score_z)]):
                collection.set_offsets(np.hstack([x[np.newaxis, :].T,
                                                  y[np.newaxis, :].T]))
        self.score_ax1.set_ylim(min(self.score_y)-0.05, max(self.score_y)+0.05)
        self.score_ax1.set_xlim(min(self.score_x)-0.05, max(self.score_x)+0.05)
        self.score_ax2.set_ylim(min(self.score_y)-0.05, max(self.score_y)+0.05)
        self.score_ax2.set_xlim(min(self.score_z)-0.05, max(self.score_z)+0.05)
        self.score_ax3.set_ylim(min(self.score_z)-0.05, max(self.score_z)+0.05)
        self.score_ax3.set_xlim(min(self.score_x)-0.05, max(self.score_x)+0.05)

    def plot_data(self):
        # Right: raw data
        all_raw_data = self.raw_data/(1 + self.raw_data.mean(1)[:, np.newaxis])
        cmax         = 0.1*all_raw_data.max()
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
        self.data_image.set_clim(0, cmax)
        #self.inspect_markers, = self.data_ax.plot([], [], 'bo',
        #                                          clip_on=False, ms=10)
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

    def data_tooltip(self, x, y):
        row = int(y)
        if row >= 0 and row < len(self.raw_data):
            all_raw_data = self.raw_data/(1 + self.raw_data.mean(axis=1)[:, np.newaxis])
            data_idx     = self.sort_idcs[row]
            lag_diff     = np.abs(x - self.raw_lags)
            nearest_lag_idx = np.argmin(lag_diff)
            nearest_lag = self.raw_lags[nearest_lag_idx]
            value = all_raw_data[data_idx, nearest_lag_idx]
            return ('%.2f - lag: %.2fms (template similarity: %.2f  '
                    'cross-correlation metric %.2f)') % (value, nearest_lag,
                                                         self.score_x[data_idx],
                                                         self.score_y[data_idx])
        else:
            return ''

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

        #score_x, score_y = self.
        self.score_ax = eclick.inaxes

        if self.score_ax == self.score_ax1:
            score_x, score_y = self.score_x, self.score_y
        elif self.score_ax == self.score_ax2:
            score_x, score_y = self.score_z, self.score_y
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
        self.update_inspect(indices, add_or_remove)

    def zoom(self, event):
        # only zoom in the score plot
        if event.inaxes not in [self.score_ax1, self.score_ax2, self.score_ax3]:
            return
        # get the current x and y limits
        self.score_ax = event.inaxes

        cur_xlim = self.score_ax.get_xlim()
        cur_ylim = self.score_ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/2.0
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = 2.0
        else:
            # deal with something that should never happen
            scale_factor = 1
            print event.button
        # set new limits
        newxmin = np.clip(xdata - cur_xrange*scale_factor, np.min(self.score_x)-0.05, np.max(self.score_x)+0.05)
        newxmax = np.clip(xdata + cur_xrange*scale_factor, np.min(self.score_x)-0.05, np.max(self.score_x)+0.05)
        newymin = np.clip(ydata - cur_yrange*scale_factor, -0.025, 1.025)
        newymax = np.clip(ydata + cur_yrange*scale_factor, -0.025, 1.025)
        self.score_ax.set_xlim(newxmin, newxmax)
        self.score_ax.set_ylim(newymin, newymax)
        self.fig.canvas.draw_idle()

    def update_lag(self, lag):
        actual_lag = self.raw_lags[np.argmin(np.abs(self.raw_lags - lag))]
        self.use_lag = actual_lag
        self.score_x, self.score_y, self.score_z = self.calc_scores(lag=self.use_lag)
        self.points = [zip(self.score_x, self.score_y),
                       zip(self.score_z, self.score_y),
                       zip(self.score_x, self.score_z)]
        self.line_lag1.set_xdata((lag, lag))
        self.line_lag2.set_xdata((-lag, -lag))
        self.data_ax.set_xlabel('lag (ms) -- cutoff: %.2fms' % self.use_lag)
        self.plot_scores()  # will also trigger a draw

    def update_rect_selector(self, event):
        for selector in self.rect_selectors:
            selector.set_active(self.rect_button.toggled)
        self.fig.canvas.draw_idle()

    def update_detail_plot(self):
        self.detail_ax.clear()
        indices = self.inspect_points
        all_raw_data    = self.raw_data/(1 + self.raw_data.mean(1)[:, np.newaxis])
        all_raw_control = self.raw_control/(1 + self.raw_control.mean(1)[:, np.newaxis])

        for count, idx in enumerate(indices):
            data_line, = self.detail_ax.plot(self.raw_lags,
                                             all_raw_data[idx, :].T, lw=2, color=self.inspect_colors[count])
            self.detail_ax.plot(self.raw_lags, all_raw_control[idx, :].T, ':',
                                color=self.inspect_colors[count], lw=2)
        self.detail_ax.set_ylim(0, 3)
        self.detail_ax.set_xticks(self.data_ax.get_xticks())
        self.detail_ax.set_xticklabels([])

    def update_sort_idcs(self):
        # The selected points are sorted before all the other points -- an easy
        # way to achieve this is to add the maximum score to their score
        if self.current_order == 'template similarity':
            score = self.score_x
        elif self.current_order == 'cross-correlation':
            score = self.score_y
        elif self.current_order == 'normalized cross-correlation':
            score = self.score_z
        else:
            raise AssertionError(self.current_order)
        score = score.copy()
        if len(self.selected_points):
            score[np.array(sorted(self.selected_points))] += score.max()

        self.sort_idcs = np.argsort(score)

    def update_data_plot(self):
        reverse_sort = np.argsort(self.sort_idcs)

        if len(self.inspect_points):
            inspect = reverse_sort[np.array(sorted(self.inspect_points))]
            #self.inspect_markers.set_xdata(np.ones(len(inspect))*self.raw_lags[-1])
            #self.inspect_markers.set_ydata(inspect+0.5)
            
            data = numpy.vstack((np.ones(len(inspect))*(2*self.raw_lags[-1]-self.raw_lags[-2]), inspect+0.5)).T
            self.inspect_markers.set_offsets(data)
            self.inspect_markers.set_color(self.inspect_colors)
        else:
            #self.inspect_markers.set_xdata([])
            #self.inspect_markers.set_ydata([])
            self.inspect_markers.set_offsets([])
            self.inspect_markers.set_color([])

        self.fig.canvas.draw_idle()

    def update_data_sort_order(self, new_sort_order=None):
        if new_sort_order is not None:
            self.current_order = new_sort_order
        self.update_sort_idcs()
        self.data_image.set_extent((self.raw_lags[0], self.raw_lags[-1],
                            0, len(self.sort_idcs)))
        self.data_ax.set_ylim(0, len(self.sort_idcs))
        all_raw_data  = self.raw_data
        all_raw_data /= (1 + self.raw_data.mean(1)[:, np.newaxis])
        cmax          = 0.1*all_raw_data.max()
        all_raw_data  = all_raw_data[self.sort_idcs, :]
        self.data_image.set_data(all_raw_data)
        self.data_image.set_clim(0, cmax)
        self.data_selection.set_y(len(self.sort_idcs)-len(self.selected_points))
        self.data_selection.set_height(len(self.selected_points))
        self.update_data_plot()

    def update_score_plot(self):
        for collection in self.collections:
            fcolors = collection.get_facecolors()
            colorin = colorConverter.to_rgba('black', alpha=0.25)
            colorout = colorConverter.to_rgba('black')

            fcolors[:] = colorout
            for p in self.selected_points:
                fcolors[p] = colorin
            for idx, p in enumerate(self.inspect_points):
                fcolors[p] = colorConverter.to_rgba(self.inspect_colors[idx])

    def update_inspect(self, indices, add_or_remove=None):
        all_colors = colorConverter.to_rgba_array(plt.rcParams['axes.color_cycle'])

        if add_or_remove is 'add':
            indices = set(self.inspect_points) | set(indices)
        elif add_or_remove is 'remove':
            indices = set(self.inspect_points) - set(indices)

        self.inspect_points = sorted(indices)
        # We use a deterministic mapping to colors, based on their index
        self.inspect_colors = [all_colors[idx % len(all_colors)]
                               for idx in self.inspect_points]

        self.update_score_plot()
        self.update_detail_plot()
        self.update_data_plot()

    def update_selection(self, indices, add_or_remove=None):
        if add_or_remove is None:
            self.selected_points.clear()

        if add_or_remove == 'remove':
            self.selected_points.difference_update(set(indices))
        else:
            self.selected_points.update(set(indices))

        self.update_score_plot()
        self.update_detail_plot()
        self.update_data_sort_order()

    def add_to_selection(self, event):
        to_add = set(self.inspect_points)
        self.inspect_points = set()
        self.update_selection(to_add, add_or_remove='add')

    def remove_selection(self, event):
        self.inspect_points = set()
        self.update_selection(self.selected_points, add_or_remove='remove')

    def on_mouse_press(self, event):
        if event.inaxes in [self.score_ax1, self.score_ax2, self.score_ax3]:
            if self.lasso_button.toggled:
                # Select multiple points
                self.start_lasso_select(event)
            elif self.rect_button.toggled:
                pass  # handled already by rect selector
            elif self.pick_button.toggled:
                # Select a single point for display
                # Find the closest point
                if event.inaxes == self.score_ax1:
                    x = self.score_x
                    y = self.score_y
                elif event.inaxes == self.score_ax2:
                    x = self.score_z
                    y = self.score_y
                elif event.inaxes == self.score_ax3:
                    x = self.score_x
                    y = self.score_z
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
                self.update_inspect(selection, add_or_remove)
            else:
                raise AssertionError('No tool active')
        elif event.inaxes == self.data_ax:
            # Update lag
            self.update_lag(abs(event.xdata))
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

    def do_merge(self, event):
        # This simply removes the data points for now
        print 'Data indices to merge: ', sorted(self.selected_points)
        
        for pair in self.pairs[list(self.selected_points), :]:

            one_merge = [self.indices[pair[0]], self.indices[pair[1]]]

            elec_ic1  = self.clusters['electrodes'][one_merge[0]]
            elec_ic2  = self.clusters['electrodes'][one_merge[1]]
            nic1      = one_merge[0] - numpy.where(self.clusters['electrodes'] == elec_ic1)[0][0]
            nic2      = one_merge[1] - numpy.where(self.clusters['electrodes'] == elec_ic2)[0][0]
            mask1     = self.clusters['clusters_' + str(elec_ic1)] > -1
            mask2     = self.clusters['clusters_' + str(elec_ic2)] > -1
            tmp1      = numpy.unique(self.clusters['clusters_' + str(elec_ic1)][mask1])
            tmp2      = numpy.unique(self.clusters['clusters_' + str(elec_ic2)][mask2])
            elements1 = numpy.where(self.clusters['clusters_' + str(elec_ic1)] == tmp1[nic1])[0]
            elements2 = numpy.where(self.clusters['clusters_' + str(elec_ic2)] == tmp2[nic2])[0]

            if len(elements1) > len(elements2):
                to_remove = one_merge[1]
                to_keep   = one_merge[0]
                elec      = elec_ic2
                elements  = elements2
            else:
                to_remove = one_merge[0]
                to_keep   = one_merge[1]
                elec      = elec_ic1
                elements  = elements1

            if to_keep != to_remove:
                key        = 'temp_' + str(to_keep)
                key2       = 'temp_' + str(to_remove)
                spikes     = self.result['spiketimes'][key2]
                amplitudes = self.result['amplitudes'][key2]
                n1, n2     = len(self.result['amplitudes'][key2]), len(self.result['amplitudes'][key])
                self.result['amplitudes'][key] = numpy.vstack((self.result['amplitudes'][key].reshape(n2, 2), amplitudes.reshape(n1, 2)))
                self.result['spiketimes'][key] = numpy.concatenate((self.result['spiketimes'][key], spikes))
                idx                            = numpy.argsort(self.result['spiketimes'][key])
                self.result['spiketimes'][key] = self.result['spiketimes'][key][idx]
                self.result['amplitudes'][key] = self.result['amplitudes'][key][idx]
                self.result['spiketimes'].pop(key2)
                self.result['amplitudes'].pop(key2)
            
                self.all_merges   = numpy.vstack((self.all_merges, [self.indices[to_keep], self.indices[to_remove]]))
                idx               = numpy.where(self.indices == to_remove)[0]
                self.indices[idx] = self.indices[to_keep]
        
        self.generate_data()
        self.collections = None
        self.selected_points = set()
        self.score_ax1.clear()
        self.score_ax2.clear()
        self.score_ax3.clear()
        self.update_lag(self.use_lag)
        self.update_data_sort_order()
        self.update_detail_plot()

    def finalize(self, event):

        if comm.rank == 0:
            self.mpi_wait = self.comm.bcast(numpy.array([1], dtype=numpy.int32), root=0)

        comm.Barrier()
        self.all_merges = self.comm.bcast(self.all_merges, root=0)
        
        slice_templates(self.comm, self.params, to_merge=self.all_merges, extension='-merged')
        slice_clusters(self.comm, self.params, self.clusters, to_merge=self.all_merges, extension='-merged')

        if self.comm.rank == 0:
            new_result = {'spiketimes' : {}, 'amplitudes' : {}} 
            for count, temp_id in enumerate(numpy.unique(self.indices)):
                key_before = 'temp_' + str(temp_id)
                key_after  = 'temp_' + str(count)
                new_result['spiketimes'][key_after] = self.result['spiketimes'].pop(key_before)
                new_result['amplitudes'][key_after] = self.result['amplitudes'].pop(key_before)
            
            keys = ['spiketimes', 'amplitudes']
            mydata = h5py.File(self.file_out_suff + '.result-merged.hdf5', 'w', libver='latest')
            for key in keys:
                mydata.create_group(key)
                for temp in new_result[key].keys():
                    tmp_path = '%s/%s' %(key, temp)
                    mydata.create_dataset(tmp_path, data=new_result[key][temp])
            mydata.close()
            
            mydata  = h5py.File(self.file_out_suff + '.templates-merged.hdf5', 'r+', libver='latest')
            to_keep = numpy.unique(self.indices)
            maxoverlaps = mydata.create_dataset('maxoverlap', shape=(len(to_keep), len(to_keep)), dtype=numpy.float32)
            for c, i in enumerate(to_keep):
                maxoverlaps[c, :] = self.overlap[i, to_keep]*self.shape[0] * self.shape[1]
            mydata.close()

        sys.exit(0)