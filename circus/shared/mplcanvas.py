from PyQt4 import QtGui, uic, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams

class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        # Use smaller labels
        rcParams['axes.labelsize'] = 'small'
        rcParams['xtick.labelsize'] = 'small'
        rcParams['ytick.labelsize'] = 'small'
        self.axes = fig.add_axes([0.15, 0.15, 0.85, 0.85])
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setFocus()
        fig.patch.set_alpha(0)

    def resizeEvent(self, event):
        w = event.size().width()
        h = event.size().height()
        # Leave a fixed amount for the axes
        padding = 7.5*FontProperties(size=rcParams['axes.labelsize']).get_size_in_points()
        posx = padding/w
        posy = padding/h
        self.axes.set_position([posx, posy, 0.97-posx, 0.97-posy])
        super(MplCanvas, self).resizeEvent(event)
