import matplotlib
import os
if 'DISPLAY' in os.environ and os.environ['DISPLAY'] in [":0", ":1"]:
    try:
        import PyQt5
        matplotlib.use('Qt5Agg', warn=False)
    except ImportError:
        matplotlib.use('Qt4Agg', warn=False)
else:
    matplotlib.use('Agg', warn=False)

import files
import parser
import algorithms
import plot
import utils
#import gui
