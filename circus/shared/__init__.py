import matplotlib
import os
if 'DISPLAY' in os.environ and os.environ['DISPLAY'] == ":0":
    matplotlib.use('Qt4Agg', warn=False)
else:
    matplotlib.use('Agg', warn=False)

import files
import parser
import algorithms
import plot
import utils
#import gui
