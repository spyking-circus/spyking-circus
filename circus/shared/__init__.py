import matplotlib
import os
if 'DISPLAY' in os.environ and os.environ['DISPLAY'] in [":0", ":1", ":2"]:
    try:
        import PyQt5
        matplotlib.use('Qt5Agg')
    except ImportError:
        matplotlib.use('Qt4Agg')
else:
    matplotlib.use('Agg')

# import circus.shared.files as files
# import circus.shared.parser as parser
# import circus.shared.algorithms as algorithms
# import circus.shared.plot as plot
# import circus.shared.utils as utils
# import circus.shared.gui as gui
