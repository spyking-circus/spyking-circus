"""Display the ISI distribution for a given template."""
import argparse
import matplotlib.pyplot as plt
import numpy as np

from circus.shared.parser import CircusParser
from circus.shared.files import load_data


# Parse arguments.
parser = argparse.ArgumentParser(description="Display the ISI distribution for a given template.")
parser.add_argument('datafile', help="data file")
parser.add_argument('-t', '--template', default=0, type=int, help="template index", dest='template_id')
parser.add_argument('-mi', '--maximum-interval', default=50.0, type=float, help="maximum interval (ms)")
parser.add_argument('-bs', '--bin-size', default=1.0, type=float, help="bin size (ms)")
args = parser.parse_args()

# Load parameters.
params = CircusParser(args.datafile)
_ = params.get_data_file()
sampling_rate = params.rate

# Load spike intervals.
results = load_data(params, 'results')
spike_times = results['spiketimes']['temp_{}'.format(args.template_id)]
spike_intervals = np.diff(spike_times) / sampling_rate * 1e+3

# Display ISI.
fig, ax = plt.subplots()
nb_bins = int(np.ceil(args.maximum_interval / args.bin_size))
maximum_interval = float(nb_bins) * args.bin_size
hist_kwargs = {
    'bins': nb_bins,
    'range': (0.0, maximum_interval),
    'density': True,
    'color': 'black',
}
ax.hist(spike_intervals, **hist_kwargs)
ax.set_xlim(0.0, maximum_interval)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 0))
ax.set_xlabel("interval (ms)")
ax.set_ylabel("probability")
ax.set_title("ISI (template {})".format(args.template_id))
fig.tight_layout()
plt.show()
