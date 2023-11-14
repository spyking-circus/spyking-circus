"""Inspect refractory period violations."""
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from circus.shared.parser import CircusParser
from circus.shared.files import load_data


# Parse arguments.
parser = argparse.ArgumentParser(description="Inspect refractory period violations.")
parser.add_argument('datafile', help="data file")
parser.add_argument('-e', '--extension', default=None, help="data extension")
parser.add_argument('-t', '--template', default=0, type=int, help="template index", dest='template_id')
parser.add_argument('-rp', '--refractory-period', default=3.0, type=float, help="refractory period (in ms)")
args = parser.parse_args()

# Adjust arguments.
if args.extension is None:
    args.extension = ""
else:
    args.extension = "-" + args.extension
args.refractory_period *= 1e-3  # ms to s

# Load parameters.
params = CircusParser(args.datafile)
_ = params.get_data_file()
sampling_rate = params.rate
duration = params.data_file.duration

# Load spike times and amplitudes.
results = load_data(params, 'results', extension=args.extension)
template_key = 'temp_{}'.format(args.template_id)
spike_times = results['spiketimes'][template_key] / sampling_rate  # s
amplitudes = results['amplitudes'][template_key][:, 0]

# Find RPVs.
nb_spike_times = spike_times.size
spike_intervals = np.diff(spike_times)
are_pre_violations = np.zeros(nb_spike_times, dtype=bool)
are_pre_violations[:-1] = (spike_intervals < args.refractory_period)
are_post_violations = np.zeros(nb_spike_times, dtype=bool)
are_post_violations[+1:] = (spike_intervals < args.refractory_period)

# Plot amplitudes.
fig, ax = plt.subplots()
c = np.array([mcolors.to_rgb('tab:gray') for _ in range(0, nb_spike_times)])
c[are_pre_violations] = mcolors.to_rgb('tab:blue')
c[are_post_violations] = mcolors.to_rgb('tab:orange')
c[are_pre_violations & are_post_violations] = mcolors.to_rgb('tab:red')
scatter_kwargs = {
    's': 3 ** 2,
    'c': c,
    # 'alpha': 0.5,
    # 'edgecolors': 'none',
}
ax.scatter(spike_times, amplitudes, **scatter_kwargs)
axline_kwargs = {
    'color': 'black',
    'linewidth': 0.5,
}
ax.axvline(x=0.0, **axline_kwargs)
ax.axvline(x=(duration / sampling_rate), **axline_kwargs)
ax.axhline(y=0.0, **axline_kwargs)
ax.axhline(y=1.0, **axline_kwargs)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("time (s)")
ax.set_ylabel("amplitude")
ax.set_title("amplitudes (template {})".format(args.template_id))
fig.tight_layout()
# plt.show()
