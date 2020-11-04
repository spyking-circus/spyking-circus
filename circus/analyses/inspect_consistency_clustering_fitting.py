"""Inspect consistency of fitted spike times with peak times collected during clustering."""
import argparse
import matplotlib.pyplot as plt
import numpy as np

from circus.shared.parser import CircusParser
from circus.shared.files import load_data

from circus.analyses.utils import load_clusters_data


# Parse arguments.
parser = argparse.ArgumentParser(  # noqa
    description="Inspect fitted snippets for a given template.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('datafile', help="data file")
parser.add_argument('-i', '--interval', default=3.0, type=float, help="threshold interval (in ms)")
parser.add_argument('-t', '--template', default=None, type=int, help="template identifier", dest='template_id')
args = parser.parse_args()

# Load parameters.
params = CircusParser(args.datafile)
_ = params.get_data_file()
sampling_rate = params.rate
# duration = params.data_file.duration
nb_time_steps = params.getint('detection', 'N_t')

# Load spike times.
results = load_data(params, 'results', extension='')  # TODO support other extensions?
results_data = dict()
for key in results['spiketimes'].keys():
    if key[:5] == 'temp_':
        template_key = key
        template_id = int(key[5:])
        if 'spike_times' not in results_data:
            results_data['spike_times'] = dict()
        results_data['spike_times'][template_id] = results['spiketimes'][template_key][:]

# Load peak times.
clusters_data = load_clusters_data(params, extension='')  # TODO support other extensions?


def compute_minimum_intervals(spike_times, peak_times):

    if spike_times.size > 0:
        indices = np.searchsorted(spike_times, peak_times)
        pre_spike_times = spike_times[np.maximum(indices - 1, 0)]
        post_spike_times = spike_times[np.minimum(indices, spike_times.size - 1)]
        minimum_intervals = np.minimum(
            np.abs(peak_times - pre_spike_times),
            np.abs(peak_times - post_spike_times),
        )
    else:
        minimum_intervals = np.iinfo(peak_times.dtype).max * np.ones_like(peak_times)

    return minimum_intervals


# Compute proportions of minimum intervals below the given threshold interval.
template_ids = np.array(results_data['spike_times'].keys())
proportions = np.zeros_like(template_ids, dtype=np.float)
for k, template_id in enumerate(template_ids):
    # Select spike times.
    spike_times = results_data['spike_times'][template_id]
    spike_times = np.sort(spike_times)
    # Select peak times.
    preferred_electrode = clusters_data['electrodes'][template_id]
    local_cluster = clusters_data['local_clusters'][template_id]
    clusters = clusters_data['clusters'][preferred_electrode]
    times = clusters_data['times'][preferred_electrode]
    assert local_cluster in np.unique(clusters), np.unique(clusters)
    selection = (clusters == local_cluster)
    peak_times = times[selection]
    peak_times = np.sort(peak_times)
    # Compute minimum intervals between spike times and peak times.
    minimum_intervals = compute_minimum_intervals(spike_times, peak_times)
    minimum_intervals = minimum_intervals / sampling_rate * 1e+3  # ms
    # Compute proportion.
    proportion = np.count_nonzero(minimum_intervals < args.interval) / minimum_intervals.size
    proportions[k] = proportion

# Plot proportions.
fig, ax = plt.subplots()
x = template_ids
y = proportions
scatter_kwargs = {
    's': (3 ** 2),
    'color': 'black',
}
ax.scatter(x, y, **scatter_kwargs)
axline_kwargs = {
    'color': 'black',
    'linewidth': 0.5,
}
ax.axhline(y=0.0, **axline_kwargs)
ax.axhline(y=1.0, **axline_kwargs)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("template")
ax.set_ylabel("proportion")
ax.set_title("peak times as spike times (+/- {} ms)".format(args.interval))
fig.tight_layout()


if args.template_id is not None:

    # Select spike times.
    spike_times = results_data['spike_times'][args.template_id]
    spike_times = np.sort(spike_times)

    # Select peak times.
    preferred_electrode = clusters_data['electrodes'][args.template_id]
    local_cluster = clusters_data['local_clusters'][args.template_id]
    clusters = clusters_data['clusters'][preferred_electrode]
    times = clusters_data['times'][preferred_electrode]
    assert local_cluster in np.unique(clusters), np.unique(clusters)
    selection = (clusters == local_cluster)
    peak_times = times[selection]
    peak_times = np.sort(peak_times)

    # Compute minimum intervals between spike times and peak times.
    minimum_intervals = compute_minimum_intervals(spike_times, peak_times)
    minimum_intervals = minimum_intervals / sampling_rate  # s

    # Plot PDF of these minimum intervals.
    fig, axes = plt.subplots(nrows=2, squeeze=False)
    step_kwargs = {
        'where': 'post',
        'color': 'black',
    }
    axline_kwargs = {
        'color': 'black',
        'linewidth': 0.5,
    }
    # # 1st subplot.
    ax = axes[0, 0]
    x = np.sort(minimum_intervals)  # s
    y = np.arange(1, minimum_intervals.size + 1) / minimum_intervals.size
    ax.step(x, y, **step_kwargs)
    ax.axvline(x=0.0, **axline_kwargs)
    ax.axhline(y=0.0, **axline_kwargs)
    ax.axhline(y=1.0, **axline_kwargs)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("interval (s)")
    ax.set_ylabel("probability")
    ax.set_title("spike-peak intervals PDF (template {})".format(args.template_id))
    fig.tight_layout()
    # # 2nd subplot.
    ax = axes[1, 0]
    x = np.sort(minimum_intervals) * 1e+3  # ms
    y = np.arange(1, minimum_intervals.size + 1) / minimum_intervals.size
    ax.step(x, y, **step_kwargs)
    ax.axvline(x=0.0, **axline_kwargs)
    ax.axhline(y=0.0, **axline_kwargs)
    ax.axhline(y=1.0, **axline_kwargs)
    vmin, vmax = 0.0, 2.0 * nb_time_steps / sampling_rate * 1e+3  # ms
    xmin = vmin - 0.05 * (vmax - vmin)
    xmax = vmax + 0.05 * (vmax - vmin)
    ax.set_xlim(xmin, xmax)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("interval (ms)")
    ax.set_ylabel("probability")
    fig.tight_layout()
