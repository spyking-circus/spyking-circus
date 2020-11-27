"""Inspect consistency of peak times collected during clustering with fitted spike times during fitting."""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from circus.shared.parser import CircusParser
from circus.shared.files import load_data

from circus.analyses.utils import load_clusters_data
from circus.analyses.utils import load_templates, load_template, plot_template
from circus.analyses.utils import load_snippets, plot_snippets, plot_snippet


# Parse arguments.
parser = argparse.ArgumentParser(  # noqa
    description="Inspect consistency of peak times collected during clustering with fitted spike times during fitting.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('datafile', help="data file")
parser.add_argument('-i', '--interval', default=3.0, type=float, help="threshold interval (in ms)")
parser.add_argument('-t', '--template', default=None, type=int, help="template identifier", dest='template_id')
# parser.add_argument('-s', '--size', choices=['amplitude', 'norm'], help="marker size")
parser.add_argument('-c', '--color', choices=['amplitude', 'norm'], help="marker color")
parser.add_argument('-n', '--nb-snippets-max', default=20, type=int, help="maximum number of snippets")
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


# Check if saved local clusters are coherent with the recomputed ones.
electrodes = clusters_data['electrodes']
local_clusters = clusters_data['local_clusters']
clusters = clusters_data['clusters']
# # Recompute local clusters based on the order of the electrodes.
nb_templates = electrodes.size
template_ids = np.arange(0, nb_templates)
local_clusters_bis = [None for template_id in template_ids]
current_local_cluster_ids = dict([
    (electrode, 0)
    for electrode in np.unique(electrodes)
])
for k, electrode in enumerate(electrodes):
    local_cluster_id = current_local_cluster_ids[electrode]
    local_clusters_bis[k] = np.unique(clusters[electrode][clusters[electrode] > - 1])[local_cluster_id]
    current_local_cluster_ids[electrode] += 1
# # Create data frame.
data = {
    'template_id': template_ids,
    'electrode': electrodes,
    'local_cluster': local_clusters,
    'local_cluster_bis': local_clusters_bis,
}
df = pd.DataFrame.from_dict(data)
df.set_index('template_id')
# # Check coherence between saved and recomputed local clusters.
selection = pd.Series(df['local_cluster'] != df['local_cluster_bis'])
if selection.any():
    pd.set_option('display.max_rows', 100)
    print(df[selection])
    raise UserWarning("Found a mismatch between saved and recomputed local clusters.")


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
    assert local_cluster in np.unique(clusters), (local_cluster, np.unique(clusters))
    selection = (clusters == local_cluster)
    peak_times = times[selection]
    peak_times = np.sort(peak_times)
    # Compute minimum intervals between spike times and peak times.
    minimum_intervals = compute_minimum_intervals(spike_times, peak_times)
    minimum_intervals = minimum_intervals / sampling_rate * 1e+3  # ms
    # Compute proportion.
    proportion = np.count_nonzero(minimum_intervals < args.interval) / minimum_intervals.size
    proportions[k] = proportion

templates = load_templates(params, extension='')  # TODO support other extensions?
# nb_time_steps, nb_channels, nb_templates = templates.shape
template_amplitudes = np.zeros_like(template_ids, dtype=np.float)
template_norms = np.zeros_like(template_ids, dtype=np.float)
for k, template_id in enumerate(template_ids):
    template_amplitudes[k] = np.max(np.abs(templates[:, :, template_id]))
    template_norms[k] = np.linalg.norm(templates[:, :, template_id])

# Plot proportions.
fig, ax = plt.subplots()
x = template_ids
y = proportions
scatter_kwargs = {
    's': (3 ** 2),
}
if args.color is None:
    scatter_kwargs['color'] = 'black'
elif args.color == 'amplitude':
    scatter_kwargs['c'] = template_amplitudes
elif args.color == 'norm':
    scatter_kwargs['c'] = template_norms
pc = ax.scatter(x, y, **scatter_kwargs)
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
if args.color is not None:
    cb = fig.colorbar(pc, ax=ax)
    cb.set_label("{}".format(args.color))
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

    assert minimum_intervals.size == peak_times.size, (minimum_intervals.size, peak_times.size)

    # Select randomly fitted peak times.
    fitted_peak_ids = np.where(minimum_intervals < args.interval)[0]
    selected_fitted_peak_ids = np.random.choice(
        fitted_peak_ids, size=min(fitted_peak_ids.size, args.nb_snippets_max), replace=False
    )
    selected_fitted_peak_time_steps = peak_times[selected_fitted_peak_ids]
    selected_fitted_snippets = load_snippets(selected_fitted_peak_time_steps, params)
    # Select randomly not fitted peak times.
    not_fitted_peak_ids = np.where(minimum_intervals >= args.interval)[0]
    selected_not_fitted_peak_ids = np.random.choice(
        not_fitted_peak_ids, size=min(not_fitted_peak_ids.size, args.nb_snippets_max), replace=False
    )
    selected_not_fitted_time_steps = peak_times[selected_not_fitted_peak_ids]
    selected_not_fitted_snippets = load_snippets(selected_not_fitted_time_steps, params)

    # Load template.
    template = load_template(args.template_id, params, extension='')   # TODO support other extensions?

    # Plot figure.
    fig, axes = plt.subplots(ncols=2, squeeze=False, sharex='all', sharey='all', figsize=(4.0 * 1.6, 2.0 * 1.6))
    kwargs = dict(
        vmin=min(np.min(selected_fitted_snippets), np.min(selected_not_fitted_snippets)),
        vmax=max(np.max(selected_fitted_snippets), np.max(selected_not_fitted_snippets)),
    )
    # # 1st subplot (fitted snippets).
    ax = axes[0, 0]
    plot_snippets(ax, selected_fitted_snippets, params, color='tab:gray', **kwargs)
    plot_snippet(ax, np.median(selected_fitted_snippets, axis=0), params, color='black', label="median", **kwargs)
    plot_template(ax, template, params, color='tab:blue', label="template", **kwargs)
    ax.set_xticks([])
    ax.legend()
    ax.set_title("fitted clustering snippets")
    # # 2nd subplot (not fitted snippets).
    ax = axes[0, 1]
    plot_snippets(ax, selected_not_fitted_snippets, params, color='tab:gray', **kwargs)
    plot_snippet(ax, np.median(selected_not_fitted_snippets, axis=0), params, color='black', label="median", **kwargs)
    plot_template(ax, template, params, color='tab:blue', label="template", **kwargs)
    ax.set_yticks([])
    ax.legend()
    ax.set_title("not fitted clustering snippets")
    fig.tight_layout()


if args.template_id is not None:

    # Select peak times.
    preferred_electrode = clusters_data['electrodes'][args.template_id]
    local_cluster = clusters_data['local_clusters'][args.template_id]
    clusters = clusters_data['clusters'][preferred_electrode]
    times = clusters_data['times'][preferred_electrode]
    assert local_cluster in np.unique(clusters), (local_cluster, np.unique(clusters))
    selection = (clusters == local_cluster)
    peak_times = times[selection]
    peak_times = np.sort(peak_times)

    # Find templates whose spike times match these peak times.
    proportions = np.zeros_like(template_ids, dtype=np.float)
    for k, template_id in enumerate(template_ids):
        # Select spike times.
        spike_times = results_data['spike_times'][template_id]
        spike_times = np.sort(spike_times)
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
    ax.set_title("peak times as spike times (+/- {} ms, template {})".format(args.interval, args.template_id))
    fig.tight_layout()
