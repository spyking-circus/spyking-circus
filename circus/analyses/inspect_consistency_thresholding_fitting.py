"""Inspect consistency between thresholding and fitting."""

import argparse
import matplotlib.pyplot as plt
import numpy as np

from circus.shared.parser import CircusParser
from circus.analyses.utils import load_mua_data
from circus.analyses.utils import load_result_data
from circus.analyses.utils import compute_smallest_intervals
from circus.analyses.utils import load_snippets, plot_snippets
from circus.analyses.utils import load_template, plot_template


# Parse arguments.
parser = argparse.ArgumentParser(description="Inspect consistency between thresholding and fitting.")
parser.add_argument('datafile', help="data file")
parser.add_argument('channel_id', type=int, help="channel identifier")
parser.add_argument(
    '-a', '--amplitude', default=None, type=float, help="threshold amplitude"
)
parser.add_argument(
    '-i', '--interval', default=1.0, type=float, help="threshold interval (in ms)"
)
parser.add_argument(
    '-t', '--templates', default=None, nargs='*', type=int, help="template identifiers", dest='template_ids'
)
args = parser.parse_args()
# Adjust parameters.
args.extension = ''


# Load parameters.
params = CircusParser(args.datafile)
_ = params.get_data_file()
sampling_rate = params.rate  # in Hz

# Load multi-unit activity data.
mua_data = load_mua_data(params)

# Load result data.
result_data = load_result_data(params, extension=args.extension)

# Get peak time (for specified channel).
peak_times = mua_data['spiketimes'][args.channel_id]
peak_amplitudes = mua_data['amplitudes'][args.channel_id]
# # Sort.
indices = np.argsort(peak_times)
peak_times = peak_times[indices]
peak_amplitudes = peak_amplitudes[indices]
# # Select.
if args.amplitude is not None:
    print(np.min(peak_amplitudes))
    print(np.max(peak_amplitudes))
    selection = peak_amplitudes < args.amplitude
    peak_times = peak_times[selection]
    peak_amplitudes = peak_amplitudes[selection]

# Get spike times (for each templates).
spike_times = result_data['spiketimes']

# Get template identifiers.
template_ids = np.sort(spike_times.keys())


# TODO mark spike times near peak times.
intervals = dict()
for template_id in template_ids:
    intervals[template_id] = compute_smallest_intervals(peak_times, spike_times[template_id])
are_matched = dict()
for template_id in template_ids:
    are_matched[template_id] = np.abs(intervals[template_id] / sampling_rate * 1e+3) < args.interval
matching_proportions = np.array([
    np.count_nonzero(are_matched[template_id]) / peak_times.size
    for template_id in template_ids
])
nbs_spikes = np.array([
    spike_times[template_id].size
    for template_id in template_ids
])


# TODO plot...
fig, ax = plt.subplots()
x = peak_times / sampling_rate
y = peak_amplitudes
kwargs = dict(
    s=(3 ** 2),
    color='black',
)
pc = ax.scatter(x, y, **kwargs)
ax.axhline(y=0.0, color='tab:gray', linewidth=0.5, zorder=0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("time (s)")
ax.set_ylabel("amplitude")
ax.set_title(f"peaks c{args.channel_id}")
fig.tight_layout()


# TODO plot ...
fig, ax = plt.subplots()
x = template_ids
y = matching_proportions
kwargs = dict(
    s=(3 ** 2),
    c=nbs_spikes,
)
ax.scatter(x, y, **kwargs)
ax.axhline(y=0.0, color='tab:gray', linewidth=0.5, zorder=0)
ax.axhline(y=1.0, color='tab:gray', linewidth=0.5, zorder=0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("template")
ax.set_ylabel("proportion")
ax.set_title(f"proportions of peaks on c{args.channel_id} fitted by each template")
fig.tight_layout()


if args.template_ids is not None:

    intervals = dict()
    are_matched = dict()
    for template_id in args.template_ids:
        intervals[template_id] = compute_smallest_intervals(peak_times, spike_times[template_id])
        are_matched[template_id] = np.abs(intervals[template_id] / sampling_rate * 1e+3) < args.interval
    # are_not_matched = np.full_like(peak_times, True, dtype=np.bool)
    # for template_id in args.template_ids:
    #     are_not_matched[are_matched[template_id]] = False
    nb_matches = np.zeros_like(peak_times, dtype=np.int)
    for template_id in args.template_ids:
        nb_matches[are_matched[template_id]] += 1
    are_not_matched = (nb_matches == 0)
    are_multi_matched = (nb_matches > 1)

    fig, ax = plt.subplots()
    for k, template_id in enumerate(args.template_ids):
        selection = are_matched[template_id]
        x = peak_times[selection] / sampling_rate
        y = peak_amplitudes[selection]
        kwargs = dict(
            s=(3 ** 2),
            color=f'C{k % 10}',
            label=f"t{template_id} ({100.0 * np.count_nonzero(selection) / selection.size:.1f} \%)",
        )
        ax.scatter(x, y, **kwargs)
    # # Plot peaks matched multiple times.
    selection = are_multi_matched
    x = peak_times[selection] / sampling_rate
    y = peak_amplitudes[selection]
    kwargs = dict(
        s=(3 ** 2),
        color='tab:red',
        label=f"multi ({100.0 * np.count_nonzero(selection) / selection.size:.1f} \%)"
    )
    ax.scatter(x, y, **kwargs)
    # # Plot peaks not matched.
    selection = are_not_matched
    x = peak_times[selection] / sampling_rate
    y = peak_amplitudes[selection]
    kwargs = dict(
        s=(3 ** 2),
        color='black',
        label=f"none ({100.0 * np.count_nonzero(selection) / selection.size:.1f} \%)"
    )
    ax.scatter(x, y, **kwargs)
    ax.axhline(y=0.0, color='tab:gray', linewidth=0.5, zorder=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.legend()
    ax.set_title(f"peaks c{args.channel_id}")
    fig.tight_layout()

    # Inspect peak snippets.

    snippets = dict()
    for template_id in args.template_ids:
        selected_peak_times = np.random.choice(
            peak_times[are_matched[template_id]], size=min(20, np.count_nonzero(are_matched[template_id]))
        )
        snippets[template_id] = load_snippets(selected_peak_times, params)
    selected_peak_times = np.random.choice(peak_times[are_not_matched], size=min(20, np.count_nonzero(are_not_matched)))
    snippets[None] = load_snippets(selected_peak_times, params)

    templates = dict()
    for template_id in args.template_ids:
        templates[template_id] = load_template(template_id, params, extension=args.extension)

    fig, axes = plt.subplots(
        ncols=len(snippets), squeeze=False,
        sharex='all', sharey='all',
        figsize=(3.0 * float(len(snippets)) * 1.6, 3.0 * 1.6)
    )
    kwargs = dict(
        vmin=np.min([np.min(s) for s in snippets.values()]),
        vmax=np.max([np.max(s) for s in snippets.values()]),
    )
    for k, template_id in enumerate(snippets.keys()):
        ax = axes[0, k]
        plot_snippets(ax, snippets[template_id], params, color='black', alpha=0.2, **kwargs)
        if template_id is not None:
            plot_template(ax, templates[template_id], params, color=f'C{k % 10}', **kwargs)
            ax.set_title(f"t{template_id}")
        else:
            for k_, template_id_ in enumerate(args.template_ids):
                plot_template(ax, templates[template_id_], params, color=f'C{k_ % 10}', **kwargs)
            ax.set_title("none")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
