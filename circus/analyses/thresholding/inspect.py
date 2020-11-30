"""Inspect multi-unit activity (MUA)"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.collections as mcollections
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

from circus.shared.parser import CircusParser
from circus.shared.probes import get_nodes_and_edges
from circus.analyses.utils import load_mua_data


# Parse arguments.
parser = argparse.ArgumentParser(description="Inspect multi-unit activity.")
parser.add_argument('datafile', help="data file")
parser.add_argument(
    '-c', '--channels', default=None, nargs='*', type=int, help="channel identifiers", dest='channel_ids'
)
args = parser.parse_args()

# Load parameters.
params = CircusParser(args.datafile)
_ = params.get_data_file()
sampling_rate = params.rate  # in Hz
duration = params.data_file.duration / sampling_rate  # in s

# Load MUA data.
mua_data = load_mua_data(params)

channel_ids = np.sort(mua_data['spiketimes'].keys())
# assert np.all(np.diff(channel_ids) == 1)  # i.e. no missing channel

nbs_peaks = np.zeros_like(channel_ids, dtype=np.int)
for k, channel_id in enumerate(channel_ids):
    nbs_peaks[k] = mua_data['spiketimes'][channel_id].size

peak_rates = nbs_peaks / sampling_rate  # in Hz

minimum_amplitudes = np.zeros_like(channel_ids, dtype=np.float)
for k, channel_id in enumerate(channel_ids):
    minimum_amplitudes[k] = np.min(mua_data['amplitudes'][channel_id])

probe = params.probe
nodes, edges = get_nodes_and_edges(params)
positions = []
for i in probe['channel_groups'][1]['geometry'].keys():
    positions.append(probe['channel_groups'][1]['geometry'][i])
positions = np.array(positions)
dx = np.median(np.diff(np.unique(positions[:, 0])))  # horizontal inter-electrode distance
dy = np.median(np.diff(np.unique(positions[:, 1])))  # vertical inter-electrode distance
x_min = np.min(positions[:, 0]) - dx
x_max = np.max(positions[:, 0]) + dx
y_min = np.min(positions[:, 1]) - dy
y_max = np.max(positions[:, 1]) + dy


def plot_probe_scatter(values, label, vmin=None, vmax=None, gamma=1.0):

    patches = []
    colors = []
    for channel_id, value in zip(channel_ids, values):
        xy = positions[nodes[channel_id]]
        kwargs = dict(
            radius=(0.8 * min(dx, dy) / 2.0),
        )
        patch = mpatches.Circle(xy, **kwargs)
        patches.append(patch)
        colors.append(value)
    patches = np.array(patches)
    colors = np.array(colors)

    kwargs = dict(
        edgecolors='black',
        norm=mcolors.PowerNorm(gamma=gamma),
        cmap='viridis',
        # cmap='Greys',
    )
    collection = mcollections.PatchCollection(patches, **kwargs)
    collection.set_array(colors)
    collection.set_clim(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.add_collection(collection)
    if args.channel_ids is None:
        for channel_id in channel_ids:
            x, y = positions[nodes[channel_id]]
            s = f"{channel_id:03d}"
            _ = ax.text(x, y, s, fontsize=7, verticalalignment='center_baseline', horizontalalignment='center')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(collection, ax=ax)
    cb.set_label(label)
    ax.set_title("multi-unit activity")
    fig.tight_layout()

    return


plot_probe_scatter(peak_rates, "peak rate (Hz)", vmin=0.0, gamma=0.5)


plot_probe_scatter(minimum_amplitudes, "minimum amplitude", vmax=0.0)


if args.channel_ids is not None:

    nb_selected_channels = len(args.channel_ids)

    fig, axes = plt.subplots(
        nrows=nb_selected_channels, squeeze=False, sharex='all', sharey='all',
        figsize=(4.0 * 1.6, 1.0 * float(nb_selected_channels) * 1.6)
    )

    for k, channel_id in enumerate(args.channel_ids):

        peak_times = mua_data['spiketimes'][channel_id] / sampling_rate  # in s
        amplitudes = mua_data['amplitudes'][channel_id]  # in ?

        ax = axes[k, 0]
        kwargs = dict(
            s=(2 ** 2),
            color='black',
            alpha=0.1,
            edgecolor='none',
        )
        ax.scatter(peak_times, amplitudes, **kwargs)
        ax.axvline(x=0.0, color='tab:gray', linewidth=0.5, zorder=0)
        ax.axvline(x=duration, color='tab:gray', linewidth=0.5, zorder=0)
        ax.axhline(y=0.0, color='tab:gray', linewidth=0.5, zorder=0)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel("time (s)") if ax.get_xticklabels() else None
        ax.set_ylabel("amplitude")
        ax.set_title(f"channel {channel_id:03d}")

    fig.tight_layout()
