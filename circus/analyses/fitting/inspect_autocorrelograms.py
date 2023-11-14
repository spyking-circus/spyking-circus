"""Inspect autocorrelograms."""
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import tqdm

from circus.shared.parser import CircusParser
from circus.shared.files import load_data


# Parse arguments.
parser = argparse.ArgumentParser(  # noqa
    description="Inspect autocorrelograms.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('datafile', help="data file")
parser.add_argument('-e', '--extension', default=None, help="data extension")
parser.add_argument('-ws', '--window-size', default=51.0, type=float, help="window size (ms)")
parser.add_argument('-bs', '--bin-size', default=1.0, type=float, help="bin size (ms)")
parser.add_argument(
    '-g', '--gamma', default=0.5, type=float,
    help="parameter used to remap the colors onto a power-law relationship (linear: 1.0)"
)
args = parser.parse_args()
# # Adjust extension argument.
if args.extension is None:
    args.extension = ""
else:
    args.extension = "-" + args.extension

# Load parameters.
params = CircusParser(args.datafile)
_ = params.get_data_file()
sampling_rate = params.rate  # Hz
duration = params.data_file.duration / sampling_rate  # s

window_size = args.window_size  # ms
bin_size = args.bin_size  # ms

nb_bins = int(np.ceil(window_size / bin_size))
nb_bins = nb_bins + 1 if nb_bins % 2 == 0 else nb_bins
lag_min = - (float((nb_bins - 1) // 2) + 0.5) * bin_size  # ms
lag_max = + (float((nb_bins - 1) // 2) + 0.5) * bin_size  # ms
window_size = lag_max - lag_min  # ms

# Load spike times.
results = load_data(params, 'results', extension=args.extension)
spike_times = dict()
for key in list(results['spiketimes'].keys()):
    if key[:5] == 'temp_':
        template_key = key
        template_id = int(key[5:])
        spike_times[template_id] = np.sort(results['spiketimes'][template_key]) / sampling_rate  # s


def compute_autocorrelogram(spike_times, bin_size, nb_bins, normalize=False):

    assert nb_bins % 2 == 1, nb_bins

    autocorrelogram = np.zeros(nb_bins, dtype=int)
    b = np.unique(np.round(spike_times / bin_size))
    for k in range(0, nb_bins):
        autocorrelogram[k] += len(np.intersect1d(b, b + k - (nb_bins - 1) // 2, assume_unique=True))
    autocorrelogram[(nb_bins - 1) // 2] = 0

    if normalize:
        # v1
        autocorrelogram = autocorrelogram / duration
        autocorrelogram = autocorrelogram / bin_size
        # # v2
        # autocorrelogram = autocorrelogram / max(len(spike_times), 1)  # i.e. firing rate x duration
        # autocorrelogram = autocorrelogram / bin_size
        # # v3
        # autocorrelogram = autocorrelogram / max(int(np.sum(autocorrelogram)), 1)  # i.e. density
        # autocorrelogram = autocorrelogram / bin_size

    bin_edges = np.arange(0, nb_bins + 1) - float(nb_bins) / 2.0

    return autocorrelogram, bin_edges


# Compute autocorrelograms.
template_ids = np.sort(list(spike_times.keys()))
nb_templates = template_ids.size
autocorrelograms = np.zeros((nb_templates, nb_bins))
bin_edges = None
for k in tqdm.tqdm(list(range(0, nb_templates))):
    template_id = template_ids[k]
    autocorrelograms[k, :], bin_edges = \
        compute_autocorrelogram(spike_times[template_id], bin_size * 1e-3, nb_bins, normalize=True)

# Plot autocorrelograms.
fig, ax = plt.subplots()
imshow_kwargs = {
    'cmap': 'Greys',
    'norm': mcolors.PowerNorm(gamma=args.gamma),
    'aspect': 'auto',
    'vmin': 0.0,
    'extent': (
        bin_edges[0],
        bin_edges[-1],
        float(nb_templates - 1) + 0.5,
        float(0) - 0.5,
    ),
}
ai = ax.imshow(autocorrelograms, **imshow_kwargs)
ax.axvline(x=0.0, color='black', linewidth=0.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("lag (ms)")
ax.set_ylabel("template")
ax.set_title("autocorrelograms")
cb = fig.colorbar(ai, ax=ax)
fig.tight_layout()
# plot.show()