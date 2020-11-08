"""Inspect template similarities."""
import argparse
import matplotlib as mpl
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np

from circus.shared.parser import CircusParser
from circus.shared.files import load_data
from circus.analyses.utils import load_templates_data


# Parse arguments.
parser = argparse.ArgumentParser(description="Inspect similarities vs support sizes")
parser.add_argument('datafile', help="data file")
parser.add_argument(
    '-t', '--threshold', default=50, type=int, help="threshold to adapt the exponent", dest='adapted_thr'
)
parser.add_argument('-c', '--cc_merge', default=0.95, type=float, help="similarity threshold (between 0.0 and 1.0)", dest='cc_merge')
args = parser.parse_args()
# # Adjust arguments.

# Load parameters.
params = CircusParser(args.datafile)

_ = params.get_data_file()
sampling_rate = params.rate  # Hz
file_out_suff = params.get('data', 'file_out_suff')
nb_channels = params.getint('data', 'N_e')
nb_time_steps = params.getint('detection', 'N_t')
common_supports = load_data(params, 'common-supports')

# Load maximum values of the overlaps as template similarities.
templates_data = load_templates_data(params, extension='', keys=['maxoverlap', 'maxlag'])
template_similarities = templates_data['maxoverlap'] / float(nb_channels * nb_time_steps)

assert np.min(template_similarities) >= 0.0, np.min(template_similarities)
assert np.max(template_similarities) <= 1.0, np.max(template_similarities)

# ...
print("template similarity max.: {}".format(np.max(template_similarities)))
print("template_similarity min.: {}".format(np.min(template_similarities)))

# Plot template similarities.
fig, ax = plt.subplots(2)

data = template_similarities.flatten()
support = common_supports.flatten()

xaxis = np.arange(nb_channels)
average = np.zeros(nb_channels, dtype=np.float32)
variance = np.zeros(nb_channels, dtype=np.float32)
for i in range(nb_channels):
    idx = np.where(support == i)[0]
    average[i] = np.mean(data[idx])
    variance[i] = np.std(data[idx])

idx = np.where(data < args.cc_merge)[0]
ax[0].plot(support[idx].flatten(), data[idx].flatten(), '.', c='k')
idx = np.where(data >= args.cc_merge)[0]
nb_merges = len(idx)
ax[0].plot(support[idx].flatten(), data[idx].flatten(), '.', c='r')

ax[0].fill_between(np.arange(nb_channels), average -variance, average+variance, color='r', alpha=0.25)
ax[0].fill_between([0, nb_channels], [args.cc_merge, args.cc_merge], [1, 1], color='k', alpha=0.25)
ax[0].set_ylabel("similarity")
ax[0].plot(xaxis, average, 'r', lw=1)
ax[0].set_xlim(support.min(), support.max())
ax[0].set_title('Fixed threshold [%d merges]' %nb_merges)

cc_merge_bis = args.cc_merge**(1/np.exp(-support/args.adapted_thr))
idx = np.where(data < cc_merge_bis)[0]
ax[1].plot(support[idx].flatten(), data[idx].flatten(), '.', c='k')
idx = np.where(data >= cc_merge_bis)[0]
nb_merges = len(idx)
ax[1].plot(support[idx].flatten(), data[idx].flatten(), '.', c='r')
ax[1].set_xlabel("# common channels")
ax[1].set_ylabel("similarity")

cc_merge_bis = args.cc_merge**(1/np.exp(-xaxis/args.adapted_thr))
ax[1].fill_between(xaxis, cc_merge_bis, np.ones(nb_channels), color='k', alpha=0.25)
ax[1].fill_between(xaxis, average -variance, average+variance, color='r', alpha=0.25)
ax[1].plot([0, nb_channels], [args.cc_merge, args.cc_merge], 'k--')
ax[1].plot(xaxis, average, 'r', lw=1)
ax[1].set_xlim(support.min(), support.max())
ax[1].set_title('Adaptive threshold [%d merges]' %nb_merges)

fig.tight_layout()