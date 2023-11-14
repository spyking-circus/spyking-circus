"""Inspect clusters snippets for given channel."""
import argparse
import matplotlib.pyplot as plt
import numpy as np

from circus.shared.parser import CircusParser
from circus.analyses.utils import plot_probe
from circus.analyses.utils import load_snippets, plot_snippets
from circus.analyses.utils import load_clusters_data


# Parse arguments.
parser = argparse.ArgumentParser(description="Inspect clusters snippets for given channel.")
parser.add_argument('datafile', help="data file")
group = parser.add_mutually_exclusive_group()
group.add_argument('-c', '--channel', default=0, type=int, help="channel index", dest='channel_id')
group.add_argument('-t', '--template', default=None, type=int, help="template identifier", dest='template_id')
args = parser.parse_args()

# Load parameters.
params = CircusParser(args.datafile)
_ = params.get_data_file()
sampling_rate = params.rate
file_out_suff = params.get('data', 'file_out_suff')
nb_channels = params.getint('data', 'N_e')
nb_time_steps = params.getint('detection', 'N_t')

# Load clusters data.
clusters_data = load_clusters_data(params, extension='')  # TODO support other extensions?

if args.template_id is not None:
    args.channel_id = clusters_data['electrodes'][args.template_id]

data = clusters_data['data']
electrode_nbs = np.sort(list(data.keys()))
nb_electrodes, = electrode_nbs.shape
clusters = clusters_data['clusters']
times = clusters_data['times']

nb_points = np.zeros(nb_electrodes, dtype=np.int)
nb_components = None
for k in range(0, nb_electrodes):
    nb_points[k], nb_components = data[electrode_nbs[k]].shape

data_ = data[args.channel_id]
clusters_ = clusters[args.channel_id]
times_ = times[args.channel_id]
unique_clusters_ = np.unique(clusters_)
nb_clusters_ = unique_clusters_.size
template_ids = np.where(clusters_data['electrodes'] == args.channel_id)[0]

# Collect snippets.
snippets = dict()
for cluster_nb in range(0, nb_clusters_):
    selection = (clusters_ == unique_clusters_[cluster_nb])
    selected_times__ = times_[selection]
    if selected_times__.size > 10:
        # selected_times__ = np.random.choice(selected_times__, size=25, replace=False)
        order = np.argsort(np.abs(data_[selection, :][:, 0]))
        # selected_times__ = selected_times__[order[:10]]
        selected_times__ = selected_times__[order[-10:]]
    snippets[cluster_nb] = load_snippets(selected_times__, params)

# Plot snippets.
vmin = np.min([np.min(snippets[cluster_nb]) for cluster_nb in range(0, nb_clusters_)])
vmax = np.max([np.max(snippets[cluster_nb]) for cluster_nb in range(0, nb_clusters_)])
for cluster_nb in range(0, nb_clusters_):
    fig, ax = plt.subplots()
    plot_probe(ax, params, channel_ids=[args.channel_id])
    kwargs = {
        'color': 'C{}'.format(unique_clusters_[cluster_nb] % 10),
        'vmin': vmin,
        'vmax': vmax,
    }
    plot_snippets(ax, snippets[cluster_nb], params, **kwargs)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.set_xlabel("time")
    # ax.set_ylabel("voltage")
    ax.set_title("t{} (e{}, c{})".format(template_ids[cluster_nb], args.channel_id, cluster_nb))
    fig.tight_layout()
# plt.show()
