"""Inspect clusters for a given channel."""
import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import re

from circus.shared.parser import CircusParser


# Parse arguments.
parser = argparse.ArgumentParser(description="Inspect clusters for a given channel.")
parser.add_argument('datafile', help="data file")
parser.add_argument('-c', '--channel', default=0, type=int, help="channel index", dest='channel_id')
args = parser.parse_args()

# Load parameters.
params = CircusParser(args.datafile)
_ = params.get_data_file()
sampling_rate = params.rate
file_out_suff = params.get('data', 'file_out_suff')
nb_channels = params.getint('data', 'N_e')
nb_time_steps = params.getint('detection', 'N_t')

# Load clusters.
clusters_path = "{}.clusters.hdf5".format(file_out_suff)
if not os.path.isfile(clusters_path):
    raise FileNotFoundError(clusters_path)
with h5py.File(clusters_path, mode='r', libver='earliest') as clusters_file:
    clusters_data = dict()
    p = re.compile('_\d*$')  # noqa
    for key in list(clusters_file.keys()):
        m = p.search(key)
        if m is None:
            clusters_data[key] = clusters_file[key][:]
        else:
            k_start, k_stop = m.span()
            key_ = key[0:k_start]
            channel_nb = int(key[k_start+1:k_stop])
            if key_ not in clusters_data:
                clusters_data[key_] = dict()
            clusters_data[key_][channel_nb] = clusters_file[key][:]

data = clusters_data['data']
electrode_nbs = np.sort(list(data.keys()))
nb_electrodes, = electrode_nbs.shape
clusters = clusters_data['clusters']
nb_points = np.zeros(nb_electrodes, dtype=np.int)
nb_components = None
for k in range(0, nb_electrodes):
    nb_points[k], nb_components = data[electrode_nbs[k]].shape

# Plot clusters.
subplots_kwargs = {
    'nrows': 2,
    'ncols': 5,
    'squeeze': False,
    'sharex': True,
    'sharey': True,
    'figsize': (2.0 * 6.4, 4.8)
}
fig, axes = plt.subplots(**subplots_kwargs)
axes = axes.flatten()
for k in range(0, nb_components):
    ax = axes[k]
    ax.set_aspect('equal')
    component_nb_1 = k
    component_nb_2 = (k + 1) % nb_components
    x = data[args.channel_id][:, component_nb_1]
    y = data[args.channel_id][:, component_nb_2]
    c = [
        'C{}'.format(cluster_nb % 10)
        for cluster_nb in clusters[args.channel_id]
    ]
    ax.scatter(x, y, s=(3 ** 2), c=c, edgecolors='none')
    ax.axvline(color='black', linewidth=0.5)
    ax.axhline(color='black', linewidth=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("component {}".format(component_nb_1))
    ax.set_ylabel("component {}".format(component_nb_2))
fig.tight_layout()
# plt.show()
