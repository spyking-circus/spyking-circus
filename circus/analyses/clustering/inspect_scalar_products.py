"""Inspect scalar products."""
import argparse
import matplotlib.pyplot as plt
import numpy as np

from circus.shared.parser import CircusParser
from circus.analyses.utils import load_clusters_data, load_template, plot_snippets, plot_template
from circus.shared.probes import get_nodes_and_edges
from circus.shared.files import get_stas


# Parse arguments.
parser = argparse.ArgumentParser(description="Inspect scalar products.")
parser.add_argument('datafile', help="data file")
parser.add_argument('-t', '--template', default=0, type=int, help="template index", dest='template_id')
args = parser.parse_args()

# Load parameters.
params = CircusParser(args.datafile)
_ = params.get_data_file()
nb_electrodes = params.nb_channels
sampling_rate = params.rate  # Hz
duration = params.data_file.duration / sampling_rate  # s
nb_channels = params.getint('data', 'N_e')
nb_time_steps = params.getint('detection', 'N_t')
nb_snippets = params.getint('clustering', 'nb_snippets')

# Load snippets.
clusters_data = load_clusters_data(params, extension='')
electrode = clusters_data['electrodes'][args.template_id]
local_cluster = clusters_data['local_clusters'][args.template_id]
assert electrode.shape == local_cluster.shape, (electrode.shape, local_cluster.shape)
times = clusters_data['times'][electrode]
clusters = clusters_data['clusters'][electrode]
assert times.shape == clusters.shape, (times.shape, clusters.shape)
selection = (clusters == local_cluster)
times = times[selection]
clusters = clusters[selection]
if times.size > nb_snippets:
    indices = np.random.choice(times.size, size=nb_snippets)
    indices = np.sort(indices)
    times = times[indices]
    clusters = clusters[indices]
nodes, _ = get_nodes_and_edges(params)
inv_nodes = np.zeros(nb_electrodes, dtype=int)
inv_nodes[nodes] = np.arange(len(nodes))
indices = inv_nodes[nodes]
snippets = get_stas(params, times, clusters, electrode, indices, nodes=nodes)
snippets = np.transpose(snippets, axes=(0, 2, 1))
assert snippets.shape == (nb_snippets, nb_time_steps, nb_channels), \
    (snippets.shape, nb_snippets, nb_time_steps, nb_channels)

# Load template.
template = load_template(args.template_id, params, extension='')
assert template.shape == (nb_time_steps, nb_channels)

# Compute the scalar products.
snippets_ = np.reshape(snippets, (nb_snippets, nb_channels * nb_time_steps))
template_ = np.reshape(template, (nb_channels * nb_time_steps, 1))
# print(snippets_.shape)
# print(template_.shape)
scalar_products_ = snippets_.dot(template_)
scalar_products = np.reshape(scalar_products_, (nb_snippets,))
# print(scalar_products.shape)
# print(scalar_products)
# print(np.mean(scalar_products))
normalized_scalar_products = scalar_products / np.linalg.norm(template) ** 2
# print(normalized_scalar_products.shape)
# print(normalized_scalar_products)
# print(np.mean(normalized_scalar_products))

# Plot normalized scalar products.
fig, ax = plt.subplots()
x = times / sampling_rate  # s
y = normalized_scalar_products
ax.scatter(x, y, s=(3 ** 2), color='black')
axline_kwargs = {
    'color': 'black',
    'linewidth': 0.5,
}
ax.axvline(x=0.0, **axline_kwargs)
ax.axvline(x=duration, **axline_kwargs)
ax.axhline(y=0.0, **axline_kwargs)
ax.axhline(y=1.0, **axline_kwargs)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("time (s)")
ax.set_ylabel("scalar product")
ax.set_title("clustering scalar products (template {})".format(args.template_id))
fig.tight_layout()

# Plot snippets and template.
snippets = snippets[:10, :, :]
fig, ax = plt.subplots()
kwargs = {
    'vmin': np.min([np.min(a) for a in [snippets, template]]),
    'vmax': np.min([np.max(a) for a in [snippets, template]]),
}
plot_snippets(ax, snippets, params, color='black', **kwargs)
plot_template(ax, template, params, color='tab:blue', label="template", **kwargs)
ax.set_xticks([])
ax.set_yticks([])
ax.legend()
ax.set_title("clustering snippets (template {})".format(args.template_id))
fig.tight_layout()
