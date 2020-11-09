"""Inspect clustering snippets for a given template."""
import argparse
import matplotlib.pyplot as plt
import numpy as np

from circus.shared.parser import CircusParser
from circus.analyses.utils import \
    load_clusters_data, load_snippets, plot_snippets, plot_snippet, load_template, plot_template


# Parse arguments.
parser = argparse.ArgumentParser(description="Inspect clustering snippets for a given template.")
parser.add_argument('datafile', help="data file")
parser.add_argument('-t', '--template', default=0, type=int, help="template index", dest='template_id')
parser.add_argument(
    '-n', '--nb-snippets', default=10, type=int, help="maximum number of snippets", dest='nb_snippets_max'
)
args = parser.parse_args()

# Load parameters.
params = CircusParser(args.datafile)
_ = params.get_data_file()
sampling_rate = params.rate
file_out_suff = params.get('data', 'file_out_suff')
nb_channels = params.getint('data', 'N_e')
nb_time_steps = params.getint('detection', 'N_t')

# Load clusters data.
clusters_data = load_clusters_data(params, extension='')

# ...
channel_id = clusters_data['electrodes'][args.template_id]
local_cluster_id = clusters_data['local_clusters'][args.template_id]

# ...
data = clusters_data['data'][channel_id]
local_clusters = clusters_data['clusters'][channel_id]
times = clusters_data['times'][channel_id]

# Collect snippets.
selection = np.array(local_clusters == local_cluster_id)
selected_times = times[selection]
if selected_times.size > args.nb_snippets_max:
    selected_times = np.random.choice(selected_times, size=args.nb_snippets_max, replace=False)
    # order = np.argsort(np.abs(data[selection, :][:, 0]))
    # # selected_times = selected_times[order[:args.nb_snippets_max]]
    # selected_times = selected_times[order[-args.nb_snippets_max:]]
snippets = load_snippets(selected_times, params)

# Compute mean snippets.
median_snippet = np.mean(snippets, axis=0)

# Load template.
template = load_template(args.template_id, params, extension='')

# Shift template.
shifted_template = np.zeros_like(template)
lag = np.argmin(template[:, channel_id]) - (nb_time_steps - 1) // 2
shifted_template[max(0 - lag, 0):min(nb_time_steps - lag, nb_time_steps)] = \
    template[max(0 + lag, 0):min(nb_time_steps + lag, nb_time_steps)]
print("lag: {}".format(lag))

# Plot snippets.
fig, ax = plt.subplots()
vmin = np.min(snippets)
vmax = np.max(snippets)
kwargs = dict(
    color='tab:gray',
    vmin=vmin,
    vmax=vmax,
    label="data"
)
plot_snippets(ax, snippets, params, **kwargs)
kwargs = dict(
    color='black',
    vmin=vmin,
    vmax=vmax,
    label="mean"
)
plot_snippet(ax, median_snippet, params, **kwargs)
kwargs = dict(
    color='tab:blue',
    vmin=vmin,
    vmax=vmax,
    label="template"
)
plot_template(ax, template, params, **kwargs)
if lag != 0:
    kwargs = dict(
        color='tab:orange',
        vmin=vmin,
        vmax=vmax,
        label="shifted template"
    )
    plot_template(ax, shifted_template, params, **kwargs)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("clustering snippets (template {})".format(args.template_id))
ax.legend()
fig.tight_layout()
# plt.show()
