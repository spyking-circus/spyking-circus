"""Inspect numbers of fitted spikes."""
import argparse
import matplotlib.pyplot as plt
import numpy as np

from circus.shared.parser import CircusParser
from circus.shared.files import load_data


# Parse arguments.
parser = argparse.ArgumentParser(description="Inspect numbers of fitted spikes.")
parser.add_argument('datafile', help="data file")
parser.add_argument('-e', '--extension', default=None, help="data extension")
args = parser.parse_args()
# # Adjust extension argument.
if args.extension is None:
    args.extension = ""
else:
    args.extension = "-" + args.extension

# Load parameters.
params = CircusParser(args.datafile)
_ = params.get_data_file()

# Load spike times.
results = load_data(params, 'results', extension=args.extension)
spike_times = dict()
for key in list(results['spiketimes'].keys()):
    if key[:5] == 'temp_':
        template_key = key
        template_id = int(key[5:])
        spike_times[template_id] = np.sort(results['spiketimes'][template_key])  # sample

# Compute number of spikes.
template_ids = np.array(list(spike_times.keys()))
nb_spikes = np.array([
    spike_times[template_id].size
    for template_id in template_ids
])

# Plot number of spikes.
fig, ax = plt.subplots()
x = template_ids
y = nb_spikes
scatter_kwargs = {
    's': 3 ** 2,
    'color': 'black',
}
ax.scatter(x, y, **scatter_kwargs)
ax.axhline(y=0.0, color='black', linewidth=0.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 2))
ax.set_xlabel("template")
ax.set_ylabel("nb. spikes")
ax.set_title("numbers of spikes")
fig.tight_layout()
# plot.show()
