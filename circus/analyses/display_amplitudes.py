"""Display the amplitudes over time for a given template."""
import argparse
import matplotlib.pyplot as plt

from circus.shared.parser import CircusParser
from circus.shared.files import load_data


# Parse arguments.
parser = argparse.ArgumentParser(description="Display the amplitudes over time for a given template.")
parser.add_argument('datafile', help="data file")
parser.add_argument('-e', '--extension', default=None, help="data extension")
parser.add_argument('-t', '--template', default=0, type=int, help="template index", dest='template_id')
args = parser.parse_args()
# # Adjust extension argument.
if args.extension is None:
    args.extension = ""
else:
    args.extension = "-" + args.extension

# Load parameters.
params = CircusParser(args.datafile)
_ = params.get_data_file()
sampling_rate = params.rate
duration = params.data_file.duration

# Load amplitudes over time.
results = load_data(params, 'results', extension=args.extension)
template_key = 'temp_{}'.format(args.template_id)
spike_times = results['spiketimes'][template_key]
amplitudes = results['amplitudes'][template_key][:, 0]

# Display amplitudes over time.
fig, ax = plt.subplots()
scatter_kwargs = {
    's': 3 ** 2,
    'c': 'black',
    # 'alpha': 0.5,
    # 'edgecolors': 'none',
}
ax.scatter(spike_times / sampling_rate, amplitudes, **scatter_kwargs)
axline_kwargs = {
    'color': 'black',
    'linewidth': 0.5,
}
ax.axvline(x=0.0, **axline_kwargs)
ax.axvline(x=(duration / sampling_rate), **axline_kwargs)
ax.axhline(y=0.0, **axline_kwargs)
ax.axhline(y=1.0, **axline_kwargs)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("time (s)")
ax.set_ylabel("amplitude")
ax.set_title("amplitudes (template {})".format(args.template_id))
fig.tight_layout()
# plt.show()
