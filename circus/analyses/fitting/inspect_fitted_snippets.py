"""Inspect fitted snippets for a given template."""
import argparse
import matplotlib.pyplot as plt
import numpy as np

from circus.shared.parser import CircusParser
from circus.shared.files import load_data

from circus.analyses.utils import load_snippets, plot_snippets, plot_snippet, load_template, plot_template


# Parse arguments.
parser = argparse.ArgumentParser(  # noqa
    description="Inspect fitted snippets for a given template.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('datafile', help="data file")
parser.add_argument('-e', '--extension', default=None, help="data extension")
parser.add_argument('-t', '--template', default=0, type=int, help="template identifier", dest='template_id')
parser.add_argument(
    '-s', '--selection', default='first',
    choices=[
        'first',
        'last',
        'random',
        'lowest_amplitudes',
        'highest_amplitudes',
        'amplitudes_close_to_1',
        'refractory_period_violations', 'rpvs',
    ],
    help="fitted snippets selection"
)
parser.add_argument('-n', '--nb-snippets', default=10, type=int, help="number of snippets to select")
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

# Load spike times and amplitudes.
results = load_data(params, 'results', extension=args.extension)
template_key = 'temp_{}'.format(args.template_id)
spike_times = results['spiketimes'][template_key]
amplitudes = results['amplitudes'][template_key][:, 0]

# Select snippets.
if args.selection == 'first':
    ids = np.argsort(spike_times)
    selected_ids = ids[:+args.nb_snippets]
elif args.selection == 'last':
    ids = np.argsort(spike_times)
    selected_ids = ids[-args.nb_snippets:]
elif args.selection == 'random':
    ids = np.argsort(spike_times)
    selected_ids = np.random.choice(ids, size=args.nb_snippets, replace=False)
elif args.selection == 'lowest_amplitudes':
    ids = np.argsort(amplitudes)
    selected_ids = ids[:+args.nb_snippets]
elif args.selection == 'highest_amplitudes':
    ids = np.argsort(amplitudes)
    selected_ids = ids[-args.nb_snippets:]
elif args.selection == 'amplitudes_close_to_1':
    ids = np.argsort(np.abs(amplitudes - 1.0))
    selected_ids = ids[:args.nb_snippets]
elif args.selection in ['refractory_period_violations', 'rpvs']:
    assert np.all(np.diff(spike_times) > 0.0)
    are_violations = np.zeros(spike_times.size, dtype=np.bool)
    spike_intervals = np.diff(spike_times) / sampling_rate
    refractory_period = 3.0e-3  # s
    are_violations[:-1] = (spike_intervals < refractory_period)
    are_violations[+1:] = (spike_intervals < refractory_period)
    ids = np.where(are_violations)[0]
    size = min(ids.size, args.nb_snippets)
    selected_ids = np.random.choice(ids, size=size, replace=False)
else:
    assert False

# Collect snippets.
spike_times_ = spike_times[selected_ids]
snippets = load_snippets(spike_times_, params)

# Compute median snippet.
median_snippet = np.median(snippets, axis=0)

# Load template.
template = load_template(args.template_id, params, extension=args.extension)

# Plot.
kwargs = {
    'vmin': np.min([np.min(a) for a in [snippets, median_snippet, template]]),
    'vmax': np.max([np.max(a) for a in [snippets, median_snippet, template]]),
}
fig, ax = plt.subplots()
ax.set_axis_off()
plot_snippets(ax, snippets, params, color='tab:grey', **kwargs)
plot_snippet(ax, median_snippet, params, color='black', label="median", **kwargs)
plot_template(ax, template, params, color='tab:blue', label="template", **kwargs)
ax.legend()
fig.tight_layout()
# plt.show()
