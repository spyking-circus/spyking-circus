"""Inspect template similarities vs support sizes."""
import argparse
import matplotlib.pyplot as plt
import numpy as np

from circus.shared.parser import CircusParser
from circus.shared.files import load_data
from circus.analyses.utils import load_templates, load_templates_data


# Parse arguments.
parser = argparse.ArgumentParser(description="Inspect similarities vs support sizes")
parser.add_argument('datafile', help="data file")
parser.add_argument(
    '-t', '--threshold', default=10, type=int, help="threshold to adapt the exponent", dest='adapted_thr'
)
parser.add_argument('-c', '--cc_merge', default=0.95, type=float, help="similarity threshold (between 0.0 and 1.0)", dest='cc_merge')
args = parser.parse_args()

# Load parameters.
params = CircusParser(args.datafile)
_ = params.get_data_file()
sampling_rate = params.rate  # Hz
file_out_suff = params.get('data', 'file_out_suff')
nb_channels = params.getint('data', 'N_e')
nb_time_steps = params.getint('detection', 'N_t')

common_supports = load_data(params, 'common-supports')

# Load maximum values of the overlaps as template similarities.
templates = load_templates(params, extension='')
templates_data = load_templates_data(params, extension='', keys=['maxoverlap', 'maxlag'])
template_similarities = templates_data['maxoverlap'] / float(nb_channels * nb_time_steps)

nb_templates, nb_time_steps, nb_channels = templates.shape
template_ids = np.arange(0, nb_templates)
template_ids_1, template_ids_2 = np.meshgrid(template_ids, template_ids, indexing='ij')

template_supports = np.count_nonzero(np.any(templates != 0.0, axis=1), axis=1)

assert np.min(template_similarities) >= 0.0, np.min(template_similarities)
assert np.max(template_similarities) <= 1.0, np.max(template_similarities)

# ...
print("template similarity max.: {}".format(np.max(template_similarities)))
print("template_similarity min.: {}".format(np.min(template_similarities)))

# ...
selection = np.full_like(template_similarities, True, dtype=np.bool)
# # Remove entries below the main diagonal
for template_id in template_ids:
    selection[template_id, 0:(template_id + 1)] = False
# # Remove entries for noisy templates.
for template_id in template_ids:
    if template_supports[template_id] <= 1:  # i.e. support less than nb_channels
        selection[template_id, :] = False
        selection[:, template_id] = False

# Plot template similarities.
fig, axes = plt.subplots(nrows=2, squeeze=False, sharex='all')

data = template_similarities[selection]
support = common_supports[selection]
ids_1 = template_ids_1[selection]
ids_2 = template_ids_2[selection]

average = np.zeros(nb_channels, dtype=np.float32)
variance = np.zeros(nb_channels, dtype=np.float32)
for i in range(nb_channels):
    idx = np.where((support == i) & (data > 0.0))[0]
    average[i] = np.mean(data[idx])
    variance[i] = np.std(data[idx])

# Top subplot.
ax = axes[0, 0]
x = support
y = data
c = np.full_like(y, 'black', dtype=np.object)
c[data < 0.0] = 'tab:gray'
c[data >= args.cc_merge] = 'tab:red'
pc = ax.scatter(x, y, s=(3 ** 2), c=c, picker=True)
ax.fill_between(np.arange(nb_channels), average - variance, average + variance, color='black', alpha=0.25)
ax.fill_between([0, nb_channels], [args.cc_merge, args.cc_merge], [1, 1], color='tab:red', alpha=0.25)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.set_xlabel("nb. channels")
ax.set_ylabel("similarity")

annot = ax.annotate(
    "", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
    bbox=dict(boxstyle="round", fc="w"),
    arrowprops=dict(arrowstyle="->")
)
annot.set_visible(False)


def update_annot(k):

    pos = pc.get_offsets()[k]
    annot.xy = pos
    text = "({}, {})".format(ids_1[k], ids_2[k])
    annot.set_text(text)

    return


def pick_handler(event):

    k = np.random.choice(event.ind)
    update_annot(k)
    annot.set_visible(True)
    fig.canvas.draw_idle()

    return


fig.canvas.mpl_connect('pick_event', pick_handler)


exponents = np.exp(- support / args.adapted_thr)
new_data = data ** exponents

average = np.zeros(nb_channels, dtype=np.float32)
variance = np.zeros(nb_channels, dtype=np.float32)
for i in range(nb_channels):
    idx = np.where((support == i) & (data > 0.0))[0]
    average[i] = np.mean(new_data[idx])
    variance[i] = np.std(new_data[idx])

# Bottom subplot.
ax = axes[1, 0]
x = support
y = new_data
c = np.full_like(y, 'black', dtype=np.object)
c[new_data <= 0.0] = 'tab:gray'
c[new_data >= args.cc_merge] = 'tab:red'
ax.scatter(x, y, s=(3 ** 2), c=c)
ax.fill_between(np.arange(nb_channels), average - variance, average + variance, color='black', alpha=0.25)
ax.fill_between([0, nb_channels], [args.cc_merge, args.cc_merge], [1, 1], color='tab:red', alpha=0.25)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("nb. channels")
ax.set_ylabel("corrected similarity")


fig.tight_layout()
# plt.show()
