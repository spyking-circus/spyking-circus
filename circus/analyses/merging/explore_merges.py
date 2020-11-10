"""Explore merges."""
import argparse
# import matplotlib as mpl
# import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from circus.shared.parser import CircusParser
from circus.shared.files import load_data
from circus.analyses.utils import load_templates, load_templates_data, load_spike_times


# Parse arguments.
parser = argparse.ArgumentParser(description="Explore merges")
parser.add_argument('datafile', help="data file")
# parser.add_argument(
#     '-t', '--threshold', default=10, type=int, help="threshold to adapt the exponent", dest='adapted_thr'
# )  # TODO remove?
parser.add_argument(
    '-c', '--cc_merge', default=0.95, type=float, help="similarity threshold (between 0.0 and 1.0)", dest='cc_merge'
)
parser.add_argument(
    '-t', '--threshold', default=75, type=int, help="threshold to adapt the exponent", dest='adapted_thr'
)
args = parser.parse_args()

# Load parameters.
params = CircusParser(args.datafile)
_ = params.get_data_file()
sampling_rate = params.rate  # Hz
file_out_suff = params.get('data', 'file_out_suff')
nb_channels = params.getint('data', 'N_e')
nb_time_steps = params.getint('detection', 'N_t')

# Load templates.
templates = load_templates(params, extension='')

# ...
# nb_templates, nb_time_steps, nb_channels = templates.shape
nb_templates, _, _ = templates.shape
template_ids = np.arange(0, nb_templates)
# ...
template_supports = np.any(templates != 0.0, axis=1)

# Load template norms.
template_norms = load_data(params, 'norm-templates')
template_norms = template_norms[:nb_templates]  # i.e. remove norms of second components

# Sanity prints.
print("np.min(template_norms): {}".format(np.min(template_norms)))
print("np.max(template_norms): {}".format(np.max(template_norms)))

# # Recompute the template norms.
# template_norms_bis = np.array([
#     np.linalg.norm(templates[template_id, :, :]) / np.sqrt(nb_channels * nb_time_steps)
#     for template_id in template_ids
# ])
#
# # Check correctness of recomputed template norms.
# assert template_norms_bis.shape == template_norms.shape, (template_norms_bis.shape, template_norms.shape)
# assert np.allclose(template_norms_bis, template_norms), np.max(np.ab(template_norms_bis - template_norms))

# Load spike times.
spike_times = load_spike_times(params, extension='')

# Load template similarities.
templates_data = load_templates_data(params, extension='', keys=['maxoverlap', 'maxlag'])
template_similarities = templates_data['maxoverlap'] / float(nb_channels * nb_time_steps)
template_lags = templates_data['maxlag']

# Sanity prints.
print("np.min(template_similarities): {}".format(np.min(template_similarities)))
print("np.max(template_similarities): {}".format(np.max(template_similarities)))

# # Recompute the template similarities.
# template_similarities_bis = np.zeros_like(template_similarities)
# for id_1 in tqdm.tqdm(template_ids):
#     template_1 = templates[id_1, :, :] / template_norms[id_1]
#     support_1 = template_supports[id_1, :]
#     for id_2 in template_ids[(id_1 + 1):]:
#         template_2 = templates[id_2, :, :] / template_norms[id_2]
#         support_2 = template_supports[id_2, :]
#         intersection_support = support_1 & support_2
#         lag = - template_lags[id_1, id_2] + 1
#         if np.any(intersection_support):
#             similarity = np.dot(
#                 template_1[max(0, 0 + lag):min(nb_time_steps, nb_time_steps + lag), intersection_support].flat,
#                 template_2[max(0, 0 - lag):min(nb_time_steps, nb_time_steps - lag), intersection_support].flat
#             ) / float(nb_channels * nb_time_steps)
#         else:
#             similarity = 0.0
#         template_similarities_bis[id_1, id_2] = similarity
#         template_similarities_bis[id_2, id_1] = similarity
#
# # Check correctness of recomputed template similarities.
# assert template_similarities_bis.shape == template_similarities.shape, \
#     (template_similarities.shape, template_similarities_bis.shape)
# assert np.allclose(template_similarities_bis, template_similarities), \
#     np.max(np.abs(template_similarities_bis - template_similarities))

# # Compute alternative template similarities.
# template_similarities = np.zeros_like(template_similarities)
# for id_1 in tqdm.tqdm(template_ids):
#     template_1 = templates[id_1, :, :]
#     support_1 = template_supports[id_1, :]
#     for id_2 in template_ids[(id_1 + 1):]:
#         template_2 = templates[id_2, :, :]
#         support_2 = template_supports[id_2, :]
#         intersection_support = support_1 & support_2
#         lag = - template_lags[id_1, id_2] + 1
#         norm = max(template_norms[id_1], template_norms[id_2])
#         if np.any(intersection_support):
#             similarity = np.dot(
#                 template_1[max(0, 0 + lag):min(nb_time_steps, nb_time_steps + lag), intersection_support].flat,
#                 template_2[max(0, 0 - lag):min(nb_time_steps, nb_time_steps - lag), intersection_support].flat
#             ) / norm ** 2.0 / float(nb_channels * nb_time_steps)
#         else:
#             similarity = 0.0
#         template_similarities[id_1, id_2] = similarity
#         template_similarities[id_2, id_1] = similarity

# ...
template_ids_1, template_ids_2 = np.meshgrid(template_ids, template_ids, indexing='ij')

# ...
union_support_sizes = np.zeros_like(template_similarities)
for id_1 in template_ids:
    for id_2 in template_ids:
        union_support_size = np.count_nonzero(template_supports[id_1, :] | template_supports[id_2, :])
        union_support_sizes[id_1, id_2] = union_support_size
        union_support_sizes[id_1, id_2] = union_support_size

# ...
norm_maximums = np.zeros_like(template_similarities)
for id_1 in template_ids:
    for id_2 in template_ids:
        norm_maximum = max(template_norms[id_1], template_norms[id_2])
        norm_maximums[id_1, id_2] = norm_maximum
        norm_maximums[id_1, id_2] = norm_maximum

# ...
norm_minimums = np.zeros_like(template_similarities)
for id_1 in template_ids:
    for id_2 in template_ids:
        norm_minimum = min(template_norms[id_1], template_norms[id_2])
        norm_minimums[id_1, id_2] = norm_minimum
        norm_minimums[id_1, id_2] = norm_minimum

# ...
nb_spikes_minimums = np.zeros_like(template_similarities)
for id_1 in template_ids:
    for id_2 in template_ids:
        nb_spikes_minimum = min(len(spike_times[id_1]), len(spike_times[id_2]))
        nb_spikes_minimums[id_1, id_2] = nb_spikes_minimum
        nb_spikes_minimums[id_2, id_1] = nb_spikes_minimum

# ...
selection = np.full_like(template_similarities, True, dtype=np.bool)
for id_ in template_ids:
    selection[id_, 0:id_] = False
# for id_ in template_ids:
#     if np.count_nonzero(template_supports[id_, :]) <= 10:
#             selection[id_, :] = False
#         selection[:, id_] = False
#     if len(spike_times[id_]) < 10000:
#         selection[id_, :] = False
#         selection[:, id_] = False

# ...
order = np.argsort(norm_minimums[selection])

# Plot template similarities vs union support sizes.
fig, ax = plt.subplots()
x = union_support_sizes[selection][order]
y = template_similarities[selection][order]
scatter_kwargs = dict(
    # s=(3 ** 2),
    s=((6 ** 2) * (nb_spikes_minimums[selection][order] / np.max(nb_spikes_minimums[selection][order]))),
    # c=norm_maximums[selection][order],
    c=norm_minimums[selection][order],
    # TODO nb. spikes?
    # TODO cross-correlation?
    vmin=0.0,
    picker=True,
)
pc = ax.scatter(x, y, **scatter_kwargs)
ids_1 = template_ids_1[selection][order]
ids_2 = template_ids_2[selection][order]
annotate_kwargs = dict(
    xy=(0, 0),
    xytext=(10, 10),
    textcoords="offset points",
    bbox=dict(
        boxstyle="round",
        fc="w",
    ),
    arrowprops=dict(
        arrowstyle="->",
    ),
)
a = ax.annotate("", **annotate_kwargs)
a.set_visible(False)
axline_kwargs = dict(
    color='black',
    linewidth=0.5,
)
ax.axvline(x=0, **axline_kwargs)
# ax.axvline(x=(nb_channels - 1), **axline_kwargs)
ax.axhline(y=0.0, **axline_kwargs)
ax.axhline(y=1.0, **axline_kwargs)

xmin, xmax = ax.get_xlim()
support = np.arange(0, xmax)
exponents = np.exp(-support/args.adapted_thr)
new_data = args.cc_merge**(1/exponents)

ax.plot(support, new_data, color='tab:red')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("size support (union)")
ax.set_ylabel("similarity")
cb = fig.colorbar(pc, ax=ax)
cb.set_label("norm (min.)")
fig.tight_layout()


def update_annotation(k):

    pos = pc.get_offsets()[k]
    a.xy = pos
    text = "({}, {})".format(ids_1[k], ids_2[k])
    a.set_text(text)

    return


def pick_handler(event):

    k = np.random.choice(event.ind)
    update_annotation(k)
    a.set_visible(True)
    fig.canvas.draw_idle()

    return


fig.canvas.mpl_connect('pick_event', pick_handler)


# plt.show()
