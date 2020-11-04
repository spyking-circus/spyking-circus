"""Inspect template similarities."""
import argparse
import matplotlib as mpl
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np

from circus.shared.parser import CircusParser
from circus.analyses.utils import load_templates_data


# Parse arguments.
parser = argparse.ArgumentParser(description="Inspect template similarities.")
parser.add_argument('datafile', help="data file")
parser.add_argument(
    '-t', '--templates', default=None, nargs='*', type=int, help="template indices to restrict to", dest='template_ids'
)
args = parser.parse_args()

# Load parameters.
params = CircusParser(args.datafile)
_ = params.get_data_file()
sampling_rate = params.rate  # Hz
file_out_suff = params.get('data', 'file_out_suff')
nb_time_steps = params.getint('detection', 'N_t')

# Load maximum values of the overlaps as template similarities.
templates_data = load_templates_data(params, extension='', keys=['maxoverlap', 'maxlag'])
template_similarities = templates_data['maxoverlap']
template_lags = templates_data['maxlag'] / sampling_rate * 1e+3  # ms
template_lag_max = nb_time_steps / sampling_rate * 1e+3  # ms

assert np.min(template_similarities) >= 0.0, np.min(template_similarities)

# Plot template similarities.
fig, ax = plt.subplots()
if args.template_ids is None:
    image = template_similarities
    imshow_kwargs = {
        'cmap': 'Greys',
        'vmin': 0.0,
    }
    ai = ax.imshow(image, **imshow_kwargs)
else:
    image = template_similarities[args.template_ids, :][:, args.template_ids]
    imshow_kwargs = {
        'cmap': 'Greys',
        'vmin': 0.0,
    }
    ai = ax.imshow(image, **imshow_kwargs)
    ax.set_xticks([k for k, _ in enumerate(args.template_ids)])
    ax.set_xticklabels([template_id for template_id in args.template_ids])
    ax.set_yticks([k for k, _ in enumerate(args.template_ids)])
    ax.set_yticklabels([template_id for template_id in args.template_ids])
ax.set_xlabel("template")
ax.set_ylabel("template")
ax.set_title("template similarities")
cb = fig.colorbar(ai, ax=ax)
cb.set_label("similarity")
cb.ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 2))
fig.tight_layout()
# plt.show()


# Plot template lags.
fig, ax = plt.subplots()
cmap = mpl.cm.get_cmap('seismic')
cmap.set_bad(color='tab:gray', alpha=0.25)
imshow_kwargs = {
    'cmap': cmap,
    'vmin': - template_lag_max,
    'vmax': + template_lag_max,
}
if args.template_ids is None:
    mask = (np.abs(template_lags) >= template_lag_max)
    image = np.ma.array(template_lags, mask=mask)
    ai = ax.imshow(image, **imshow_kwargs)
else:
    image = template_lags[args.template_ids, :][:, args.template_ids]
    ai = ax.imshow(image, **imshow_kwargs)
    ax.set_xticks([k for k, _ in enumerate(args.template_ids)])
    ax.set_xticklabels([template_id for template_id in args.template_ids])
    ax.set_yticks([k for k, _ in enumerate(args.template_ids)])
    ax.set_yticklabels([template_id for template_id in args.template_ids])
ax.set_xlabel("template")
ax.set_ylabel("template")
ax.set_title("template lags")
cb = fig.colorbar(ai, ax=ax)
cb.set_label("lag (ms)")
cb.ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
fig.tight_layout()
# plt.show()
