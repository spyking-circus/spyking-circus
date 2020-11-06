"""Inspect amplitude probability density functions."""
import argparse
import matplotlib.pyplot as plt
import numpy as np

from circus.shared.parser import CircusParser
from circus.shared.files import load_data


# Parse arguments.
parser = argparse.ArgumentParser(description="Inspect amplitude probability density functions.")
parser.add_argument('datafile', help="data file")
parser.add_argument('-e', '--extension', default=None, help="data extension")
parser.add_argument('-o', '--order', choices=['argmax'], help="template ordering")
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

# Load amplitudes.
results = load_data(params, 'results', extension=args.extension)
amplitudes = dict()
for key in results['amplitudes']:
    assert key[:5] == 'temp_', key
    template_id = int(key[5:])
    amplitudes[template_id] = results['amplitudes'][key][:, 0]

# Compute amplitude distributions.
template_ids = np.sort(amplitudes.keys())
a_min = 0.0
# a_min = 1.0
# for template_id in template_ids:
#     if amplitudes[template_id].size > 0:
#         a_min = min(a_min, np.min(amplitudes[template_id]))
a_max = 2.0
for template_id in template_ids:
    if amplitudes[template_id].size > 0:
        a_max = max(a_max, np.max(amplitudes[template_id]))
nb_templates = template_ids.size
# nb_bins = 500
nb_bins = 250
amplitude_pdfs = np.zeros((nb_templates, nb_bins))
for template_id in template_ids:
    amplitude_pdfs[template_id, :], _ = np.histogram(
        amplitudes[template_id], bins=nb_bins, range=(a_min, a_max), density=True
    )
amplitude_pdfs[np.isnan(amplitude_pdfs)] = 0.0
assert not np.any(np.isnan(amplitude_pdfs))

# Reorder templates (if necessary).
if args.order == 'argmax':
    order = np.argsort(np.argmax(amplitude_pdfs, axis=1))
else:
    order = np.argsort(template_ids)

# # Mask zeros.
# image = np.ma.masked_equal(amplitude_pdfs, 0.0)

# Display amplitudes over time.
fig, ax = plt.subplots()
imshow_kwargs = {
    'cmap': 'Greys',
    'aspect': 'auto',
    'vmin': 0.0,
    'extent': (
        a_min,
        a_max,
        float(nb_templates - 1) + 0.5,
        float(0) - 0.5,
    )
}
ai = ax.imshow(amplitude_pdfs[order, :], **imshow_kwargs)
ax.axvline(x=1.0, color='black', linewidth=0.5, alpha=0.50)
ax.axvline(x=0.5, color='black', linewidth=0.5, alpha=0.25)
ax.axvline(x=1.5, color='black', linewidth=0.5, alpha=0.25)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_yticks([]) if args.order is not None else None
ax.set_xlabel("amplitude")
ax.set_ylabel("template (reordered)" if args.order is not None else "template")
ax.set_title("amplitude PDFs")
cb = fig.colorbar(ai, ax=ax)
cb.set_label("probability density")
fig.tight_layout()
# plt.show()
