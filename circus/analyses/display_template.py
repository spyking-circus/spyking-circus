"""Display a given template."""
import argparse
import matplotlib.pyplot as plt
import numpy as np

from circus.shared.parser import CircusParser
from circus.shared.files import load_data


# Parse arguments.
parser = argparse.ArgumentParser(description="Display a template.")
parser.add_argument('datafile', help="data file")
parser.add_argument('-t', '--template', default=0, type=int, help="template index", dest='template_id')
args = parser.parse_args()

# Load parameters.
params = CircusParser(args.datafile)
_ = params.get_data_file()
params.get_data_file()
sampling_rate = params.rate
nb_channels = params.getint('data', 'N_e')
nb_time_steps = params.getint('detection', 'N_t')

# Load template.
templates = load_data(params, 'templates')
_, nb_template_components = templates.shape
nb_templates = nb_template_components // 2
if args.template_id < 0 or args.template_id >= nb_templates:
    raise IndexError("template index out of range (0 to {})".format(nb_templates - 1))
template = templates[:, args.template_id]  # selected the wanted template
template = template.toarray()
template = template.reshape(nb_channels, nb_time_steps)

# Display template.
fig, ax = plt.subplots()
imshow_kwargs = {
    'cmap': 'seismic',
    'aspect': 'auto',
    'vmin': - np.max(np.abs(template)),
    'vmax': + np.max(np.abs(template)),
    'extent': (
        - (float((nb_time_steps - 1) // 2) - 0.5) / sampling_rate * 1e+3,  # left
        + (float((nb_time_steps - 1) // 2) + 0.5) / sampling_rate * 1e+3,  # right
        float(nb_channels - 1) + 0.5,  # bottom
        float(0) - 0.5,  # top
    ),
}
ai = ax.imshow(template, **imshow_kwargs)
ax.axvline(x=0.0, color='black', linewidth=0.5)
ax.set_xlabel("time (ms)")
ax.set_ylabel("channel")
ax.set_title("template {}".format(args.template_id))
cb = fig.colorbar(ai, ax=ax)
cb.set_label("voltage")
fig.tight_layout()
plt.show()
