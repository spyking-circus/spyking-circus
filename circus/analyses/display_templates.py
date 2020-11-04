"""Display given templates."""
import argparse
import matplotlib.pyplot as plt
import numpy as np

from circus.shared.parser import CircusParser
from circus.shared.files import load_data
from circus.analyses.utils import plot_template


# Parse arguments.
parser = argparse.ArgumentParser(description="Display given templates.")
parser.add_argument('datafile', help="data file")
parser.add_argument('-t', '--templates', default=[0], nargs='*', type=int, help="template indices", dest='template_ids')
args = parser.parse_args()
# # Adjust arguments.
args.template_ids = np.array(args.template_ids)

# Load parameters.
params = CircusParser(args.datafile)
_ = params.get_data_file()
params.get_data_file()
sampling_rate = params.rate
nb_channels = params.getint('data', 'N_e')
nb_time_steps = params.getint('detection', 'N_t')

# Load templates.
templates = load_data(params, 'templates')
_, nb_template_components = templates.shape
nb_templates = nb_template_components // 2

# Check template indices.
for template_id in args.template_ids:
    if template_id < 0 or template_id >= nb_templates:
        raise IndexError("template index {} out of range (0 to {})".format(template_id, nb_templates - 1))

selected_templates = templates[:, args.template_ids]  # selected the wanted templates
selected_templates = selected_templates.toarray()
nb_selected_templates = args.template_ids.size
selected_templates = selected_templates.reshape(nb_channels, nb_time_steps, nb_selected_templates)
selected_templates = np.transpose(selected_templates, axes=(1, 0, 2))

# Display template.
fig, ax = plt.subplots()
for k, template_id in enumerate(args.template_ids):
    template = selected_templates[:, :, k]
    kwargs = {
        'color': 'C{}'.format(k % 10),
        'label': '{}'.format(template_id),
        'vmin': np.min(selected_templates),
        'vmax': np.max(selected_templates),
    }
    plot_template(ax, template, params, **kwargs)
ax.legend()
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("templates")
fig.tight_layout()
# plt.show()
