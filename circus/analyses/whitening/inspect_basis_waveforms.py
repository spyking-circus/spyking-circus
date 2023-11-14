"""Inspect basis waveforms."""
import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from circus.shared.parser import CircusParser


# Parse arguments.
parser = argparse.ArgumentParser(description="Inspect basis waveforms.")
parser.add_argument('datafile', help="data file")
args = parser.parse_args()

# Load parameters.
params = CircusParser(args.datafile)
_ = params.get_data_file()
sampling_rate = params.rate
file_out_suff = params.get('data', 'file_out_suff')
nb_channels = params.getint('data', 'N_e')
nb_time_steps = params.getint('detection', 'N_t')

# Load basis waveforms.
basis_path = "{}.basis.hdf5".format(file_out_suff)
if not os.path.isfile(basis_path):
    raise FileNotFoundError(basis_path)
with h5py.File(basis_path, mode='r', libver='earliest') as basis_file:
    if 'proj' not in basis_file:
        raise RuntimeError("No projection matrix found in {}.".format(basis_path))
    projection_matrix = basis_file['proj'][:]
    assert projection_matrix.shape[0] == nb_time_steps, (projection_matrix.shape, nb_time_steps)
    _, nb_components = projection_matrix.shape

# Plot basis waveforms.
t_min = - float((nb_time_steps - 1) // 2) / (sampling_rate * 1e-3)  # in ms
t_max = + float((nb_time_steps - 1) // 2) / (sampling_rate * 1e-3)  # in ms
times = np.linspace(t_min, t_max, num=nb_time_steps)
fig, ax = plt.subplots()
for component_nb in range(0, nb_components):
    x = times
    y = projection_matrix[:, component_nb]
    plot_kwargs = {
        'c': 'C{}'.format(component_nb),
        'label': "{}".format(component_nb),
    }
    ax.plot(x, y, **plot_kwargs)
ax.axhline(color='black', linewidth=0.5)
ax.axvline(color='black', linewidth=0.5)
ax.set_xlim(t_min, t_max)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel("time (ms)")
ax.set_ylabel("amplitude (arb. unit)")
ax.set_title("basis waveforms")
ax.legend()
fig.tight_layout()
# plt.show()
