"""Inspect (some) collected waveforms."""
import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from circus.shared.parser import CircusParser


# Parse arguments.
parser = argparse.ArgumentParser(description="Inspect (some) collected waveforms.")
parser.add_argument('datafile', help="data file")
args = parser.parse_args()

# Load parameters.
params = CircusParser(args.datafile)
_ = params.get_data_file()
sampling_rate = params.rate
file_out_suff = params.get('data', 'file_out_suff')
nb_channels = params.getint('data', 'N_e')
nb_time_steps = params.getint('detection', 'N_t')

# Load collected waveforms.
basis_path = "{}.basis.hdf5".format(file_out_suff)
if not os.path.isfile(basis_path):
    raise FileNotFoundError(basis_path)
with h5py.File(basis_path, mode='r', libver='earliest') as basis_file:
    assert 'waveform' in basis_file, basis_file.keys()
    assert 'waveforms' in basis_file, basis_file.keys()
    median_waveform = basis_file['waveform'][:]
    assert median_waveform.shape == (nb_time_steps,)
    waveforms = basis_file['waveforms'][:]
    assert waveforms.shape[1] == nb_time_steps
    nb_waveforms, _ = waveforms.shape

# Plot collected waveforms.
t_min = - float((nb_time_steps - 1) // 2) / (sampling_rate * 1e-3)  # in ms
t_max = + float((nb_time_steps - 1) // 2) / (sampling_rate * 1e-3)  # in ms
times = np.linspace(t_min, t_max, num=nb_time_steps)
fig, ax = plt.subplots()
for waveform_nb in range(0, nb_waveforms):
    x = times
    y = waveforms[waveform_nb, :]
    ax.plot(x, y, c='tab:gray', linewidth=0.5, alpha=0.25)
x = times
y = median_waveform
ax.plot(x, y, c='black', label="median")
for waveform_nb in np.random.choice(nb_waveforms, 1, replace=False):
    x = times
    y = waveforms[waveform_nb, :]
    ax.plot(x, y, c='tab:blue', label="random")
ax.axhline(color='black', linewidth=0.5)
ax.axvline(color='black', linewidth=0.5)
ax.set_xlim(t_min, t_max)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel("time (ms)")
ax.set_ylabel("amplitude (arb. unit)")
ax.legend(loc='lower right')
ax.set_title("collected waveforms")
fig.tight_layout()
plt.show()
