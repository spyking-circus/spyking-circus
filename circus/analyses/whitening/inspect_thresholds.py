"""Inspect thresholds."""
import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from circus.shared.parser import CircusParser


# Parse arguments.
parser = argparse.ArgumentParser(description="Inspect threhsolds.")
parser.add_argument('datafile', help="data file")
args = parser.parse_args()

# Load parameters.
params = CircusParser(args.datafile)
_ = params.get_data_file()
file_out_suff = params.get('data', 'file_out_suff')
nb_channels = params.getint('data', 'N_e')

# Load spatial matrix.
basis_path = "{}.basis.hdf5".format(file_out_suff)
if not os.path.isfile(basis_path):
    raise FileNotFoundError(basis_path)
with h5py.File(basis_path, mode='r', libver='earliest') as basis_file:
    if 'thresholds' not in basis_file:
        raise RuntimeError("No thresholds found in {}".format(basis_path))
    thresholds = basis_file['thresholds'][:]
    assert thresholds.shape == (nb_channels,), (thresholds.shape, nb_channels)

# Plot thresholds.
fig, axes = plt.subplots(nrows=2, squeeze=False, sharex=True)
# # 1st subplot.
ax = axes[0, 0]
x = np.arange(0, nb_channels)
y = thresholds
ax.scatter(x, y, s=(3 ** 2), color='black')
ax.axhline(color='black', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.set_xlabel("channel")
ax.set_ylabel("threshold")
ax.set_title("thresholds")
# # 2nd subplot.
ax = axes[1, 0]
x = np.arange(0, nb_channels)
y = thresholds
ax.scatter(x, y, s=(2 ** 2), color='black')
# ax.axhline(color='black', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel("channel")
ax.set_ylabel("threshold")
# ax.set_title("thresholds")
fig.tight_layout()
plt.show()
