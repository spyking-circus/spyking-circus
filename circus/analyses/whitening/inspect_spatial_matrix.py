"""Inspect the spatial whitening matrix."""
import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from circus.shared.parser import CircusParser


# Parse arguments.
parser = argparse.ArgumentParser(description="Inspect the spatial whitening matrix.")
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
    if 'spatial' not in basis_file:
        raise RuntimeError("No spatial whitening matrix found in {}.".format(basis_path))
    spatial_matrix = basis_file['spatial'][:]
    assert spatial_matrix.shape == (nb_channels, nb_channels), (spatial_matrix.shape, nb_channels)

# Plot spatial matrix.
fig, ax = plt.subplots()
image = spatial_matrix
vlim = np.max(np.abs(image))
imshow_kwargs = {
    'cmap': 'seismic',
    'vmin': - vlim,
    'vmax': + vlim,
}
ai = ax.imshow(image, **imshow_kwargs)
fig.colorbar(ai, ax=ax)
ax.set_xlabel("channel")
ax.set_ylabel("channel")
ax.set_title("spatial whitening matrix")
fig.tight_layout()
# plt.show()
