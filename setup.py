import os
from os.path import join as pjoin
import sys

from setuptools import setup

if 'CONDA_BUILD' in os.environ and 'RECIPE_DIR' in os.environ:
    # We seem to be running under a "conda build"
    data_path = pjoin('data', 'spyking-circus')
else:
    data_path = pjoin(os.path.expanduser('~'), 'spyking-circus')

setup(name='spyking-circus',
      version='0.1',
      description='Fast spike sorting by template matching',
      url='http://www.yger.net/software/spyking-circus',
      author='Pierre Yger and Olivier Marre',
      author_email='pierre.yger@inserm.fr',
      license='MIT',
      packages=['circus', 'circus.shared'],
      setup_requires=['cython', 'numpy', 'setuptools>0.18'],
      install_requires=['progressbar', 'mpi4py', 'mdp', 'numpy', 'cython', 'scipy', 'matplotlib', 'h5py', 'hdf5storage', 'termcolor'],
      scripts=[pjoin('bin', 'spyking-circus'),
               pjoin('bin', 'spyking-circus-subtask.py'),
               pjoin('bin', 'circus-gui')],
      package_data={'circus': ['config.params',
                               # Only include the actual GUI, not other test scripts
                               'matlab_GUI/SortingGUI.m',
                               'matlab_GUI/SortingGUI.fig',
                               'matlab_GUI/strjoin.m',
                               'matlab_GUI/strsplit.m']},
      data_files=[(data_path, ['circus/config.params']),
                  (pjoin(data_path, 'probes/'), ['probes/mea_252.prb']),
                  (pjoin(data_path, 'probes/'), ['probes/small_mea_252.prb']),
                  (pjoin(data_path, 'probes/'), ['probes/wide_mea_252.prb']),
                  (pjoin(data_path, 'probes/'), ['probes/groundtruth.prb']),
                  (pjoin(data_path, 'probes/'), ['probes/imec.prb']),
                  (pjoin(data_path, 'probes/'), ['probes/mea_4225.prb'])],
      classifiers=['Development Status :: 3 - Alpha'],
      zip_safe=False)
