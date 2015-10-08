from setuptools import setup
import os
from os.path import join as pjoin

setup(name='circus',
      version='0.1',
      description='Fast spike sorting by template matching',
      url='http://www.yger.net/software/spyking-circus',
      author='Pierre Yger and Olivier Marre',
      author_email='pierre.yger@inserm.fr',
      license='MIT',
      packages=['circus'],
      setup_requires=['cython', 'numpy', 'setuptools>0.18'],
      extras_require={'cuda' : ['cudamat']},
      install_requires=['progressbar', 'mpi4py', 'mdp', 'hdf5storage', 'numpy', 'cython', 'scipy', 'matplotlib', 'configparser', 'h5py', 'termcolor'],
      dependency_links=['http://github.com/cudamat/cudamat/tarball/master#egg=cudamat'],
      scripts=[pjoin('bin', 'spyking-circus'),
               pjoin('bin', 'spyking-circus-subtask.py'),
               pjoin('bin', 'circus-gui')],
      package_data={'circus': ['config.params',
                               # Only include the actual GUI, not other test scripts
                               'matlab_GUI/SortingGUI.m',
                               'matlab_GUI/SortingGUI.fig',
                               'matlab_GUI/strjoin.m',
                               'matlab_GUI/strsplit.m']},
      data_files=[(os.path.expanduser('~/spyking-circus/'), ['circus/config.params'])],
      classifiers=['Development Status :: 3 - Alpha'],
      zip_safe=False)
