import os
from os.path import join as pjoin
import sys

from setuptools import setup

if sys.version_info < (2, 7):
    raise RuntimeError('Only Python versions >= 2.7 are supported')

if 'CONDA_BUILD' in os.environ and 'RECIPE_DIR' in os.environ:
    # We seem to be running under a "conda build"
    data_path = pjoin('data', 'spyking-circus')
else:
    data_path = pjoin(os.path.expanduser('~'), 'spyking-circus')

setup(name='spyking-circus',
      version='0.3',
      description='Fast spike sorting by template matching',
      url='http://www.yger.net/software/spyking-circus',
      author='Pierre Yger and Olivier Marre',
      author_email='pierre.yger@inserm.fr',
      license='License :: OSI Approved :: UPMC CNRS INSERM Logiciel Libre License, version 2.1 (CeCILL-2.1)',
      packages=['circus', 'circus.shared'],
      setup_requires=['cython', 'numpy', 'setuptools>0.18'],
      install_requires=['progressbar', 'mpi4py', 'numpy', 'cython', 'scipy', 'matplotlib', 'h5py', 'termcolor', 'colorama'],
      scripts=[pjoin('bin', 'spyking-circus'),
               pjoin('bin', 'spyking-circus-subtask.py'),
               pjoin('bin', 'circus-gui'),
	       pjoin('bin', 'circus-multi')],
      package_data={'circus': ['config.params',
                               # Only include the actual GUI, not other test scripts
                               pjoin('matlab_GUI', 'SortingGUI.m'),
                               pjoin('matlab_GUI', 'SortingGUI.fig'),
                               pjoin('matlab_GUI', 'strjoin.m'),
                               pjoin('matlab_GUI', 'strsplit.m'),
                               pjoin('matlab_GUI', 'DATA_SortingGUI.m'),
                               pjoin('icons', 'gimp-tool-color-picker.png'),
                               pjoin('icons', 'gimp-tool-free-select.png'),
                               pjoin('icons', 'gimp-tool-rect-select.png')]},
      data_files=[(data_path, [pjoin('circus', 'config.params')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'mea_252.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'small_mea_252.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'wide_mea_252.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'groundtruth.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'imec.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'mea_4225.prb')])],
      use_2to3=True,
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: UPMC CNRS INSERM Logiciel Libre License, version 2.1 (CeCILL-2.1)',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Bio-Informatics'
      ],
      zip_safe=False)
