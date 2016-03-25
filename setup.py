import os
from os.path import join as pjoin
import sys, subprocess

requires = ['progressbar2', 'mpi4py', 'numpy', 'cython', 'scipy', 'matplotlib', 'h5py', 'colorama']

try:
  subprocess.check_call(['nvcc', '--version'])
  requires += ['cudamat==0.3circus']
  HAVE_CUDA = True
except (OSError, subprocess.CalledProcessError):
  print "CUDA not found"
  HAVE_CUDA = False

from setuptools import setup
from setuptools.command.install import install

if sys.version_info < (2, 7):
    raise RuntimeError('Only Python versions >= 2.7 are supported')

if 'CONDA_BUILD' in os.environ and 'RECIPE_DIR' in os.environ:
    # We seem to be running under a "conda build"
    data_path = pjoin('data', 'spyking-circus')
else:
    data_path = pjoin(os.path.expanduser('~'), 'spyking-circus')


class cudamat_install(install):
    '''
    This class allows the install of CUDAMAT only if GPU is detected
    '''
    def run(self):
        try:
          print "GPU DETECTED, installing CUDAMAT"
          requires = ['progressbar2', 'mpi4py', 'numpy', 'cython', 'scipy', 'matplotlib', 'h5py', 'colorama', 'cudamat==0.3circus']
          self.do_egg_install()
        except Exception:
          print "GPU not DETECTED, skipping CUDAMAT"
          requires = ['progressbar2', 'mpi4py', 'numpy', 'cython', 'scipy', 'matplotlib', 'h5py', 'colorama']
          self.do_egg_install()


setup(name='spyking-circus',
      version='0.3',
      description='Fast spike sorting by template matching',
      url='http://www.yger.net/software/spyking-circus',
      author='Pierre Yger and Olivier Marre',
      author_email='pierre.yger@inserm.fr',
      license='License :: OSI Approved :: UPMC CNRS INSERM Logiciel Libre License, version 2.1 (CeCILL-2.1)',
      packages=['circus', 'circus.shared', 'circus.scripts'],
      setup_requires=['cython', 'numpy', 'setuptools>0.18'],
      dependency_links=["https://github.com/yger/cudamat/archive/master.zip#egg=cudamat-0.3circus"],
      install_requires=requires,
      entry_points={
          'console_scripts': [
              'spyking-circus=circus.scripts.launch:main',
              'spyking-circus-subtask=circus.scripts.subtask:main',
              'circus-gui-matlab=circus.scripts.matlab_gui:main',
              'circus-gui-python=circus.scripts.python_gui:main',
              'circus-multi=circus.scripts.circus_multi:main'
          ],
      },
      package_data={'circus': ['config.params',
                               # Only include the actual GUI, not other test scripts
                               pjoin('matlab_GUI', 'SortingGUI.m'),
                               pjoin('matlab_GUI', 'SortingGUI.fig'),
                               pjoin('matlab_GUI', 'strjoin.m'),
                               pjoin('matlab_GUI', 'strsplit.m'),
                               pjoin('matlab_GUI', 'DATA_SortingGUI.m'),
                               pjoin('icons', 'gimp-tool-color-picker.png'),
                               pjoin('icons', 'gimp-tool-free-select.png'),
                               pjoin('icons', 'gimp-tool-rect-select.png')],
                    'circus.shared': ['qt_merge.ui', 'qt_preview.ui']},
      data_files=[(data_path, [pjoin('circus', 'config.params')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'mea_64.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'mea_252.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'small_mea_252.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'wide_mea_252.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'imec.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'kampff_32.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'kampff_128.prb')]),
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

msg = ['################################################################################',
'# Probes files and parameter template have been copied to $HOME/spyking-circus #']


if HAVE_CUDA:
  msg += ['# GPU support has been correctly installed                                     #']
else:
  msg += ['# GPU support was NOT installed. Be sure to have a valid nvcc command          #']

msg += ['################################################################################']

for line in msg:
  print line
