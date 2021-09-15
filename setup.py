
import os
from os.path import join as pjoin
from setuptools import setup
import sys
# import subprocess
import re
import versioneer


requires = [
    'mpi4py', 'numpy', 'cython', 'scipy', 'matplotlib', 'h5py', 'colorama',
    'psutil', 'tqdm', 'blosc', 'statsmodels', 'setuptools',
]


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


if sys.version_info < (2, 7):
    raise RuntimeError('Only Python versions >= 2.7 are supported')

if 'CONDA_BUILD' in os.environ and 'RECIPE_DIR' in os.environ:
    # We seem to be running under a "conda build"
    data_path = pjoin('data', 'spyking-circus')
else:
    data_path = pjoin(os.path.expanduser('~'), 'spyking-circus')


def _package_tree(pkgroot):
    path = os.path.dirname(__file__)
    subdirs = [
        os.path.relpath(i[0], path).replace(os.path.sep, '.')
        for i in os.walk(os.path.join(path, pkgroot))
        if '__init__.py' in i[2]
    ]
    return subdirs


setup(name='spyking-circus',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Fast spike sorting by template matching',
      long_description=read('README.rst'),
      url='http://spyking-circus.rtfd.org',
      author='Pierre Yger and Olivier Marre',
      author_email='pierre.yger@inserm.fr',
      license='License :: OSI Approved :: CeCILL-2.1',
      keywords="spike sorting template matching tetrodes extracellular",
      packages=_package_tree('circus'),
      setup_requires=['setuptools'],
      # dependency_links=["https://github.com/yger/cudamat/archive/master.zip#egg=cudamat-0.3circus"],
      install_requires=requires,
      entry_points={
          'console_scripts': [
              'spyking-circus=circus.scripts.launch:main',
              'spyking-circus-subtask=circus.scripts.subtask:main',
              'circus-multi=circus.scripts.circus_multi:main',
              'circus-folders=circus.scripts.circus_folders:main',
              'circus-artefacts=circus.scripts.circus_artefacts:main',
              'circus-gui-matlab=circus.scripts.matlab_gui:main',
              'circus-gui-python=circus.scripts.python_gui:main'
          ],
          'gui_scripts': [
              'spyking-circus-launcher=circus.scripts.launch_gui:main'
          ]
      },
      extras_require={'beer': ['scikit-learn']},
      package_data={'circus': ['config.params',
                               # Only include the actual GUI, not other test scripts
                               pjoin('matlab_GUI', 'SortingGUI.m'),
                               pjoin('matlab_GUI', 'SortingGUI.fig'),
                               pjoin('matlab_GUI', 'circusjoin.m'),
                               pjoin('matlab_GUI', 'circussplit.m'),
                               pjoin('matlab_GUI', 'DATA_SortingGUI.m'),
                               pjoin('icons', 'gimp-tool-color-picker.png'),
                               pjoin('icons', 'gimp-tool-free-select.png'),
                               pjoin('icons', 'logo.jpg'),
                               pjoin('icons', 'icon.png'),
                               pjoin('icons', 'gimp-tool-rect-select.png'),
                               pjoin('qt_GUI', 'qt_merge.ui'),
                               pjoin('qt_GUI', 'qt_preview.ui'),
                               pjoin('qt_GUI', 'qt_launcher.ui')]},
      data_files=[(data_path, [pjoin('circus', 'config.params')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'mea_64.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'mea_32.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'small_mea_252.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'small_mea_256.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'wide_mea_252.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'wide_mea_256.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'imec.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'kampff_32.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'kampff_128.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'mea_4225.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'adrien.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'emilie.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'dan.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'ASSY-01-P.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'ASSY-37-P.prb')]),
                  (pjoin(data_path, 'probes'), [pjoin('probes', 'ASSY-77-P.prb')])],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'License :: Other/Proprietary License',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Bio-Informatics'
      ],
      zip_safe=False)

msg = [
    '################################################################################',
    '# Probes files and parameter template have been copied to $HOME/spyking-circus #',
    '# If you want to use the phy GUI, please see documentation to install it       #',
    '################################################################################'
]

for line in msg:
    print(line)
