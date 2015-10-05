from setuptools import setup
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
      install_requires=['progressbar', 'mpi4py', 'mdp', 'hdf5storage', 'numpy', 'cython', 'scipy', 'matplotlib', 'configparser', 'h5py', 'termcolor', 'cudamat'],
      dependency_links=['http://github.com/cudamat/cudamat/tarball/master#egg=cudamat'],
      scripts=[pjoin('bin', 'spyking-circus'),
               pjoin('bin', 'spyking-circus-subtask.py')],
      classifiers=['Development Status :: 3 - Alpha'],
      zip_safe=False)
