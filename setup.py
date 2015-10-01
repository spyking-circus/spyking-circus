from setuptools import setup
from os.path import join as pjoin, dirname

setup(name='circus',
    version='0.1',
    description='Fast spike sorting by template matching',
    url='http://www.yger.net/software/spyking-circus',
    author='Pierre Yger and Olivier Marre',
    author_email='pierre.yger@inserm.fr',
    license='MIT',
    packages=['circus'],
    install_requires=['progressbar', 'mpi4py', 'mdp', 'hdf5storage', 'numpy', 'scipy', 'matplotlib', 'configparser', 'h5py', 'cudamat'],
    scripts = [pjoin('bin', 'spyking-circus')],
    classifiers=['Development Status :: 3 - Alpha'],
    data_files=[('scripts', ['circus/whitening.py', 
                          'circus/basis.py', 
                          'circus/clustering.py', 
                          'circus/fitting.py',
                          'circus/gathering.py',
                          'circus/extracting.py',
                          'circus/infos.py',
                          'circus/export_phy.py'])],
    zip_safe=False)
