Installation
============

The SpyKING CIRCUS comes as a python package, and it at this stage, note that only unix systems have been tested

How to install
--------------

Installation with CONDA
~~~~~~~~~~~~~~~~~~~~~~~

Install Anaconda_ or miniconda_, e.g. all on the terminal (but there is also a .exe installer for Windows, etc.):

For linux, just type::

    >> wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
    >> bash Miniconda-latest-Linux-x86_64.sh 

Add a custom channel for ``hdf5storage`` and ``termcolor`` packages::

    >> conda config --add channels auto 

Then install the software itself::

    >> conda install spyking-circus-0.1-0.tar.bz2

Installation with pip
~~~~~~~~~~~~~~~~~~~~~

To do so, use the ``pip`` utility::

    pip install spyking-circus-0.1.tar.gz

You might want to add the ``--user`` flag, to install SpyKING CIRCUS for the local user
only, which means that you don't need administrator privileges for the
installation.

In principle, the above command also install SpyKING CIRCUS's dependencies. Once the install is complete, you need to add the ``PATH`` where SpyKING CIRCUS has been installed into your local ``PATH``, if not already the case. To do so, simply edit your ``/home/user/.bashrc`` and add the following line::

    export PATH=$PATH:$HOME/.local/bin

Then you have to relaunch the shell, and you should now have the SpyKING CIRCUS installed!

Installation from source
~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, you can download the source package directly and uncompress it. You can then simply run::

    >> python setup.py install --prefix=/home/user


Creation of a home Directory
----------------------------

During the install, the code will create a ``spyking-circus`` folder in ``/home/user`` where it will copy several probe designs, and a copy of the default parameter file. Note that if you are always using a similar setup, you can edit this template, as this is the one that will be used by default.


Parallelism
-----------

Using MPI
~~~~~~~~~

If you are planning to use MPI_, the best solution is to create a file ``/home/user/spyking-circus/circus.hosts`` with the lists of available nodes (see :doc:`Configuration of MPI <../introduction/mpi>`)

Using CUDA
~~~~~~~~~~

Using CUDA_ is highly recommended since it can **drastically** increase the speed of algorithm. To use it, you need to have a working CUDA_ environment installed onto the machine. To install CUDAMAT and get support for the GPU, just do::

    >> pip install https://github.com/cudamat/cudamat/archive/master.zip

.. note::
    You must have a valid CUDA installation, and ``nvcc`` installed.


Dependencies
------------

For information, here is the list of all the dependencies required by the SpyKING CIRCUS:
    1. ``progressbar`` 
    2. ``mpi4py`` 
    3. ``mdp``
    4. ``numpy`` 
    5. ``cython`` 
    6. ``scipy``
    7. ``matplotlib`` 
    8. ``h5py``
    9. ``hdf5storage`` 
    10. ``termcolor``

.. _CUDA: https://developer.nvidia.com/cuda-downloads
.. _Anaconda: https://www.continuum.io/downloads
.. _miniconda: http://conda.pydata.org/miniconda.html
.. _OpenMPI: http://www.open-mpi.org/
