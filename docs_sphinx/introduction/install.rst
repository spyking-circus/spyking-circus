Installation
============

The SpyKING CIRCUS comes as a python package, and it at this stage, note that mostly unix systems have been tested. However, users managed to get the software running on Mac OS X, and on Windows 7,8, or 10. We are doing our best, using your feedbacks, to improve the packaging and make the whole process as smooth as possible on all platforms. 

.. image::  https://anaconda.org/spyking-circus/spyking-circus/badges/version.svg
    :target: https://anaconda.org/spyking-circus/spyking-circus

.. image::  https://badge.fury.io/py/spyking-circus.svg
    :target: https://badge.fury.io/py/spyking-circus

How to install
--------------

.. note::
    
    If you are a Windows or a Mac user, we recommend using Anaconda_, and:

    * :doc:`see here for detailed instructions on Windows <../introduction/windows>` 
    * :doc:`see here for detailed instructions on Mac OS X <../introduction/mac>`


Using with CONDA
~~~~~~~~~~~~~~~~

Install Anaconda_ or miniconda_, e.g. all on the terminal (but there is also a .exe installer for Windows, etc.):

As an example for linux, just type::

    >> wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
    >> bash Miniconda-latest-Linux-x86_64.sh

Then install the software itself::

    >> conda install -c spyking-circus progressbar2
    >> conda install -c mpi4py mpi4py
    >> conda install -c spyking-circus/label/dev spyking-circus


If you want to get a support for GPU, see the devoted section on CUDA_.

Using pip
~~~~~~~~~

To do so, use the ``pip`` utility::

    >> pip install spyking-circus --process-dependency-links --pre

You might want to add the ``--user`` flag, to install SpyKING CIRCUS for the local user only, which means that you don't need administrator privileges for the installation.

In principle, the above command also install SpyKING CIRCUS's dependencies, and CUDA_ support if ``nvcc`` command is found in your environment. Once the install is complete, you need to add the ``PATH`` where SpyKING CIRCUS has been installed into your local ``PATH``, if not already the case. To do so, simply edit your ``$HOME/.bashrc`` and add the following line::

    export PATH=$PATH:$HOME/.local/bin

Then you have to relaunch the shell, and you should now have the SpyKING CIRCUS installed!

Using sources
~~~~~~~~~~~~~

Alternatively, you can download the source package directly and uncompress it, or work directly with the git folder https://github.com/spyking-circus/spyking-circus to be in sync with bug fixes. You can then simply run::

    >> pip install . --user --process-dependency-links

Or even better, you can install it with the develop mode::

    >> pip install . -e --user --process-dependency-links


Such that if you do a git pull in the software directory, you do not need to reinstall it.


For those that are not pip users, it is equivalent to::

    >> python setup.py install

Or to keep the folder in sync with the install in a develop mode::

    >> python setup.py develop 


.. note::

    If you want to install ``scikit-learn``, needed to get the BEER estimates, you need to add ``[beer]`` to any pip install


.. note::

    If you experience some issues with Qt4 or pyQt, you may need to install it manually on your system. For linux users, simply use your software distribution system (apt for example). For windows user, please see `here <http://doc.qt.io/qt-5/windows-support.html>`_



Activating CUDA
---------------

Using CUDA_ can, depending on your hardware, **drastically** increase the speed of algorithm. However, in 0.5, 1 GPU is faster than 1 CPU but not faster than several CPUs. This is something we are working on to improve in future releases. To use your GPU, you need to have a working CUDA_ environment installed onto the machine. During the pip installation, the code should automatically detect it and install CUDA_ bindings if possible. Otherwise, to get support for the GPU with an Anaconda_ install, just do::

    >> pip install https://github.com/yger/cudamat/archive/master.zip#egg=cudamat-0.3circus

.. note::
    You must have a valid CUDA installation, and ``nvcc`` installed. If you do not want CUDAMAT to be install automatically, simply do ``python setup.py install --nocuda`` while installing the software


Home Directory
--------------

During the install, the code creates a ``spyking-circus`` folder in ``/home/user`` where it will copy several probe designs, and a copy of the default parameter file. Note that if you are always using the code with a similar setup, you can edit this template, as this is the one that will be used by default.

Parallelism
-----------

Using MPI
~~~~~~~~~

If you are planning to use MPI_, the best solution is to create a file ``$HOME/spyking-circus/circus.hosts`` with the lists of available nodes (see :doc:`Configuration of MPI <../introduction/mpi>`). You should also make sure, for large number of electrodes, that your MPI implementation is compatible recent enough such that it can allow shared memory within processes.

Using HDF5 with MPI
~~~~~~~~~~~~~~~~~~~

If you are planning to use large number of electrodes (> 500), then you may use the fact that the code can use parallel HDF5_. This will speed everything and reduce disk usage. To know more about how to activate it, see (see :doc:`Parallel HDF5 <../introduction/hdf5>`). 


Dependencies
------------

For information, here is the list of all the dependencies required by the SpyKING CIRCUS:
    1. ``progressbar2`` 
    2. ``mpi4py`` 
    3. ``numpy`` 
    4. ``cython`` 
    5. ``scipy``
    6. ``matplotlib`` 
    7. ``h5py``
    8. ``colorama``
    9. ``cudamat`` [optional, CUDA_ only]
    10. ``sklearn`` [optional, only for BEER estimate]

.. _CUDA: https://developer.nvidia.com/cuda-downloads
.. _Anaconda: https://www.continuum.io/downloads
.. _miniconda: http://conda.pydata.org/miniconda.html
.. _MPI: http://www.open-mpi.org/
.. _Xcode: https://developer.apple.com/xcode/download/
.. _HDF5: https://www.hdfgroup.org
