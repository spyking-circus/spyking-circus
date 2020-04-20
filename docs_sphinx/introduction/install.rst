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
    
    We recommend using Anaconda_, with a simple install:

    * :doc:`see here for detailed instructions on Windows <../introduction/windows>` 
    * :doc:`see here for detailed instructions on Mac OS X <../introduction/mac>`


Using with CONDA
~~~~~~~~~~~~~~~~

Install Anaconda_ or miniconda_, e.g. all on the terminal (but there is also a .exe installer for Windows, etc.):

As an example for linux, just type::

    >> wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
    >> bash Miniconda-latest-Linux-x86_64.sh

If you want, first, the best is to create a dedicated environment::

    >> conda create -n circus python=3.6

Then activate the environment::

    >> conda activate circus

Then install the software itself::

    (circus) >> conda install -c conda-forge -c intel -c spyking-circus spyking-circus


Using pip
~~~~~~~~~

To do so, use the ``pip`` utility::

    >> pip install spyking-circus


Note that if you are using a linux distribution, you must be sure that you have ``mpich`` instead of ``openmpi`` (default on Ubuntu). To do that, please do::

    >> sudo apt remove openmpi
    >> sudo apt install mpich libmpich-dev

And to be sure that mpi4py is not installed with precompiled binary that would link with openmpi, you need to do::

    >> pip install spyking-circus --no-binary=mpi4py

You might want to add the ``--user`` flag, to install SpyKING CIRCUS for the local user only, which means that you don't need administrator privileges for the installation.

In principle, the above command also install SpyKING CIRCUS's dependencies. Once the install is complete, you need to add the ``PATH`` where SpyKING CIRCUS has been installed into your local ``PATH``, if not already the case. To do so, simply edit your ``$HOME/.bashrc`` and add the following line::

    export PATH=$PATH:$HOME/.local/bin

Then you have to relaunch the shell, and you should now have the SpyKING CIRCUS installed!

Using sources
~~~~~~~~~~~~~

Alternatively, you can download the source package directly and uncompress it, or work directly with the git folder https://github.com/spyking-circus/spyking-circus to be in sync with bug fixes. You can then simply run::

    >> pip install . --user

Or even better, you can install it with the develop mode::

    >> pip install . -e --user


Such that if you do a git pull in the software directory, you do not need to reinstall it.


For those that are not pip users, it is equivalent to::

    >> python setup.py install

Or to keep the folder in sync with the install in a develop mode::

    >> python setup.py develop 


.. note::

    If you want to install ``scikit-learn``, needed to get the BEER estimates, you need to add ``[beer]`` to any pip install


.. note::

    If you experience some issues with Qt or pyQt, you may need to install it manually on your system. For linux users, simply use your software distribution system (apt for example). For windows user, please see `here <http://doc.qt.io/qt-5/windows-support.html>`_


Installing phy 2.0
~~~~~~~~~~~~~~~~~~

If you want to use the phy GUI to visualize your results, you may need to install phy_ 2.0 (only compatible with python 3). If you have installed SpyKING CIRCUS within a conda environment, first activate it::

    >> conda activate circus

Once this is done, install phy_ 2.0::

    (circus) >> pip install colorcet pyopengl qtconsole requests traitlets tqdm joblib click mkdocs dask toolz mtscomp
    (circus) >> pip install --upgrade https://github.com/cortex-lab/phy/archive/master.zip
    (circus) >> pip install --upgrade https://github.com/cortex-lab/phylib/archive/master.zip

You can see more details on the `phy website <https://phy.readthedocs.io/en/latest/installation/>`_


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
    1. ``tqdm`` 
    2. ``mpi4py`` 
    3. ``numpy`` 
    4. ``cython`` 
    5. ``scipy``
    6. ``matplotlib`` 
    7. ``h5py``
    8. ``colorama``
    9. ``blosc``
    10. ``scikit-learn``
    11. ``statsmodels``
    
.. _Anaconda: https://www.anaconda.com/distribution/
.. _miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _MPI: https://www.mpich.org/
.. _Xcode: https://developer.apple.com/xcode/download/
.. _HDF5: https://www.hdfgroup.org
.. _phy: https://github.com/cortex-lab/phy
