Parallel HDF5
=============

The code can make use of parallel HDF5, if this feature is available on your system. This will reduce, during the execution, the size consumed by temporay files on your hard drive, and also speed up the computation in various part of the algorithm. This is especially important for large number of electrodes, leading to a large number of templates (> 2000).

To know if your ``hdf5`` implementation is compatible with MPI, you just need to launch python and do::

    >> import h5py
    >> h5py.get_config().mpi

Note that by installing the code with Conda or pip, this may not be activated, as this often requires to download the source of hdf5 and compile them with the ``mpi`` option, not activated by default. To do so, just download HDF5_, and then, in the folder that we can call ``HDFHOME``, do::

    > ./configure --enable-parallel --enable-shared
    > make
    > make install

This will install the parallel libraries into ``HDFHOME/hdf5``. To have them available in your workspace, you often need to add them in your environment variable, such as for example by adding this following line to ``/home/user/.bashrc``::

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:HDFHOME/hdf5

Once this is done, you need to install h5py_ and link it toward this particular library. This can easily be done by downloading h5py_, and then in the folder, do::

    > python setup.py configure --mpi --hdf5=HDFHOME/hdf5
    > python setup.py build_ext --include-dirs=/usr/lib/openmpi/include
    > python setup.py install --user

.. _HDF5: https://www.hdfgroup.org/HDF5/release/obtain5.html
.. _h5py: https://pypi.python.org/pypi/h5py


