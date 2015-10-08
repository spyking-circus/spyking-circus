Parameters
==========

Command line Parameters
-----------------------

To know what are all the parameters of the software, just do::
    
    >> spyking-circus -h

The parameters to launch the program are:

* ``-m`` or ``--method``

What are the steps of the algorithm you would like to perform. Defaults steps are:
1. filtering
2. whitening
3. clustering
4. fitting

Note that filtering is performed only once, and if the code is relaunched on the same data, a flag in the parameter file will prevent the code to filter twice. You can specify only a subset of steps by doing::
    
    >> spyking-circus path/mydata.extension -m clustering,fitting

* ``-c`` or ``--n_cpu``

The number of CPU that will be used by the code, at least during the first three steps. Note that if CUDA is present, and if the GPU are not turned off (with -g 0), GPUs are always prefered to CPU during the fitting phase. 

For example, just do::

    >> spyking-circus path/mydata.extension -m clustering,fitting -c 10    

* ``-g`` or ``--n_gpu``

The number of GPU that will be used by the code during the fitting phase. If you have CUDA, but a slow GPU and a lot of CPUs (for example 10), you can disable the GPU usage by setting::
    
    >> spyking-circus path/mydata.extension -g 0 -c 10

.. warning::

    Currently, nodes with several GPUs are not properly handled. Be in touch if interested by the functionnality

* ``-H`` or ``--hostfile``

The CPUs used depends on your MPI configuration. If you wan to configure them, you must provide a specific hostfile and do::

    >> spyking-circus path/mydata.extension -c 10 -H nodes.hosts

To know more about the host file, see the MPI section :doc:`documentation on MPI <../introduction/mpi>`


Configuration File
------------------

The code, when launched for the first time, generates a parameter file. To know more about it, :doc:`documentation on the configuration <../code/config>`