Configuration of MPI
====================

The code is able to use multiple CPU to speed up the operations. It can even use GPU during the fitting phase. However, you need to have a valid hostfile to inform MPI of what are the available nodes on your computer. By default, the code searches for the file ``circus.hosts`` in the spyking-circus folder, create during the installation ``$HOME/spyking-circus/``. Otherwise, you can provide it to the main script with the ``-H`` argument (see :doc:`documentation on the parameters <../code/parameters>`)::

    >> spyking-circus path/mydata.extesion -H mpi.hosts

Structure of the hostfile
-------------------------

Such a hostfile may depend on the fork of MPI you are using. For OpenMPI_, this will typically look like::

    192.168.0.1
    192.168.0.2
    192.168.0.3
    192.168.0.4
    192.168.0.5

If this is your parameter file, and if you launch the code with 20 CPUs::

    >> spyking-circus path/mydata.extension -c 20

Then the code will launch 4 instances of the program on the 5 nodes listed in the hostname.hosts file


.. note::
    
    If you are using multiple machines, all should read/write in a **shared** folder. This can be done with NFS_ or SAMBA_ on Windows. Usually, most clusters will provide you such a shared ``/home/user`` folder, be sure this is the case 


Shared Memory
-------------

With recent versions of MPI, you can share memory on a single machine, and this is used by the code to reduce the memory footprint. If you have large number of channels and/or templates, be sure to use a recent version of MPI_ (OpenMPI_ > 1.8.5 for example)


Handling of GPUs
----------------

By default, the code will assume that you have only one GPU per nodes. If this is not the case, then you need to specify the number of GPUs and the number of CPUs when launching the code. For example::

    >> spyking-circus path/mydata.extension -c 5 -g 10

This will tell the code that because ``n_gpu`` is larger than ``n_cpu``, several GPUs per nodes must be assumed. In this particular example, 2 GPUs per nodes. 

.. warning::

    Currently, clusters with heterogeneous numbers of GPUs per nodes are not properly handled. Be in touch if interested by the functionality
    

.. _MPI: http://www.open-mpi.org/
.. _OpenMPI: http://www.open-mpi.org/
.. _NFS: https://en.wikipedia.org/wiki/Network_File_System
.. _Samba: https://support.microsoft.com/en-us/kb/224967

