Configuration of MPI
====================

The code is able to use multiple CPU to speed up the operations. It can even use GPU during the fitting phase. However, you need to have a valid hostfile to inform MPI of what are the available nodes on your computer. By default, the code searches for the file ``circus.hosts`` in the spyking-circus folder, create during the installation ``$HOME/spyking-circus/``. Otherwise, you can provide it to the main script with the ``-H`` argument (see :doc:`documentation on the parameters <../code/parameters>`)::

    >> spyking-circus path/mydata.extesion -H mpi.hosts

Structure of the hostfile
-------------------------

Such a hostfile may depend on the fork of MPI you are using. For MPICH_, this will typically look like (if you want to use only 4 cores per machine)::

    192.168.0.1:4
    192.168.0.2:4
    192.168.0.3:4
    192.168.0.4:4
    192.168.0.5:4

For OpenMPI_, this will typically look like (if you want to use only 4 cores per machine)::

    192.168.0.1 max-slots=4
    192.168.0.2 max-slots=4
    192.168.0.3 max-slots=4
    192.168.0.4 max-slots=4
    192.168.0.5 max-slots=4

If this is your parameter file, and if you launch the code with 20 CPUs::

    >> spyking-circus path/mydata.extension -c 20

Then the code will launch 4 instances of the program on the 5 nodes listed in the hostname.hosts file


.. note::
    
    If you are using multiple machines, all should read/write in a **shared** folder. This can be done with NFS_ or SAMBA_ on Windows. Usually, most clusters will provide you such a shared ``/home/user`` folder, be sure this is the case 

.. warning::
    
    For now, the code is working with MPICH_ versions higher than 3.0, and OpenMPI_ versions below 3.0. We plan to make this more uniform in a near future, but the two softwares made different implementation choices for the MPI library


Shared Memory
-------------

With recent versions of MPI, you can share memory on a single machine, and this is used by the code to reduce the memory footprint. If you have large number of channels and/or templates, be sure to use a recent version of MPICH_ (>= 3.0) or OpenMPI_ (> 1.8.5)
    

.. _MPICH: https://www.mpich.org/
.. _OpenMPI: https://www.mpich.org/
.. _NFS: https://en.wikipedia.org/wiki/Network_File_System
.. _Samba: https://support.microsoft.com/en-us/kb/224967

