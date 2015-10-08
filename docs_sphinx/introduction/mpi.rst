Installation of MPI
===================

The code is able to use multiple CPU to speed up the operations. However, you need to have a valid hostfile to inform MPI of what are the available nodes on your computer. By default, the code searches for the file ``circus.hosts`` in the spyking-circus folder, create during the installation ``/home/user/spyking-circus/``. Otherwise, you can provide it to the main script with the -H argument (see :doc:`documentation on the parameters <../code/parameters>`)::

    >> spyking-circus path/mydata.extesion -H mpi.hosts


Structure of the hostfile
-------------------------

Such a hostfile will typically look like::

    192.168.0.1
    192.168.0.2
    192.168.0.3
    192.168.0.4
    192.168.0.5

If this is your parameter file, and if you launch the code with 20 CPUs::

    >> spyking-circus path/mydata.extension -c 20

Then the code will launch 4 instances of the program on the 5 nodes listed in the hostname.hosts file