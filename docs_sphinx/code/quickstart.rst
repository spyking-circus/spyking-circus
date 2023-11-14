Quickstart
==========

Running the algorithm
---------------------

Copy your files
~~~~~~~~~~~~~~~

First, you will need to create a directory (we call it ``path`` – usually you put both the date of the experiment and the name of the person doing the sorting). Your data file should have a name like ``path/mydata.extension`` 

.. warning::

    Your data should not be filtered, and by default the filtering will be done only once **onto** the data. So you need to keep a copy elsewhere of you raw data. If you really do not want to filter data on site, you can use the ``overwrite`` parameter (see :doc:`documentation on the code <../code/config>` for more information).

Generate a parameter file
~~~~~~~~~~~~~~~~~~~~~~~~~

Before running the algorithm, you will always need to provide parameters, as a parameter file. Note that this parameter file has to be in the same folder than your data, and should be named ``path/mydata.params``. If you have already yours, great, just copy it in the folder. Otherwise, just launch the algorithm, and the algorithm will ask you if you want to create a template one, that you have to edit before launching the code::

    >> spyking-circus.py path/mydata.extension
    ##################################################################
    #####           Welcome to the SpyKING CIRCUS (0.7.6)        #####
    #####                                                        #####
    #####              Written by P.Yger and O.Marre             #####
    ##################################################################
    The parameter file is not present!
    You must have a file named path/mydata.params, properly configured, 
    in the same folder, with the data file.
    Do you want SpyKING CIRCUS to create a template there? [y/n]

In the parameter file, you mostly have to change only informations in the ``data`` section (see :doc:`documentation on the code <../code/config>` for more information).

Run the algorithm
~~~~~~~~~~~~~~~~~

Then you should run the algorithm by typing the following command(s)::

    >> spyking-circus path/mydata.extension

It should take around the time of the recording to run – maybe a bit more. The typical output of the program will be something like::


    ##################################################################
    #####           Welcome to the SpyKING CIRCUS (0.7.6)        #####
    #####                                                        #####
    #####              Written by P.Yger and O.Marre             #####
    ##################################################################

    File          : /home/test.dat
    Steps         : filtering, whitening, clustering, fitting
    Number of CPU : 1
    Parallel HDF5 : True
    Shared memory : True
    Hostfile      : /home/pierre/spyking-circus/circus.hosts

    ##################################################################

    -------------------------  Informations  -------------------------
    | Number of recorded channels : 252
    | Number of analyzed channels : 252
    | File format                 : RAW_BINARY
    | Data type                   : int16
    | Sampling rate               : 20 kHz
    | Duration of the recording   : 4 min 0 s 0 ms
    | Width of the templates      : 3 ms
    | Spatial radius considered   : 200 um
    | Threshold crossing          : negative
    ------------------------------------------------------------------
    -------------------------  Informations  -------------------------
    | Filtering has already been done with cut off at 500Hz
    ------------------------------------------------------------------
    Analyzing data to get whitening matrices and thresholds...
    We found 20s without spikes for whitening matrices...
    Because of whitening, we need to recompute the thresholds...
    Searching spikes to construct the PCA basis...
    100%|####################################################

Note that you can of course change the number of CPU/GPU used, and also launch only a subset of the steps. See the help of the code to have more informations.

Using Several CPUs
------------------

To use several CPUs, you should have a proper installation of MPI, and a valid hostfile given to the program. See :doc:`documentation on MPI <../introduction/mpi>`. And then, you simply need to do, if *N* is the number of processors::

    >> spyking-circus path/mydata.extension -c N


Using the GUI
-------------

Get the data
~~~~~~~~~~~~

Once the algorithm has run on the data path/mydata.extension, you should have the following files in the directory path:

* ``path/mydata/mydata.result.hdf5``
* ``path/mydata/mydata.cluster.hdf5``
* ``path/mydata/mydata.overlap.hdf5``
* ``path/mydata/mydata.templates.hdf5``
* ``path/mydata/mydata.basis.hdf5``

See the details here see :doc:`file formats <../advanced/files>` to know more how those files are structured.

Since 0.8.2, you should also have the same files, but with the ``-merged`` extension for some of them. This is because the merging step has been included in the default pipeline of the algorithm. Both results (with or without this extra merging) can be visualized, and/or exported for MATLAB_ and phy_.

Matlab GUI
~~~~~~~~~~

To launch the MATLAB_ GUI provided with the software, you need of course to have a valid installation of MATLAB_, and you should be able to simply do::

    >> circus-gui-matlab path/mydata.extension

Python GUI
~~~~~~~~~~

An experimental GUI derived from phy_ and made especially for template-matching based algorithms can be launched by doing::


    >> spyking-circus path/mydata.extension -m converting
    >> circus-gui-python path/mydata.extension

To enable it, you must have a valid installation of phy_ and phylib_


To know more about the GUI section, see :doc:`documentation on the GUI <../GUI/index>`

.. _phy: https://github.com/cortex-lab/phy
.. _phylib: https://github.com/cortex-lab/phylib
.. _MATLAB: http://fr.mathworks.com/products/matlab/