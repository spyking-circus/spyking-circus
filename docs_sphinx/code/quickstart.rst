Quickstart
==========

Running the algorithm
---------------------

Copy your files
~~~~~~~~~~~~~~~

First, you will need to create a directory (we call it ``path`` – usually you put both the date of the experiment and the name of the person doing the sorting). Your data file should have a name like ``path/mydata.extension`` 

.. warning::

    Your data should not be filtered, and the filtering will be done only once **onto** the data. So you need to keep a copy elsewhere of you raw data.

Generate a parameter file
~~~~~~~~~~~~~~~~~~~~~~~~~

Before running the algorithm, you will always need to provide parameters, as a parameter file. Note that this parameter file has to be in the same folder than your data, and should be named ``path/mydata.params``. If you have already yours, great, just copy it in the folder. Otherwise, just launch the algorithm, and the algorithm will ask you if you want to create a template one, that you have to edit before launching the code::

    >> spyking-circus.py path/mydata.extension
    ##############################################################
    #####          Welcome to the SpyKING CIRCUS             #####
    #####                                                    #####
    #####          Written by P.Yger and O.Marre             #####
    ##############################################################
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
    #####              Welcome to the SpyKING CIRCUS             #####
    #####                                                        #####
    #####              Written by P.Yger and O.Marre             #####
    ##################################################################

    Steps         : filtering, whitening, clustering, fitting
    GPU detected  : False
    Number of CPU : 1
    Parallel HDF5 : True
    Shared memory : True
    Hostfile      : /home/pierre/spyking-circus/circus.hosts

    ##################################################################

    -------------------------  Informations  -------------------------
    | Number of recorded channels : 252
    | Number of analyzed channels : 252
    | Data type                   : uint16
    | Sampling rate               : 10 kHz
    | Header offset for the data  : 1794
    | Duration of the recording   : 7 min 12 s
    | Width of the templates      : 5 ms
    | Spatial radius considered   : 250 um
    | Stationarity                : True
    | Waveform alignment          : True
    | Skip strong artefacts       : False
    | Template Extraction         : median-raw
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

Matlab GUI
~~~~~~~~~~

To launch the MATLAB_ GUI provided with the software, you need of course to have a valid installation of MATLAB_, and you should be able to simply do::

    >> circus-gui-matlab path/mydata.extension

Python GUI
~~~~~~~~~~

An experimental GUI derived from phy_ and made especially for template-matching based algorithms can be launched by doing::


    >> spyking-circus path/mydata.extension -m converting
    >> circus-gui-python path/mydata.extension

To enable it, you must have a valid installation of phy_ and phycontrib_


To know more about the GUI section, see :doc:`documentation on the GUI <../GUI/index>`

.. _phy: https://github.com/kwikteam/phy
.. _phycontrib: https://github.com/kwikteam/phy-contrib
.. _MATLAB: http://fr.mathworks.com/products/matlab/