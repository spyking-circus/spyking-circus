Quickstart
============

.. _running_the_algorithm:

Running the algorithm
---------------------

Copy your files
~~~~~~~~~~~~~~~

First, you will need to create a directory (we call it **path** – usually you put both the date of the experiment and the name of the person doing the sorting). Your data file should have a name like **path/mydata.extension** Note that your data should not be filtered, and that this filtering will be done ONTO the data, so you need to keep a copy elsewhere of you raw data.

Generate a parameter file
~~~~~~~~~~~~~~~~~~~~~~~~~

Before running the algorithm, you will always need to provide parameters, as a parameter file. Note that this parameter file has to be in the same folder than your data, and should be named **path/mydata.params**. If you have already yours, great, just copy it in the folder. Otherwise, just launch the algorithm, and the algorithm will ask you if you want to create a template one, that you have to edit before launching the code::

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

Before running the algorithm, you will always need to provide parameters, at the beginning of a parameter file. Note that this parameter file has to be in the same folder than your data, and should be named **path/mydata.params**. If you launch the algorithm without this parameter file, the algorithm will ask you if you want to create a template one, that you have to edit before launching the code. In this file, you mostly have to change

Run the algorithm
~~~~~~~~~~~~~~~~~

Then you should run the algorithm by typing the following command(s)::

    >> python spyking-circus path/mydata.extension

It should take around the time of the recording to run – maybe a bit more. The typical output of the program will  be something like::


    ##############################################################
    #####          Welcome to the SpyKING CIRCUS             #####
    #####                                                    #####
    #####          Written by P.Yger and O.Marre             #####
    ##############################################################
    Steps        : filtering, whitening, clustering, fitting
    Number of CPU: 1
    Number of GPU: 1
    Hostfile     : astrocyte.hosts
    GPU detected : False
    Analyzing data to get whitening matrix and thresholds
    We have kept 398287 times without a spike to estimate the whitening matrix...
    Because of whitening, we need to recompute the thresholds...
    Searching spikes to construct the PCA basis...
    60% |################                         |ETA:  0:00:09

Note that you can of course change the number of CPU/GPU used, and also launch only a subset of the steps. See the help of the code to have more informations.

Using the GUI
-------------

Get the data
~~~~~~~~~~~~

Once the algorithm has run on the data path/mydata.extension, you should have the following files in the directory path, all stating with your mydata:

path/mydata/mydata.amplitudes.mat
path/mydata/mydata.cluster.mat
path/mydata/mydata.overlap.mat
path/mydata/mydata.templates.mat
path/mydata/mydata.spiketimes.mat

Matlab GUI
~~~~~~~~~~

To launch the MATLAB GUI provided with the software, you need of course to have a valid installtion of MATLAB, and you should be able to simply do::

    >> circus-gui path/data.extensions


Phy
~~~

This is not the default output of the SpyKING CIRCUS yet, but you can export your data into the kwik format, and being able to load them with phy. To do so, at the end of the algorithm, simply do::

    >> spyking-circus path/data.extensions -m converting

This will create in the **path** folder a file name **path/mydata.kwx**, and you can use phy to open it.

To know more about the GUI section, see :doc:`documentation on the GUI <../GUI>`