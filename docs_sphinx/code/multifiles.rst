Processing Several Files
========================

It is often the case that, during the same recording session, the experimentalist records only some temporals chunks and not the whole experiment. However, because the neurons are the same all over the recording, it is better to process them as a single datafile. One way of doing so is to manually concatenate all those individual files into a giant one, or use the ``multi-files`` option of the code.

.. note::

    If you just want to process several *independent* files, coming from different recording sessions, you need to use the batch mode (see :doc:`the documentation on the parameters <../code/parameters>`)

Activating Multi-files
----------------------

For the sake of clarity, we assume that those files are labeled

    - ``mydata_0.extension``
    - ``mydata_1.extension``
    - ...
    - ``mydata_N.extension``

Launch the code on the first file::

    >> spyking-circus mydata_0.extension

The code will create a parameter file, ``mydata_0.params``. Edit the file, and in the ``data`` section, set ``multi-files`` to ``True``. Relaunch the code on the first file only::

    >> spyking-circus mydata_0.extension

The code will now display something like::

    ##################################################################
    #####              Welcome to the SpyKING CIRCUS             #####
    #####                                                        #####
    #####              Written by P.Yger and O.Marre             #####
    ##################################################################

    Steps         : fitting
    GPU detected  : True
    Number of CPU : 12
    Number of GPU : 6
    Shared Memory : True
    Parallel HDF5 : False
    Hostfile      : /home/spiky/spyking-circus/circus.hosts

    ##################################################################

    -------------------------  Informations  -------------------------
    | Number of recorded channels : 256
    | Number of analyzed channels : 256
    | Data type                   : uint16
    | Sampling rate               : 20 kHz
    | Header offset for the data  : 1881
    | Duration of the recording   : 184 min 37 s
    | Width of the templates      : 5 ms
    | Spatial radius considered   : 250 um
    | Stationarity                : True
    | Waveform alignment          : True
    | Skip strong artefacts       : True
    | Template Extraction         : median-raw
    | Multi-files activated       : 19 files
    ------------------------------------------------------------------

The key line here is the one stating that the code has detected 19 files, and will process them as a single one.

.. note::

    The multi-files mode assumes that all files have the same properties: mapping, data type, data offset, ... It has to be the case if they are all coming from the same recording session

While running, in its first phase (filtering), instead of filtering all those individual files on site, the code will filter and concatenate them into a new file, ``mydata_all.extension``. Templates are then detected onto this single files, and fitting is also applied onto it.

Visualizing results from multi-files
------------------------------------

As said, results are obtained on a single file ``mydata_all.extension``, resulting of the concatenation of all the individual files. So when you are launching the GUI::

    >> circus-gui-matlab mydata_0.extension

what you are seeing are *all* the spikes on *all* files. Here you can delete/merge templates, see the devoted GUI section for that (:doc:`GUI <../GUI/index>`). Note that you need to process data in such a manner, because otherwise, if looking at all results individually, you would have a very hard time keeping track of the templates over several files. Plus, you would not get all the information contained in the whole recording (the same neuron could be silent during some temporal chunks, but spiking during others).

Getting results from multi-files
--------------------------------

Once your manual sorting session is done, you can simply split the results in order to get one result file per data file. To do so, simply launch::

    >> circus-multi mydata_0.extension

This will create several files
    - ``mydata_0.results.hdf5`` 
    - ``mydata_1.results.hdf5``
    - ...
    - ``mydata_N.results.hdf5``

In each of them, you'll find the spike times, between *0* and *T*, if *T* is the length of file *i*.