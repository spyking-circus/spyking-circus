Processing streams of data
==========================

It is often the case that, during the same recording session, the experimentalist records only some temporal chunks and not the whole experiment. However, because the neurons are the same all over the recording, it is better to process them as a single datafile. The code can handle such streams of data, either from multiple sources (several data files), or within the same source if supported by the file format (chunks in a single file). 


Chunks spread over several files
--------------------------------

You can use the ``multi-files`` stream mode in the ``[data]`` section.

.. note::

    If you just want to process several *independent* files, coming from different recording sessions, you need to use the batch mode (see :doc:`the documentation on the parameters <../code/parameters>`)

For the sake of clarity, we assume that all your files are labelled

    - ``mydata_0.extension``
    - ``mydata_1.extension``
    - ...
    - ``mydata_N.extension``

Launch the code on the first file::

    >> spyking-circus mydata_0.extension

The code will create a parameter file, ``mydata_0.params``. Edit the file, and in the ``[data]`` section, set ``stream_mode`` to ``multi-files``. Relaunch the code on the first file only::

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
    | Streams                     : multi-files (19 found)
    ------------------------------------------------------------------

The key line here is the one stating that the code has detected 19 files, and will process them as a single one.

.. note::

    The multi-files mode assumes that all files have the same properties: mapping, data type, data offset, ... It has to be the case if they are all coming from the same recording session

While running, in its first phase (filtering), two options are possible:
    * if your file format allows write access, and ``overwrite`` is set to ``True`` in the ``data`` section, then every individual data file will be overwritten and filtered on site
    * if your file format does not allow write access, or ``overwrite`` is ``False``, the code will filter and concatenate all files into a new file, saved as a ``float32`` binary file called ``mydata_all_sc.extension``. Templates are then detected onto this single files, and fitting is also applied onto it.


Chunks contained in the same datafile
-------------------------------------

For more complex data structures, several recordings sessions can be saved within the same datafile. Assuming the file format allows it (see :doc:`the documentation on the file formats <../code/fileformat>`), the code can still stream all those chunks of data in order to process them as a whole. To do so, use exactly the same procedure as below, except that the ``stream_mode`` may be different, for example ``single-file``.


Visualizing results from several streams
-----------------------------------------

Multi-files
~~~~~~~~~~~

As said, results are obtained on a single file ``mydata_all.extension``, resulting of the concatenation of all the individual files. So when you are launching the GUI::

    >> circus-gui-matlab mydata_0.extension

what you are seeing are *all* the spikes on *all* files. Here you can delete/merge templates, see the devoted GUI section for that (:doc:`GUI <../GUI/index>`). Note that you need to process data in such a manner, because otherwise, if looking at all results individually, you would have a very hard time keeping track of the templates over several files. Plus, you would not get all the information contained in the whole recording (the same neuron could be silent during some temporal chunks, but spiking during others).

Getting individual results from streams
---------------------------------------

Once your manual sorting session is done, you can simply split the results in order to get one result file per data file. To do so, simply launch::

    >> circus-multi mydata_0.extension

This will create several files
    - ``mydata_0.results.hdf5`` 
    - ``mydata_1.results.hdf5``
    - ...
    - ``mydata_N.results.hdf5``

In each of them, you'll find the spike times of the given streams, between *0* and *T*, if *T* is the length of file *i*.
