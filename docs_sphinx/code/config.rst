Configuration File
==================

This is the core of the algorithm, so this file has to be filled properly based on your data. Even if all key parameters of the algorithm ar listed in the file, only few are likely to be modified by a non-advanced user. The configuration file is divided in several sections. For all those sections, we will review the parameters, and tell you what are the most important ones

Data
----

The data section is::

    data_offset    = MCS                    # Length of the header ('MCS' is auto for MCS file)
    mapping        = mappings/mea_252.prb   # Mapping of the electrode (contact Pierre if changes)
    suffix         =                        # Suffix to add to generated files
    data_dtype     = uint16                 # Type of the data
    dtype_offset   = 32767                  # Padding for data (uint16 is 32767)
    gain           = 0.01                   # Gain for spike detection
    spike_thresh   = 7                      # Threshold for spike detection
    sampling_rate  = 20000                  # Sampling rate of the data [Hz]
    N_t            = 5                      # Width of the templates [in ms]
    radius         = auto                   # Radius [in um] (if auto, read from the prb file)
    global_tmp     = True                   # should be False if local /tmp/ has enough space (better for clusters)

.. warning::

    This is the most important section, that will allow the code to properly load your data. If not properly filled, then results will be wrong

Parameters that are most likely to be changed:
    * ``data_offset`` If your file has no header, put 0. Otherwise, if it has been generated with MCRack, there is a header, and let the value to MCS, such that its length will be automatically detected
    * ``mapping`` This is the path to your probe mapping (see :doc:`How to design a probe file <../code/probe>`)
    * ``data_dtype`` The type of your data (``uint16``, ``int16``,...)
    * ``dtype_offset`` If you are using ``uint16`` data, then you must have an offset. If data are ``int16``, this should be 0
    * ``spike_thresh`` 
    * ``sampling_rate`` The sampling rate of your recording
    * ``N_t`` The temporal width of the templates. For *in vitro* data, 5ms seems a good value. For *in vivo* data, you should rather use 3 or even 2ms
    * ``radius`` The spatial width of the templates. By default, this value is read from the probe file. However, if you want to specify a larger or a smaller value [in um], you can do it here
    * ``global_temp``

Filtering
---------

The filtering section is::

    cut_off        = 500       # Cut off frequency for the butterworth filter [Hz]
    filter         = True      # If True, then a low-pass filtering is performed

.. warning::

    The code performs the filtering of your data writing on the file itself. Therefore, you ``must`` have a copy of your raw data elsewhere. Note that as long as your keeping the parameter files, you can relaunch the code safely: the program will not filter multiple times the data, because of the flag ``filter_done`` at the end of the configuration file

Parameters that are most likely to be changed:
    * ``cut_off`` The default value of 500Hz has been used in various recordings, but you can change it if needed
    * ``filter`` If your data are already filtered by a third program, turn that flag to False

Whitening
---------

The whitening section is::

    safety_time    = 1         # Temporal zone around which templates are isolated [in ms]
    temporal       = True      # Perform temporal whitening
    spatial        = True      # Perform spatial whitening
    max_elts       = 10000     # Max number of events per electrode (should be compatible with nb_elts)
    nb_elts        = 0.8       # Fraction of max_elts that should be obtained per electrode [0-1]
    output_dim     = 5         # Can be in percent of variance explain, or num of dimensions for PCA on waveforms

Parameters that are most likely to be changed:
    * ``output_dim`` If you want to save some memory usage, you can reduce the number of features kept to describe a waveform.


Clustering
----------

The clustering section is::

    safety_space   = True      # If True, we exclude spikes in the vicinity of a selected spikes
    safety_time    = 1         # Temporal zone around which templates are isolated [in ms]
    max_elts       = 10000     # Max number of events per electrode (should be compatible with nb_elts)
    nb_elts        = 0.8       # Fraction of max_elts that should be obtained per electrode [0-1]
    nclus_min      = 0.002     # Min number of elements in a cluster (given in percentage)
    max_clusters   = 10        # Maximal number of clusters for every electrodes
    nb_repeats     = 3         # Number of passes used for the clustering
    make_plots     = True      # Generate sanity plots of the clustering
    sim_same_elec  = 3         # Distance within clusters under which they are re-merged
    smart_search   = 3         # Parameter for the smart search. The higher, the more strict
    test_clusters  = False     # Should be False. Only to plot injection of synthetic clusters
    noise_thr      = 0.8       # Minimal amplitudes are such than amp*min(templates) < noise_thr*threshold

.. warning::

    This is the a key section, as bad clustering will implies bad results. However, the code is very robust to parameters changes.

Parameters that are most likely to be changed:
    * ``max_elts`` The number of elements that every electrode will try to collect, in order to perform the clustering
    * ``nclus_min`` If you have too many clusters with few elements, you can increase this value. This is expressed in percentage of collected spike per electrode. So one electrode collecting *max_elts* spikes will keep clusters with more than *nclus_min.max_elts*. Otherwise, they are discarded
    * ``max_clusters`` This is the maximal number of cluster that you expect to see on a given electrode. For *in vitro* data, 10 seems to be a reasonable value. For *in vivo* data and dense probes, you should set it to 20-25
    * ``nb_repeats`` The number of passes performed by the algorithm to refine the density landscape
    * ``smart_search`` Control how different you want the collected spikes to be, on a given electrode. The more you increase this value, the more you'll discard spikes. If increased too much, you may not collect enough spikes, so 3 seems to be a good criteria. In fact, the code will suggest you to decrease this value if you are throwing away too many spikes
    * ``sim_same_elec`` Control how similar clusters have to be in order to be merged, before fitting, in order to reduce over clustering. The more you increase this value, the more you'll merge pairs of clusters. Again, 3 seems to be a good value. 
    * ``make_plots`` By default, the code generates sanity plots of the clustering, one per electrode.

Fitting
-------

The fitting section is::

    chunk          = 0.5       # Size of chunks used during fitting [in second]
    gpu_only       = True      # Use GPU for computation of b's AND fitting
    amp_limits     = (0.3, 30) # Amplitudes for the templates during spike detection
    amp_auto       = True      # True if amplitudes are adjusted automatically for every templates
    refractory     = 0         # Refractory period, in ms [0 is None]
    max_chunk      = inf       # Fit only up to max_chunk   
    spike_range    = 0         # Jitter allowed around each spike time to fit the templates (in ms) 

Parameters that are most likely to be changed:
    * ``chunk`` again, to reduce memory usage, you can reduce the size of the temporal chunks during fitting. Note that it has to be one order of magnitude higher than the template width ``N_t``
    * ``gpu_only`` By default, all operations will take place on the GPU. However, if not enough memory is available on the GPU, then you can turn this flag to False. 
    * ``max_chunk`` If you just want to fit the first *N* chunks, otherwise, the whole file is processed
    * ``spike_range`` [Experimental] May enhance the quality of the fit, but slows down the algorithm.

Extracting
----------

The extracting section is::

    safety_time    = 1         # Temporal zone around which spikes are isolated [in ms]
    max_elts       = 10000     # Max number of events per templates (should be compatible with nb_elts)
    nb_elts        = 0.8       # Fraction of max_elts that should be obtained per electrode [0-1]
    output_dim     = 5         # Percentage of variance explained while performing PCA

This is an experimental section, not used by default in the algorithm, so nothing to be changed here
