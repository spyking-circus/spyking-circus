Configuration File
==================

This is the core of the algorithm, so this file has to be filled properly based on your data. Even if all key parameters of the algorithm ar listed in the file, only few are likely to be modified by a non-advanced user. The configuration file is divided in several sections. For all those sections, we will review the parameters, and tell you what are the most important ones

Data
----

The data section is::

    data_offset    = MCS                   # Length of the header ('MCS' is auto for MCS file)
    mapping        = mappings/mea_252.prb  # Mapping of the electrode (contact Pierre if changes)
    suffix         =                       # Suffix to add to generated files
    data_dtype     = uint16                # Type of the data
    dtype_offset   = auto                  # Padding for data
    sampling_rate  = 20000                 # Sampling rate of the data [Hz]
    N_t            = 5                     # Width of the templates [in ms]
    radius         = auto                  # Radius [in um] (if auto, read from the prb file)
    global_tmp     = True                  # should be False if local /tmp/ has enough space (better for clusters)
    multi-files    = False                 # If several files mydata_0,1,..,n.dat should be processed together (see documentation)

.. warning::

    This is the most important section, that will allow the code to properly load your data. If not properly filled, then results will be wrong

Parameters that are most likely to be changed:
    * ``data_offset`` If your file has no header, put 0. Otherwise, if it has been generated with MCRack, there is a header, and let the value to MCS, such that its length will be automatically detected
    * ``mapping`` This is the path to your probe mapping (see :doc:`How to design a probe file <../code/probe>`)
    * ``data_dtype`` The type of your data (``uint16``, ``int16``,...)
    * ``dtype_offset`` If you are using ``uint16`` data, then you must have an offset. If data are ``int16``, this should be 0
    * ``sampling_rate`` The sampling rate of your recording
    * ``N_t`` The temporal width of the templates. For *in vitro* data, 5ms seems a good value. For *in vivo* data, you should rather use 3 or even 2ms
    * ``radius`` The spatial width of the templates. By default, this value is read from the probe file. However, if you want to specify a larger or a smaller value [in um], you can do it here
    * ``global_temp`` If you are using a cluster with NFS, this should be False (local /tmp/ will be used by every nodes)
    * ``multi-files`` If several files ``mydata_0,1,..,n.dat`` should be processed together (see :doc:`Using multi files <../code/multifiles>`)


Detection
---------

The detection section is::

    spike_thresh   = 6                      # Threshold for spike detection
    peaks          = negative               # Can be negative (default), positive or both
    matched-filter = False                  # If True, we perform spike detection with matched filters
    matched_thresh = 5                      # Threshold for detection if matched filter is True
    alignment      = True                   # Realign the waveforms by oversampling

Parameters that are most likely to be changed:
    * ``spike_thresh`` The threshold for spike detection. 6-7 are good values
    * ``peaks`` By default, the code detects only negative peaks, but you can search for positive peaks, or both
    * ``matched-filter`` If activated, the code will detect smaller spikes by using matched filtering
    * ``matched_threhs`` During matched filtering, the detection threshold
    * ``alignment`` By default, during clustering, the waveforms are realigned by oversampling at 5 times the sampling rate and using bicubic spline interpolation
    
Filtering
---------

The filtering section is::

    cut_off        = 500, auto # Min and Max (auto=nyquist) cut off frequencies for the band pass butterworth filter [Hz]
    filter         = True      # If True, then a low-pass filtering is performed
    remove_median  = False     # If True, median over all channels is substracted to each channels (movement artifacts)

.. warning::

    The code performs the filtering of your data writing on the file itself. Therefore, you ``must`` have a copy of your raw data elsewhere. Note that as long as your keeping the parameter files, you can relaunch the code safely: the program will not filter multiple times the data, because of the flag ``filter_done`` at the end of the configuration file.

Parameters that are most likely to be changed:
    * ``cut_off`` The default value of 500Hz has been used in various recordings, but you can change it if needed. You can also specify the upper bound of the Butterworth filter
    * ``filter`` If your data are already filtered by a third program, turn that flag to False
    * ``remove_median`` If you have some movement artifacts in your *in vivo* recording, and want to substract the median activity over all anaylyzed channels from each channel individually

Triggers
--------

The triggers section is::

    trig_file      =           # If external stimuli need to be considered as putative artefacts (see documentation)
    trig_windows   =           # The time windows of those external stimuli [in ms]
    clean_artefact = False     # If True, external artefacts induced by triggers will be suppressed from data 
    make_plots     = png       # Generate sanity plots of the averaged artefacts [Nothing or None if no plots]

Parameters that are most likely to be changed:
    * ``trig_file`` The path to the file where your artefact times and labels. See :doc:`how to deal with stimulation artefacts <../code/artefacts>`
    * ``trig_windows`` The path to file where your artefact temporal windows. See :doc:`how to deal with stimulation artefacts <../code/artefacts>`
    * ``clean_artefact`` If you want to remove any stimulation artefacts, defined in the previous file. See :doc:`how to deal with stimulation artefacts <../code/artefacts>`
    * ``make_plots`` The default format to save the plots of the artefacts, one per artefact, showing all channels. You can set it to None if you do not want any


Whitening
---------

The whitening section is::

    chunk_size     = 60        # Size of the data chunks [in s]
    safety_time    = 1         # Temporal zone around which templates are isolated [in ms]
    temporal       = False     # Perform temporal whitening
    spatial        = True      # Perform spatial whitening
    max_elts       = 10000     # Max number of events per electrode (should be compatible with nb_elts)
    nb_elts        = 0.8       # Fraction of max_elts that should be obtained per electrode [0-1]
    output_dim     = 5         # Can be in percent of variance explain, or num of dimensions for PCA on waveforms

Parameters that are most likely to be changed:
    * ``output_dim`` If you want to save some memory usage, you can reduce the number of features kept to describe a waveform.
    * ``chunk_size`` If you have a very large number of electrode, and not enough memory, you can reduce it


Clustering
----------

The clustering section is::

    extraction     = median-raw # Can be either median-raw (default), median-pca, mean-pca, mean-raw, or quadratic
    safety_space   = True       # If True, we exclude spikes in the vicinity of a selected spikes
    safety_time    = 1          # Temporal zone around which templates are isolated [in ms]
    max_elts       = 10000      # Max number of events per electrode (should be compatible with nb_elts)
    nb_elts        = 0.8        # Fraction of max_elts that should be obtained per electrode [0-1]
    nclus_min      = 0.01       # Min number of elements in a cluster (given in percentage)
    max_clusters   = 10         # Maximal number of clusters for every electrodes
    nb_repeats     = 3          # Number of passes used for the clustering
    make_plots     = png        # Generate sanity plots of the clustering
    sim_same_elec  = 3          # Distance within clusters under which they are re-merged
    cc_merge       = 0.975      # If CC between two templates is higher, they are merged
    dispersion     = (5, 5)     # Min and Max dispersion allowed for amplitudes [in MAD]
    smart_search   = False      # Parameter to activate the smart search mode
    test_clusters  = False      # Should be False. Only to plot injection of synthetic clusters
    noise_thr      = 0.8        # Minimal amplitudes are such than amp*min(templates) < noise_thr*threshold
    remove_mixture = True       # At the end of the clustering, we remove mixtures of templates

.. note::

    This is the a key section, as bad clustering will implies bad results. However, the code is very robust to parameters changes.

Parameters that are most likely to be changed:
    * ``extraction`` The method to estimate the templates. ``Raw`` methods are slower, but more accurate, as data are read from the files. ``PCA`` methods are faster, but less accurate, and may lead to some distorded templates. ``Quadratic`` is slower, and should not be used.
    * ``max_elts`` The number of elements that every electrode will try to collect, in order to perform the clustering
    * ``nclus_min`` If you have too many clusters with few elements, you can increase this value. This is expressed in percentage of collected spike per electrode. So one electrode collecting *max_elts* spikes will keep clusters with more than *nclus_min.max_elts*. Otherwise, they are discarded
    * ``max_clusters`` This is the maximal number of cluster that you expect to see on a given electrode. For *in vitro* data, 10 seems to be a reasonable value. For *in vivo* data and dense probes, you should set it to 10-15. Increase it only if the code tells you so.
    * ``nb_repeats`` The number of passes performed by the algorithm to refine the density landscape
    * ``smart_search`` By default, the code will collect only a subset of spikes, randomly, on all electrodes. However, for long recordings, or if you have low thresholds, you may want to select them in a smarter manner, in order to avoid missing the large ones, under represented. If the smart search is activated, the code will first sample the distribution of amplitudes, on all channels, and then implement a rejection algorithm such that it will try to select spikes in order to make the distribution of amplitudes more uniform. This can be very efficient, and may become True by default in future releases.
    * ``cc_merge`` After local merging per electrode, this step will make sure that you do not have duplicates in your templates, that may have been spread on several electrodes. All templates with a correlation coefficient higher than that parameter are merged. Remember that the more you merge, the faster is the fit
    * ``dispersion`` The spread of the amplitudes allowed, for every templates, around the centroid.
    * ``remove_mixture`` By default, any template that can be explained as sum of two others is deleted. 
    * ``make_plots`` By default, the code generates sanity plots of the clustering, one per electrode.

Fitting
-------

The fitting section is::

    chunk          = 1         # Size of chunks used during fitting [in second]
    gpu_only       = True      # Use GPU for computation of b's AND fitting
    amp_limits     = (0.3, 30) # Amplitudes for the templates during spike detection
    amp_auto       = True      # True if amplitudes are adjusted automatically for every templates
    max_chunk      = inf       # Fit only up to max_chunk   

Parameters that are most likely to be changed:
    * ``chunk`` again, to reduce memory usage, you can reduce the size of the temporal chunks during fitting. Note that it has to be one order of magnitude higher than the template width ``N_t``
    * ``gpu_only`` By default, all operations will take place on the GPU. However, if not enough memory is available on the GPU, then you can turn this flag to False. 
    * ``max_chunk`` If you just want to fit the first *N* chunks, otherwise, the whole file is processed


Merging
-------

The merging section is::

    cc_overlap     = 0.5       # Only templates with CC higher than cc_overlap may be merged
    cc_bin         = 2         # Bin size for computing CC [in ms]
    correct_lag    = False     # If spikes are aligned when merging. May be better for phy usage

To know more about how those merges are performed and how to use this option, see :doc:`Automatic Merging <../code/merging>`. Parameters that are most likely to be changed:
    * ``correct_lag`` By default, in the meta-merging GUI, when two templates are merged, the spike times of the one removed are simply added to the one kept, without modification. However, it is more accurate to shift those spike, in times, by the temporal shift that may exist between those two templates. This will lead to a better visualization in phy, with more aligned spikes

Converting
----------

The converting section is::

    erase_all      = True      # If False, a prompt will ask you to export if export has already been done
    export_pcs     = prompt    # Can be prompt [default] or in none, all, some

Parameters that are most likely to be changed:
    * ``erase_all`` if you want to always erase former export, and skip the prompt
    * ``export_pcs`` if you already know that you want to have all, some, or no PC and skip the prompt

Extracting
----------

The extracting section is::

    safety_time    = 1         # Temporal zone around which spikes are isolated [in ms]
    max_elts       = 10000     # Max number of events per templates (should be compatible with nb_elts)
    nb_elts        = 0.8       # Fraction of max_elts that should be obtained per electrode [0-1]
    output_dim     = 5         # Percentage of variance explained while performing PCA
    cc_merge       = 0.975     # If CC between two templates is higher, they are merged
    noise_thr      = 0.8       # Minimal amplitudes are such than amp*min(templates) < noise_thr*threshold


This is an experimental section, not used by default in the algorithm, so nothing to be changed here

Validating
----------

The validating section is::

    nearest_elec   = auto      # Validation channel (e.g. electrode closest to the ground truth cell)
    max_iter       = 200       # Maximum number of iterations of the stochastic gradient descent (SGD)
    learning_rate  = 1.0e-3    # Initial learning rate which controls the step-size of the SGD
    roc_sampling   = 10        # Number of points to estimate the ROC curve of the BEER estimate
    make_plots     = png       # Generate sanity plots of the validation [Nothing or None if no plots]
    test_size      = 0.3       # Portion of the dataset to include in the test split
    radius_factor  = 0.5       # Radius factor to modulate physical radius during validation
    juxta_dtype    = uint16    # Type of the juxtacellular data
    juxta_thresh   = 6         # Threshold for juxtacellular detection
    juxta_valley   = False     # True if juxta-cellular spikes are negative peaks

Please get in touch with us if you want to use this section, only for validation purposes. This is an implementaion of the :doc:`BEER metric <../advanced/beer>`
