Release notes
=============

Spyking CIRCUS 0.8
------------------

This is the 0.8 release of the SpyKING CIRCUS, a new approach to the problem of spike sorting. The code is based on a smart clustering with
sub sampling, and a greedy template matching approach, such that it can resolve the problem of overlapping spikes. The publication about the software 
is available at https://elifesciences.org/articles/34518


.. figure::  launcher.png
   :align:   center

   The software can be used with command line, or a dedicated GUI


.. warning::

    The code may still evolve. Even if results are or should be correct, we can expect some more optimizations in a near future, based on feedbacks obtained on multiple datasets. If you spot some problems with the results, please be in touch with pierre.yger@inserm.fr

Contributions
~~~~~~~~~~~~~
Code and documentation contributions (ordered by the number of commits):

* Pierre Yger
* Marcel Stimberg
* Baptiste Lebfevre
* Christophe Gardella
* Olivier Marre
* Cyrille Rossant
* Joze Guzman
* Ariel Burman
* Ben Acland

=============
Release 1.1.0
=============

* make the behavior of the code deterministic, to ease comparison between numerous runs
* dropping py2.7 support, since it is now deprecated
* fix a rare bug with local_chunk not defined in clustering
* fix a bug with shanks leading to rare errors

=============
Release 1.0.9
=============

* add the option to flag saturation values (with the sat_value parameter)
* fix a bug: non analyzed channels were filtered (unexpected behavior)
* add the option to flag artefacts times (as values higher than weird_thresh MAD)
* fix a nasty bug leading to missed units when too many global merges


=============
Release 1.0.8
=============

* script circus-artefacts now saves everything in ms, to be consistent
* fix a bug in the neuralynx wrapper, causing last chunk to be ignored
* add the option, in circus-multi to export everything for phy
* optimization for clusters, to secure the .params file during overwritting
* fix a bug in the preview mode with the gain

=============
Release 1.0.7
=============

* can run meta merging even without a proper DISPLAY
* fix a potential bug for positive peaks or electrodes without spikes
* the .params file was not created automatically anymore
* fix a bug while computing the support if shanks
* fix bugs in the RHD file wrapper
* add clean_merging option to remove duplicated spikes in pairs during meta merging
* fix the info function in the launcher
* fix for common ground if only one shank
* adaptive global maximal amplitude for every templates
* optimization of the mapping-file mode
* bug in the batch mode
* fix a bug when filtering multi files and not overwriting

=============
Release 1.0.6
=============

* optimizations of the fitting procedure
* optimizations in the basis for PCA (smoothing waveforms)
* optimizations to remove more mixtures (using matching)
* optimizations to make the automated amplitudes more robust
* can read non sorted channels with HDF5 file format
* fix a regression with positive peaks introduced in 1.0.4
* fix a int overflow in windows only impacting the automatic amplitudes

=============
Release 1.0.5
=============

* automatic versioning with versionneer
* conda/pip packages built with git actions (thanks to Marcel)

=============
Release 1.0.4
=============

* possibility for the amplitudes [a_min, a_max] to depend on time
* median is now removed per shanks
* common ground syntax slightly changed, to allow one ground per shank
* fix a bug when collect_all and dense templates
* fix if no templates are found
* improvements of the smart search
* add the option to collect normalized MSE during fitting. False by default
* fix the rhd wrapper
* divide and conquer assigment now based on barycenters instead of simple extremas
* exit the clustering if too many centroids are found (sign of bad channels)
* fixes in the meta merging GUI (RPV and dip)
* optimizations for the second component, less double counting
* fix to use at least 1 CPU
* the code can be launched on a .params file with [data] file_name filled
* add the mapping-file mode for neuralynx [mapping_file] (should be generalized?)
* better estimation of amplitudes for axonal spikes
* enhance the estimation of amplitudes by proper alignement

=============
Release 1.0.2
=============

* possibility for the amplitudes [a_min, a_max] to depend on time
* median is now removed per shanks
* common ground syntax slightly changed, to allow one ground per shank
* fix a bug when collect_all and dense templates
* fix if no templates are found
* improvements of the smart search
* add the option to collect normalized MSE during fitting. False by default
* fix the rhd wrapper
* divide and conquer assigment now based on barycenters instead of simple extremas
* exit the clustering if too many centroids are found (sign of bad channels)
* fixes in the meta merging GUI (RPV and dip)
* optimizations for the second component, less double counting
* fix to use at least 1 CPU
* better estimation of amplitudes for axonal spikes
* enhance the estimation of amplitudes by proper alignement

=============
Release 1.0.0
=============

* prevent the use of negative indices for channels
* fix if no templates are found
* fix if the dead file is empty
* fix for recent versions of MPI (if dead times used)
* fix if dead times are not sorted or overlapping
* add the auto_cluster param in [data] to force gobal_tmp if needed
* fix when no cluster are found on some electrodes
* fix in the MATLAB Gui if no spikes are found
* support for the maxwell file format (MaxOne and MaxTwo)
* optimizations for faster fitting
* templates are densified dring fitting if not enough sparse (faster)

=============
Release 0.9.9
=============

* fix for shanks (because of optimization in 0.9.8)
* fix for clusters (if global tmp is not created)
* fix for recent versions of MPI (shared memory issues)
* still speeding up the fitting procedure, as a final bottleneck
* fix in the smart search and chunks exploration

=============
Release 0.9.8
=============

* fix a bug while filtering HDF5 file with overwrite set to False
* fix a bug for windows and Intel MPI
* speeding up the fitting procedure
* reducing the memory footprint while optimizing amplitudes for large number of templates
* changing the way of saving overlaps, making use of internal symmetry. Lot of memory saved

=============
Release 0.9.7
=============

* fix a bug in the preview mode
* fix a bug while converting with export_all set to True
* fix a rare bug when both peaks are detect in clustering with smart search
* fix a bug if removing reference channel after filtering has been already done
* fix a bug while converting with export_all
* fix a bug in the filtering introduced in 0.9.6 (last chunk not filtered)
* fix a possible bug in smart search with dynamic bins
* enhance the robustness of the whitening for very large arrays
* speeding up the fitting procedure
* enhancing the non-selection of noisy snippets, and thus clustering
* option to dynamically adapt cc_merge for large number of electrodes
* remove putative mixtures based on variance, speeding up drastically CC estimation

=============
Release 0.9.6
=============

* fixes in the smart search (not all rare cases were covered in 0.9.5)
* fix a bug if multi file is activated with very small chunks
* speeding up the estimation of the templates: less snippets, closer to centroids
* speeding up the estimation of the amplitudes: less noise snippets
* speeding up isolation step during the smart search
* number of bins is adapted during the smart search as function of noise levels
* add the possibility to hide the status bars (for SpikeInterface logs)

=============
Release 0.9.5
=============

* speeding up the optimization of the amplitudes with MPI
* speeding up the processing of numpy datafiles (SpikeInterface)
* speeding up the smart search step (pre-generation of random numbers)
* speeding up the clustering step
* fix a bug while filtering in the preview mode introduced in 0.9.2
* speeding up the fitting step

=============
Release 0.9.4
=============

* speeding up the optimization of the amplitudes with MPI
* speeding up the processing of numpy datafiles (SpikeInterface)
* speeding up the smart search step (pre-generation of random numbers)

=============
Release 0.9.2
=============

* speeding up the algorithm
* fixing a bug in the clustering while assigining labels
* better detection of noise snippets discarded during clustering
* cosmetic changes in the sanity plots (clusters)
* better handling of overlapping chunks while filtering, removing filtering artefacts
* templates are restricted within shanks
* optimization of the amplitudes once all templates have been found
* export of a purity value, for phy, to assess how good a cluster is (between 0 and 1)
* display the purity value in MATLAB
* fix a (silent) bug in the supports introduced in 0.9.0, preventing mixture removal
* nb_chances is automatically adapted during the fitting procedure
* drifts are now automatically handled by the meta merging procedure
* enhancement in the automatic merging of drifts

=============
Release 0.9.1
=============

* Minor bug fixes in spyking-circus-launcher
* fix a bug in the amplitude display. Values were shuffled when several CPU were used
* add the option to ignore second component [clustering]->two_components

=============
Release 0.9.0
=============

* can now fit spikes below detection threshold (with spike_thresh_min)
* templates are now estimated without any spatial restrictions
* display a warning if N_t is not optimally chosen

=============
Release 0.8.9
=============

* fix a small bug in the smart search, introduced while refactoring in 0.8.7

=============
Release 0.8.8
=============

* fix a regression introduced in 0.8.7 for non contiguous probe indices

=============
Release 0.8.7
=============

* new methods to detect the peaks, more robust when low thresholds are fixed
* more accurate fitting procedure, slightly slower
* minor bug fixes
* addition of a sparsity_limit parameter in the meta merging GUI, to remove noise more precisely
* new parameter file is properly copied
* enhancement of the smoothing/alignement procedure, more accurate estimation of noisy templates
* better estimation of the amplitudes boundaries used during fitting
* optimization while removing mixtures and important bug fixes
* fix a bug in the thresholding method
* minor updates to get more refined spikes during whitening and clustering
* tests with SpikeInterface, showing clear increase in performance
* some cleaning in the parameter file
* default value for cc_merge is now 0.95, since merging functions are more robust
* noisy templates are removed by default while meta merging with a lower threshold (0.75)
* speeding up whitening and clustering steps

=============
Release 0.8.6
=============

* Export from manual sorting with MATLAB to phy is now possible
* Modification to pass SpikeSorters test suite

=============
Release 0.8.5
=============

* fix a bug while removing noisy templates in meta merging
* refactoring of the meta merging GUI, addition of bhatta distances
* meta merging more robust for non stationary recordings
* enhance logging if parameters are missing and/or not defined
* can now display the electrode labels in preview GUI
* detects if a wrong install of MPI is present (linking with mpi4py)
* conda install overwrites the old parameter file
* raw dispay of the MUA in the result GUI (to be improved)
* display an error if not all nodes on a cluster can read the datafiles
* fix a bug for thresholding method using dead times

=============
Release 0.8.4
=============

* fix if no spikes are found on some electrodes
* fix as mean/median-pca methods were broken (albeit not used)
* fix to prevent a rare crash while loading too sparse overlaps
* fix a bug with the new dip method in python 2.7
* add the thresholding method to extract only MUA activity (requested by users)
* channel lists in probe files can be non sorted
* memory usage is dynamically adapted to reduce memory footprint
* hdf5 and npy file format can now work with 3D arrays (x, y, time) or (time, x, y)
* fix a bug if basis for pos and neg spikes have different sizes
* add some docstrings (thanks to Jose Guzman)
* sparse export for phy is now the default
* comments can now be added in the trigger/dead times files
* 4096 channels can now run on a single machine, with low memory consumption
* basic support for 3d probes, without any visualization
* more robust to saturating channels with nan_to_num
* cc_merge set to 1 automatically if templates on few channels are detected
* fix a bug if only one artefact type is given
* fix a bug if only 2 spikes are found on a single electrode
* former parameters sim_same_elec and dip_threshold renamed into merge_method and merge_param
* sanity plots for local merges can now be produced during clustering (debug_plots in [clustering])

=============
Release 0.8.3
=============

* automatic suppression, during meta merging, of noisy templates (for SpikeToolKit/Forest)
* during the phy export, we can automatically pre-assign labels to neurons
* fix a bug when converting to phy with dead channels
* fix a bug when converting to phy with file formats without data_offset
* speedup the estimation of the amplitude distribution
* minor fixes for clusters
* smoothing of the templates thanks to Savitzky-Golay filtering
* fix a bug when launching GUIs for file format without data offset
* can now work with scipy 1.3 and statsmodels 0.10
* isolation mode is improved, set as default and leading to better performance
* reducing overclustering with the Hartigan dip-test of unimodality
* can now set the number of dimensions for local PCA (10 by default)

=============
Release 0.8.2
=============

* add a docker file to build the software
* add support for shanks in phy 2.0
* add support for deconverting in the qt launcher
* do not create a Qt App if merging in auto mode
* waveforms are convolved with a Hanning window to boost PCA
* oversampling in now adapted as function of the sampling rate
* reduction of I/O while oversampling
* speed improvement with undersampling while cleaning the dictionary
* automation of the software for SpikeForest/SpikeToolkit benchmarks
* merging is now included in the default pipeline
* normalization of the metrics in the meta merging GUI

=============
Release 0.8.0
=============

* major improvement in the clustering. No more max_clusters parameters
* much faster clustering (thanks to Ruben Herzog)
* added the statsmodels library as a required dependency
* enhancement of the smart search mode
* enhancement of the bicubic spline interpolation
* fix a typo when using dead times and the collect mode
* fix a minor bug when small amount of spikes are found during smart search
* fix a bug in the wrapper for BRW files
* support for phy 2.0 and phylib
* remove the strongly time shifted templates
* additing of a wrapper for MDA file format
* amplitudes for unfitted spikes is now 1 when exporting to phy
* default install is now qt5, to work with phy 2.0

=============
Release 0.7.6
=============

* cosmetic changes in the GUI
* adding a deconverting method to switch back from phy to MATLAB
* support for the lags between templates in the MATLAB GUI
* warn user if data are corrupted because of interrupted filtering
* reduction of the size for saved clusters
* display the file name in the header
* fix a nasty bug allowing spikes at the border of chunks to be fitted even during dead periods

=============
Release 0.7.5
=============

* fix a bug for MPICH when large dictionaries.
* fix a bug for numpy files when used with new numpy versions
* add the possibility to subtract one channel as a reference channel from others
* native support for blackrock files (only .ns5 tested so far)
* simplifications in the parameter file
* fix for display of progress bars with tqdm
* addition of a multi-folders mode for openephys
* hide GPU support for now, as this is not actively maintained and optimized
* fix in the MATLAB GUI for float32 data
* fix the broken log files
* default cpu number is now half the available cores

=============
Release 0.7.4
=============

* fix a regression with spline interpolation, more investigation needed

=============
Release 0.7.0
=============

* fix a possible rounding bug if triggers are given in ms
* artefacts are computed as medians and not means over the signal
* can turn off shared memory if needed
* a particular pattern can be specified for neuralynx files
* fix bugs with output_dir, as everything was not saved in the folder
* add a circus-folders script to process virtually files within several folders as a single recording
* add a circus-artefacts script to concatenate artefact files before using stream mode
* multi-files mode is now enabled for Neuralynx data
* fixes for conversion of old dataset with python GUI
* smooth exit if fitting with 0 templates (thanks to Alex Gonzalez)
* enhance the bicubic spline interpolation for oversampling
* spike times are now saved as uint32 for long recordings

=============
Release 0.6.7
=============

* optimizations for clusters (auto blosc and network bandwith)
* addition of a dead_channels option in the [detection] section, as requested
* prevent user to remove median with only 1 channel
* fix for parallel writes in HDF5 files
* hide h5py FutureWarning

=============
Release 0.6.6
=============

* fix for matplotlib 2.2.2
* fix a bug when loading merged data with phy GUI
* faster support for native MCD file with pyMCStream
* more robust whitening for large arrays with numerous overlaps
* add an experimental mode to refine coreset (isolated spikes)
* put merging units in Hz^2 in the merging GUI
* add a HDF5 compression mode to greatly reduce disk usage for very large probe
* add a Blosc compression mode to save bandwith for clusters
* fix a display bug in the merging GUI when performing multiple passes

=============
Release 0.6.5
=============

* reduce memory consumption for mixture removal with shared memory
* made an explicit parameter cc_mixtures for mixture removal in the [clustering] section
* Minor fixes in the MATLAB GUI
* fix in the exact times shown during preview if second is specified
* prevent errors if filter is False and overwrite is False

=============
Release 0.6.4
=============

* fix a bug in the BEER for windows platforms, enhancing robustness to mpi data types
* speed up the software when using ignore_dead_times
* ensure backward compatibility with hdf5 version for MATLAB
* fix a rare bug in clustering, when no spikes are found on electrodes
* fix a bug in the MATLAB GUI when reloading saved results, skipping overlap fixes

=============
Release 0.6.3
=============

* fix a bug if the parameter file have tabulations characters
* add a tab to edit parameters directly in the launcher GUI
* fix dtype offset for int32 and int64
* minor optimizations for computations of overlaps
* explicit message displayed on screen if filtering has already been performed
* can specify a distinct folder for output results with output_dir parameter
* fix a bug when launching phy GUI for datafiles without data_offset parameter (HDF5)
* fix a memory leak when using dead_times
* fix a bug for BRW and python3
* fix a bug in the BEER
* pin HDF5 to 1.8.18 versions, as MATLAB is not working well with 1.10
* fix a bug when relaunching code and overwrite is False
* fix a bug when peak detection is set on both with only one channel

=============
Release 0.6.2
=============

* fix for openephys and new python syntax
* fix in the handling of parameters 
* fix a bug on windows with unclosed hdf5 files
* fix a bug during converting with multi CPU on windows
* minor optimization in the fitting procedure
* support for qt5 (and backward compatibility with qt4 as long as phy is using Qt4)

=============
Release 0.6.1
=============

* fix for similarities and merged output from the GUIs
* fix for Python 3 and HDF5
* fix for Python 3 and launcher GUI
* fix for maxlag in the merging GUI
* optimization in the merging GUI for pairs suggestion
* addition of an auto_mode for meta merging, to suppress manual curation
* various fixes in the docs
* fix a bug when closing temporary files on windows
* allow spaces in names of probe files
* collect_all should take dead times into account
* patch to read INTAN 2.0 files
* fix in the MATLAB GUI when splitting neurons
* fix in the MATLAB GUI when selecting individual amplitudes

=============
Release 0.6.0
=============

* fix an IMPORTANT BUG in the similarities exported for phy/MATLAB, affect the suggestions in the GUI
* improvements in the neuralynx wrapper
* add the possibility to exclude some portions of the recordings from the analysis (see documentation)
* fix a small bug in MS-MPI (Windows only) when shared memory is activated and emtpy arrays are present

=============
Release 0.5.9
=============

* The validating step can now accept custom spikes as inputs
* Change the default frequency for filtering to 300Hz instead of 500Hz

=============
Release 0.5.8
=============

* fix a bug for int indices in some file wrappers (python 3.xx) (thanks to Ben Acland)
* fix a bug in the preview gui to write threshold
* fix a bug for some paths in Windows (thanks to Albert Miklos)
* add a wrapper for NeuraLynx (.ncs) file format
* fix a bug in the installation of the MATLAB GUI
* fix a bug to see results in MATLAB GUI with only 1 channel
* fix a bug to convert data to phy with only positive peaks
* add builds for python 3.6
* optimizations in file wrappers
* fix a bug for MCS headers in multifiles, if not all with same sizes
* add the possibility (with a flag) to turn off parallel HDF5 if needed
* fix a bug with latest version of HDF5, related to flush issues during clustering

=============
Release 0.5.7
=============

* Change the strsplit name in the MATLAB GUI
* Fix a bug in the numpy wrapper
* Fix a bug in the artefact removal (numpy 1.12), thanks to Chris Wilson
* Fixes in the matlab GUI to ease a refitting procedure, thanks to Chris Wilson
* Overlaps are recomputed if size of templates has changed (for refitting)
* Addition of a "second" argument for a better control of the preview mode
* Fix when using the phy GUI and the multi-file mode.
* Add a file wrapper for INTAN (RHD) file format

=============
Release 0.5.6
=============

* Fix in the smart_search when only few spikes are found
* Fix a bug in density estimation when only few spikes are found

=============
Release 0.5.5
=============

* Improvement in the smart_select option given various datasets
* Fix a regression for the clustering introduced in 0.5.2

=============
Release 0.5.2
=============

* fix for the MATLAB GUI
* smart_select can now be used [experimental]
* fix for non 0: DISPLAY
* cosmetic changes in the clustering plots
* ordering of the channels in the openephys wrapper
* fix for rates in the MATLAB GUI
* artefacts can now be given in ms or in timesteps with the trig_unit parameter

=============
Release 0.5rc
=============

* fix a bug when exporting for phy in dense mode
* compatibility with numpy 1.12
* fix a regression with artefact removal
* fix a display bug in the thresholds while previewing with a non unitary gain
* fix a bug when filtering in multi-files mode (overwrite False, various t_starts)
* fix a bug when filtering in multi-files mode (overwrite True)
* fix a bug if matlab gui (overwrite False)
* fix the gathering method, not working anymore
* smarter selection of the centroids, leading to more clusters with the smart_select option
* addition of a How to cite section, with listed publications

=============
Release 0.5b9
=============

* switch from progressbar2 to tqdm, for speed and practical issues
* optimization of the ressources by preventing numpy to use multithreading with BLAS
* fix MPI issues appearing sometimes during the fitting procedure
* fix a bug in the preview mode for OpenEphys files
* slightly more robust handling of openephys files, thanks to Ben Acland
* remove the dependency to mpi4py channel on osx, as it was crashing
* fix a bug in circus-multi when using extensions

=============
Release 0.5b8
=============

* fix a bug in the MATLAB GUI in the BestElec while saving
* more consistency with "both" peak detection mode. Twice more waveforms are always collect during whitening/clustering
* sparse export for phy is now available
* addition of a dir_path parameter to be compatible with new phy
* fix a bug in the meta merging GUI when only one template left

=============
Release 0.5b7
=============

* fix a bug while converting data to phy with a non unitary gain
* fix a bug in the merging gui with some version of numpy, forcing ucast
* fix a bug if no spikes are detected while constructing the basis
* Optimization if both positive and negative peaks are detected
* fix a bug with the preview mode, while displaying non float32 data

=============
Release 0.5b6
=============

* fix a bug while launching the MATLAB GUI

=============
Release 0.5b3
=============

* code is now hosted on GitHub
* various cosmetic changes in the terminal
* addition of a garbage collector mode, to collect also all unfitted spikes, per channel
* complete restructuration of the I/O such that the code can now handle multiple file formats
* internal refactoring to ease interaction with new file formats and readibility
* because of the file format, slight restructuration of the parameter files
* N_t and radius have been moved to the [detection] section, more consistent
* addition of an explicit file_format parameter in the [data] section
* every file format may have its own parameters, see documentation for details (or --info)
* can now work natively with open ephys data files (.openephys)
* can now work natively with MCD data files (.mcd) [using neuroshare]
* can now work natively with Kwik (KWD) data files (.kwd)
* can now work natively with NeuroDataWithoutBorders files (.nwb)
* can now work natively with NiX files (.nix)
* can now work natively with any HDF5-like structure data files (.h5)
* can now work natively with Arf data files (.arf)
* can now work natively with 3Brain data files (.brw)
* can now work natively with Numpy arrays (.npy)
* can now work natively with all file format supported by NeuroShare (plexon, blackrock, mcd, ...)
* can still work natively with raw binary files with/without headers :)
* faster IO for raw binary files
* refactoring of the exports during multi-file/preview/benchmark: everything is now handled in raw binary
* fix a bug with the size of the safety time parameter during whitening and clustering
* all the interactions with the parameters are now done in the circus/shared/parser.py file
* all the interactions with the probe are now done in the circus/shared/probes.py file
* all the messages are now handled in circus/shared/messages.py
* more robust and explicit logging system
* more robust checking of the parameters
* display the electrode number in the preview/result GUI
* setting up a continuous integration workflow to test all conda packages with appveyor and travis automatically
* cuda support is now turned off by default, for smoother install procedures (GPU yet do not bring much)
* file format can be streamed. Over several files (former multi-file mode), but also within the same file
* several cosmetic changes in the default parameter file
* clustering:smart_search and merging:correct_lag are now True by default
* fix a minor bug in the smart search, biasing the estimation of densities
* fix a bug with the masks and the smart-search: improving results
* addition of an overwrite parameter. Note that any t_start/t_stop infos are lost
* if using streams, or internal t_start, output times are on the same time axis than the datafile
* more robust parameter checking


=============
Release 0.4.3
=============

* cosmetic changes in the terminal
* suggest to reduce chunk sizes for high density probes (N_e > 500) to save memory
* fix a once-in-a-while bug in the smart-search


=============
Release 0.4.2
=============

* fix a bug in the test suite
* fix a bug in python GUI for non integer thresholds
* fix a bug with output strings in python3
* fix a bug to kill processes in windows from the launcher
* fix graphical issues in the launcher and python3
* colors are now present also in python3
* finer control of the amplitudes with the dispersion parameter
* finer control of the cut off frequencies during the filtering
* the smart search mode is now back, with a simple True/False flag. Use it for long or noisy recordings
* optimizations in the smart search mode, now implementing a rejection method based on amplitudes
* show the mean amplitude over time in the MATLAB GUI
* MATLAB is automatically closed when closing the MATLAB GUI
* mean rate is now displayed in the MATLAB GUI, for new datasets only
* spike times are now saved as uint32, for new datasets only
* various fixes in the docs
* improvements when peak detection is set on "both"
* message about cc_merge for low density probes
* message about smart search for long recordings
* various cosmetic changes
* add a conda app for anaconda navigator


=============
Release 0.4.1
=============

* fix a bug for converting millions of PCs to phy, getting rid of MPI limitation to int32
* fix bugs with install on Windows 10, forcing int64 while default is int32 even on 64bits platforms
* improved errors messages if wrong MCS headers are used
* Various cosmetic changes


===========
Release 0.4
===========

First realease of the software
