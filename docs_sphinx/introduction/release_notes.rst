Release notes
=============

Spyking CIRCUS 0.4.2
--------------------

This is the 0.4 release of the SpyKING CIRCUS, a new approach to the
problem of spike sorting. The code is based on a smart clustering with
sub sampling, and a greedy template matching approach, such that it can
resolve the problem of overlapping spikes. The publication about the software is available at http://biorxiv.org/content/early/2016/08/04/067843


.. figure::  launcher.png
   :align:   center

   The software can be used with command line, or a dedicated GUI


.. warning::

    Because this is a beta version, the code may evolve. Even if results are or should be correct, we can expect some more optimizations in a near future, based on feedbacks obtained on multiple datasets. If you spot some problems with the results, please be in touch with pierre.yger@inserm.fr

Contributions
~~~~~~~~~~~~~
Code and documentation contributions (ordered by the number of commits):

* Pierre Yger
* Marcel Stimberg
* Baptiste Lebfevre
* Christophe Gardella
* Olivier Marre
* Cyrille Rossant

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