.. SpyKING CIRCUS documentation master file, created by
   sphinx-quickstart on Mon Oct  5 13:34:57 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the SpyKING CIRCUS's documentation!
==============================================

.. figure::  circus.png
   :align:   center

The SpyKING CIRCUS is a massively parallel code to perform semi automatic spike sorting on large extra-cellular recordings. Using a smart clustering and a greedy template matching approach, the code can solve the problem of overlapping spikes, and has been tested both for *in vitro* and *in vivo* data, from tens of channels to up to 4225 channels. Results are very good, cross-validated on several datasets, and details of the algorithm can be found in the following publication: http://biorxiv.org/content/early/2016/08/04/067843

.. note::

   In 0.5, smart search is now activated by default, and important changes have been done under the hood to refactor the code in order to read/write virtually any file format, as long as a wrapper is provided. 

   * Be in touch if you are interested by writing your own file wrapper to read/write a custom file format. 
   * A "garbage collector" mode has also been added, to help the user to get a **qualitative** feedback on the performance of the algorithm, estimating the number of missed spikes: if ``collect_all`` in the ``[fitting]`` section is activated, spikes left unfitted during the fitting procedure can be collected, grouped by electrode (keep in mind that this is more for a debugging purpose, and those spikes are only an approximation).
   * A ``smart_select`` option is now available in the ``[clustering]`` section, that should considerably enhance the clustering for datasets with very low or very high firing rates. If feedback and some more quantitative benchmarks are positive, it may become the new default mode for clustering. You are welcome to give it a try

.. toctree::
   :maxdepth: 2
   :titlesonly:

   introduction/index
   code/index
   GUI/index
   advanced/index
   issues/index
   bib/index

.. figure::  GUI/standalone.png
   :align:   center

