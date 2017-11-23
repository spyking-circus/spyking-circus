.. SpyKING CIRCUS documentation master file, created by
   sphinx-quickstart on Mon Oct  5 13:34:57 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the SpyKING CIRCUS's documentation!
==============================================

.. figure::  circus.png
   :align:   center

The SpyKING CIRCUS is a massively parallel code to perform semi automatic spike sorting on large extra-cellular recordings. Using a smart clustering and a greedy template matching approach, the code can solve the problem of overlapping spikes, and has been tested both for *in vitro* and *in vivo* data, from tens of channels to up to 4225 channels. Results are very good, cross-validated on several datasets, and details of the algorithm can be found in the following publication: http://biorxiv.org/content/early/2016/08/04/067843

.. warning::
   
   We strongly recommend to upgrade to 0.6.xx, as a rare but annoying bug has been found in 0.5.xx while exporting value for post-processing GUIs (MATLAB, phy). The bug is not systematic, depending on the numbers of templates/cores, but it is worth upgrading. A patch will be automatically applied to already sorted results while launching the GUIs, but for phy users, you will need to relaunch the ``converting`` step


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

