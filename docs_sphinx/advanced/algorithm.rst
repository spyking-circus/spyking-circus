Details of the algorithm
========================

The full details of the algorithm have not been published yet, so we will only draft here the key principles and describe the ideas behind the four key steps of the algorithm. If you can not wait and really would like to know more about all its parameters, please get in touch with pierre.yger@inserm.fr


Filtering
---------

In this first step, nothing incredibily fancy is happenning. All the channels are high-pass filtered in order to remove fluctutations, and to do so, we used a classical third order Butterworth filter. This step is requiered for the algorithm to work. 

Withening
---------

In this step, we are removing the spurious spatio-temporal correlations that may exist between all the channels. By detecting temporal periods in the data without any spikes, we compute a spatial matrix and a temporal filter that are whitening the data. This is a key step in most signal processing algorithms. Because of this transformation, all the templates and data that are seen after in the MATLAB_ GUI are in fact seen in this whitened space.

Clustering
----------

This is the main step of the algorithm, the one that allows it to perform a good clustering in a high dimensionnal space, with a smart subsampling. 

A divide and conquer approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we parallelize the problem by pooling spikes per electrodes, such that we can perform *N* independent clusterings (one per electrode), instead of a giant one. By doing so, the problem becomes intrinsically parallel, and one could easily use MPI to split the load over several nodes.

A smart and robust clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We expanded on recent clustering technique `Adam Kampff <http://www.kampff-lab.org/>`_ and designed a *almost* fully automated method for clustering the data without being biased by density peaks. In fact, the good point about the template matching that we are using is that we just need the *averaged* waveforms, so we don't need to perform a clustering on all the spikes. The key point is to select only a subset

Fitting
-------


.. _MATLAB: http://fr.mathworks.com/products/matlab/
