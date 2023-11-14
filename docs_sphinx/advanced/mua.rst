Multi Unit Activity
===================

In some cases, performing the spike sorting may be an overkill. For example to quickly check if (and where) you have activity in your recordings, and/or if you are just interested in the macroscopic activity of your tissue. Albeit important, we need to keep in mind that for some scientific questions, spike sorting may not be necessary. However, when data are large, it can still be complicated to simply get the times at which you have putative spikes (i.e. threshold crossings).
This is why, with SpyKING CIRCUS, you can quickly get what we call Multi-Unit Activity (MUA), i.e. times at which you have threshold crossings on every channels. Note however that the denser the probe, the more you will overestimate the real MUA, because of spikes being counted multiple times. 

You can use the ``thresholding`` method of the software, and to launch it, a typical workflow (assuming you want to filter and whiten the data first) will be::

    >> spyking-circus path/mydata.extension -m filtering,whitening,thresholding -c N

.. note::
    This thresholding step will produce a file ``mydata/mydata.mua.hdf5`` in which you will have one entry per electrode, with all the times at which a threshold crossing has been detected on the channels. You can also retrieve the values of the signal at the the corresponding times, such that you can visualize the histogram of the amplitudes. This can be used to quickly observe if and where do you have activity.
