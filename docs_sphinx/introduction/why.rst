Why using it?
=============

.. include:: <isonum.txt>


SpyKING CIRCUS is a free, open-source, spike sorting software written entirely in python. In a nutshell, this is a fast and efficient way to perform spike sorting using a template-matching based algorithm.

Because you have too many channels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classical algorithms of spike sorting are not properly scaling up when the number of channels is increasing. Most, if not all of them would have a very hard time dealing with more than 100 channels. However, the new generation of electrodes, either *in vitro* (MEA with 4225 channels) or *in vivo* (IMEC probe with 128 channels) are providing more and more channels, such that there is a clear need for a software that would properly scale with the size of the electrodes. 

.. note::

    |rarr| The SpyKING CIRCUS, based on the MPI_ library, can be launched on several processors. Execution time scales linearly as function of the number of computing nodes, and the memory consumption scales only linearly as function of the number of channels. So far, the code can handle 4225 channels in parallel.

Because of overlapping spikes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With classical spike sorting algorithms, overlapping spikes are leading to outliers in your clusters, such that they are discarded. Therefore, each time two neurons have overlapping waveforms, their spikes are ignored. This can be problematic when you are addressing questions relying on fine temporal interactions within neurons. It is even more problematic with large and dense electrodes, with many recording sites close from each others, because those overlapping spikes start to be the rule instead of the exception. Therefore, you need to have a spike sorting algorithm that can disentangle those overlapping spikes.

.. note::

    |rarr| The SpyKING CIRCUS, using a template-matching based algorithm, reconstructs the signal as a linear sum of individual waveforms, such that it can resolve the fine cross-correlations between neurons.


Because you want to automatize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large number of channels, a lot of clusters (or equivalently templates, or cells) can be detected by spike sorting algorithms, and the time spent by a human to review those results should be reduced as much as possible. 

.. note::

    |rarr| The SpyKING CIRCUS, in its current form, aims at automatizing as much as possible the whole workflow of spike sorting, reducing the human interaction. Not that it can be zero, but the software aims toward a drastic reduction of the manual curation, and results shows that performances as good or even better than with classical spike sorting approaches can be obtained.

.. _MPI: https://www.mpich.org/



