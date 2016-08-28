SpyKING CIRCUS
==============

*A fast and scalable solution for spike sorting of large-scale extracellular recordings*

SpyKING CIRCUS is a python code to allow fast spike sorting on multi channel recordings. 
A preprint with the details of the algorithm can be found on BioRxiv at http://biorxiv.org/content/early/2016/08/04/067843. 
It has been tested on datasets coming from *in vitro* retina 
with 252 electrodes MEA, from *in vivo* hippocampus with tetrodes, *in vivo* and *in vitro* cortex 
data with 30 and up to 4225 channels, with good results. Synthetic tests on these data show 
that cells firing at more than 0.5Hz can be detected, and their spikes recovered with error 
rates at around 1%, even resolving overlapping spikes and synchronous firing. It seems to 
be compatible with optogenetic stimulation, based on experimental data obtained in the retina.

SpyKING CIRCUS is currently still under development. Please do not hesitate to report issues with the issue tracker

* Documentation can be found at http://spyking-circus.rtfd.org
* A Google group can be found at http://groups.google.com/forum/#!forum/spyking-circus-users

**The SpyKING CIRCUS is released under the terms of the CeCILL 2.1 license.**
