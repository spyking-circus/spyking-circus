Dealing with stimulation artefacts
==================================

Sometimes, because of external stimulation, you may end up having some artefacts on top of your recordings. For example, in case of optogenetic stimulation, shinning light next to your recording electrode is likely to contaminate the recording. Or it could be that those artefacts are simply affecting some portions of your recordings that you would like easily to ignore. The code has several built-in mechanisms to deal with those artefacts, in the ``triggers`` section of the parameter file. 


Ignore some portions of the recording
-------------------------------------

You can decide to ignore some portions of the recordings, because they are corrputed by artefacts.

Setting dead periods
~~~~~~~~~~~~~~~~~~~~

In a text file, you must specify all the portions [t_start, t_stop] that you want to exclude from analysis. The times can be given in ms or in timesteps, and this can be changed with the ``dead_unit`` parameter. By default, they are assumed to be in ms. Assuming we want to exclude the first 500ms of every second, such a text file will look like::
	
	0 500 
	1000 1500
	2000 2500
	...
	10000 10500

All t_start/t_stop times here in the text file are in ms, and you must use one line per portion to exclude. Use ``dead_unit`` if you want to give times in timesteps.

How to use it
~~~~~~~~~~~~~

Once this file have been created, you should provide them in the ``[triggers]`` section of the code (see :doc:`here <../code/config>`) with the ``dead_file`` parameter. You should then activate the option ``ignore_times`` by setting it to ``True``. Once the code is launched, all steps (whitening/clustering/fitting) will only work on spikes that are not in the  time periods defined by the ``dead_file``.


Substract regularly occuring artefacts
--------------------------------------

In a nutshell, the code is able, from a list of stimulation times, to simply compute an median-based average artefact, and substract it automatically to the signal during the filtering procedure.


Setting stimulation times
~~~~~~~~~~~~~~~~~~~~~~~~~

In a first text file, you must specify all the times of your artefacts, identified by a given identifier. The times can be given in ms or in timesteps, and this can be changed with the ``trig_unit`` parameter. By default, they are assumed to be in ms. For example, imagine you have 2 different stimulation protocols, each one inducing a different artefact. The text file will look like::
	
	0 500.2 
	1 1000.2
	0 1500.3
	1 2000.1
	...
	0 27364.1
	1 80402.4

This means that stim 0 is displayed at 500.2ms, then stim 1 at 1000.2ms, and so on. All times in the text file are in ms, and you must use one line per time. Use ``trig_unit`` if you want to give times in timesteps.

Setting time windows
~~~~~~~~~~~~~~~~~~~~

In a second text file, you must tell the algorithm what is the time window you want to consider for a given artefact. Using the same example, and assuming that stim 0 produces an artefact of 100ms, while stim 1 produces a longer artefact of 510ms, the file should look like::

	0 100
	1 510

Here, again, use ``trig_unit`` if you want to provide times in timesteps.

How to use it
~~~~~~~~~~~~~

Once those two files have been created, you should provide them in the ``[triggers]`` section of the code (see :doc:`here <../code/config>`) with the ``trig_file`` and ``trig_windows`` parameters. You should then activate the option ``clean_artefact`` by setting it to ``True`` before launching the filtering step. Note that by default, the code will produce one plot by artefact, showing its temporal time course on all channels, during the imposed time window. This is what is substracted, at all the given times for this unique stimulation artefact.

.. figure::  artefact_0.png
   :align:   center

   Example of a stimulation artefact on a 252 MEA, substracted during the filtering part of the algorithm.


.. note::

	If, for some reasons, you want to relaunch this step (too small time windows, not enough artefacts, ...) you will need to copy again the raw data before relaunching the filtering. This is because remember that the raw data are *always* filtered on-site.