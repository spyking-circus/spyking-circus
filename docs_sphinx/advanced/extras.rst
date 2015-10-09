Extra steps
===========

The code comes with some additional methods that are not executed by default, but that could still be useful. You can view them by simply doing::

    spyking-circus -h


Gathering
---------

The more important one is the ``gathering`` option. This option allows you, while the fitting procedure is still running, to collect the data that have already been generated and save them as a temporary result. This methods use the fact that temporal chunks are processed sequentially, so you can, at any time, review what has already been fitted. To do so, simply do::

    spyking-circus path/mydata.extension -m gathering -c N

.. warning::

    *N* must be equal to the number of nodes that are currently fitting the data, because you will collect the results from all of them

Note that the data will be saved as if they were the final results, so you can launch the GUI and review the results. If nodes have different speed, you may see gaps in the fitted chunks, because some may be slower than others. The point of this ``gathering`` function is not to provide you an *exhaustive* view of the data, but simply be sure that everything is working fine.

Converting
----------

As already said in the GUI section, this function allows you to export your results into the phy_ format. To do so, simply do::

    spyking-circus path/mydata.extension -m converting

.. warning::

    Note that currently, this format would **only** be recommended for rather small datasets. The more you have spikes, the larger the ``.kwik`` file will be. phy_ is computing the features for all the spikes in your recording, and saving them to disk, so if you have millions of spikes, this may take a while and use a large space on your hardrive. We are working with the phy_ team to enhance the integration into phy_, and on a short-term goal, we are planning to get rid of the MATLAB_ GUI.


Extracting
----------

This option allows the user to get, given a list of spike times and cluster ids, its own templates. For example one could perform the clustering with its own method, and given the results of its algorithms, extract templates and simply launch the template matching part in order to resolve overlapping spikes. To perform such a workflow, you just need to do::

    spyking-circus path/mydata.extension -m extracting,fitting 

.. warning::
    This option has not yet been tested during the integration in this alpha release, but please contact us if you are interested. 


Benchmarking
------------

This option allows the user to generate synthetic ground-truth, and assess the performance of the algorithm. We are planning to move it into a proper testsuite, and make its usage more user friendly. Currently, this is a bit undocumented and for internal use only. In a nutshell, three types of benchmarks can be performed from an already processed file:
    * ``fitting`` The code will select a given template, and inject multiple shuffled copies of it at various rates, at random places 
    * ``clustering`` The code will select a given template, and inject multiple shuffled copies of it at various rates and various amplitudes, at random places
    * ``synchrony`` The code will select a given template, and inject multiple shuffled copies of it on the same electrode, with a controlled pairwise correlation coefficient between those cells

.. warning::
    This option has not yet been well packaged during the integration in this alpha release, but please contact us if you are interested. 



.. _phy: https://github.com/kwikteam/phy
.. _MATLAB: http://fr.mathworks.com/products/matlab/

