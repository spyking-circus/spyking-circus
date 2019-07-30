Extra steps
===========

The code comes with some additional methods that are not executed by default, but that could still be useful. You can view them by simply doing::

    >> spyking-circus -h


Merging
-------

This option will launh the Meta merging GUI, allowing a fast merging of obvious pairs, based on some automatic computations performed on the cross-correlograms. To launch it, simply use::

    >> spyking-circus path/mydata.extension -m merging -c N

.. note::
    This merging step will not affect your main results, and will generate additional files with the suffix ``merged``. You can launch it safely at the end of the fitting procedure, and try various parameters. To know more about how those merges are performed, (see :doc:`Automatic Merging <../code/merging>`). Note that after, if you want to visualize this ``merged`` result with the GUIs, you need do use the ``-e`` parameter, such as for example::

        >> circus-gui-matlab path/mydata.extension -e merged


Thresholding
------------

In some cases, you may not want to spike sort the data, but you could only be interested by all the times at which you have threshold crossings, i.e. putative events or Multi Unit Activity (MUA). Note that the denser the probe, the more you will overestimate the real MUA, because of spikes being counted multiple times. To launch it, simply use::

    >> spyking-circus path/mydata.extension -m thresholding -c N

.. note::
    This thresholding step will produce a file ``mydata/mydata.mua.hdf5`` in which you will have one entry per electrode, with all the times (and amplitudes) at which threshold crossing has been detected. :doc:`More on the MUA extraction <../advanced/mua>`

Gathering
---------

The more important one is the ``gathering`` option. This option allows you, while the fitting procedure is still running, to collect the data that have already been generated and save them as a temporary result. This methods use the fact that temporal chunks are processed sequentially, so you can, at any time, review what has already been fitted. To do so, simply do::

    >> spyking-circus path/mydata.extension -m gathering -c N

.. warning::

    *N* must be equal to the number of nodes that are currently fitting the data, because you will collect the results from all of them

Note that the data will be saved as if they were the final results, so you can launch the GUI and review the results. If nodes have different speed, you may see gaps in the fitted chunks, because some may be slower than others. The point of this ``gathering`` function is not to provide you an *exhaustive* view of the data, but simply be sure that everything is working fine.

Converting
----------

As already said in the GUI section, this function allows you to export your results into the phy_ format. To do so, simply do::

    >> spyking-circus path/mydata.extension -m converting -c N


During the process, you have the option to export or not the Principal Components for all the spikes that have been found, and phy_ will display them. Note that while this is safe to export all of them for small datasets, this will not scale for very large datasets with millions of spikes. 

.. warning::

    For millions of spikes, we do not recommend to export *all* Principal Components. You can export only *some*, but then keep in mind that you can not redefine manually your clusters in phy_


Deconverting
------------

This option will allow you to convert back your results from phy to the MATLAB GUI. This could be useful if you want to compare results between the GUI, or if you need to switch because of missing functionnalities. To convert the data, simply use::

    >> spyking-circus path/mydata.extension -m deconverting

.. note::
    If you worked with data and a particular extension, then you will need to specify the extension::

    >> spyking-circus path/mydata.extension -m deconverting -e extension


Extracting
----------

This option allows the user to get, given a list of spike times and cluster ids, its own templates. For example one could perform the clustering with its own method, and given the results of its algorithms, extract templates and simply launch the template matching part in order to resolve overlapping spikes. To perform such a workflow, you just need to do::

    >> spyking-circus path/mydata.extension -m extracting,fitting 

.. warning::
    This option has not yet been tested during the integration in this 0.4 release, so please contact us if you are interested. 


Benchmarking
------------

This option allows the user to generate synthetic ground-truth, and assess the performance of the algorithm. We are planning to move it into a proper testsuite, and make its usage more user friendly. Currently, this is a bit undocumented and for internal use only. 

In a nutshell, five types of benchmarks can be performed from an already processed file:
    * ``fitting`` The code will select a given template, and inject multiple shuffled copies of it at various rates, at random places 
    * ``clustering`` The code will select a given template, and inject multiple shuffled copies of it at various rates and various amplitudes, at random places
    * ``synchrony`` The code will select a given template, and inject multiple shuffled copies of it on the same electrode, with a controlled pairwise correlation coefficient between those cells
    * ``smart-search`` To test the effect of the smart search. 10 cells are injected with various rates, and one has a low rate compared to the others.
    * ``drifts`` Similar to the clustering benchmark, but the amplitudes of the cells are drifting in time, with random slopes


Validating
----------

This method allows to compare the performance of the algorithm to those of a optimized classifier. This is an implementation of the BEER (Best Ellipsoidal Error Rate) estimate, as described in  `[Harris et al, 2000] <http://robotics.caltech.edu/~zoran/Reading/buzsaki00.pdf>`_. Note that the implementation is slightly more generic, and requires the installation of ``sklearn``. To use it, you need to have, if your datafile is ``mydata.extension``, a file named ``mydata/mydata.npy`` which is simply an array of all the ground truth spike times. To know more about the BEER estimate, see the devoted documentation (see :doc:`More on the BEER estimate <../advanced/beer>`)


.. _phy: https://github.com/cortex-lab/phy
.. _MATLAB: http://fr.mathworks.com/products/matlab/

