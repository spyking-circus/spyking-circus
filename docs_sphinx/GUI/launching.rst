Launching the GUIs
================== 

You have several options and GUIs to visualize your results, just pick the one you are the most comfortable with!

Matlab GUI
----------

To launch the MATLAB_ GUI provided with the software, you need of course to have a valid installation of MATLAB_, and you should be able to simply do::

    >> circus-gui-matlab path/mydata.extension

Note that in a near future, we plan to integrate all the views of the MATLAB_ GUI into phy_

To reload a particular dataset, that have been saved with a special ``suffix``, you just need to do::

    >> circus-gui-matlab path/mydata.extension -e suffix

This allows you to load a sorting session that has been saved and not finished. Also, if you want to load the results obtained by the :doc:`Meta Merging GUI<../code/merging>`, you need to do::

	>> circus-gui-matlab path/mydata.extension -e merged


Python GUI
----------

To launch the Python GUI, you need a valid installation of phy_ 2.0 and phylib_, and you should be able to simply do::

	>> spyking-circus path/mydata.extension -m converting -c N

Followed by::

    >> circus-gui-python path/mydata.extension

As you see, first, you need to export the data to the phy_ format using the ``converting`` option (you can use several CPUs with the ``-c`` flag if you want to export a lot of Principal Components). This is because as long as phy_ is still under development, this is not the default output of the algorithm. Depending on your parameters, a prompt will ask you if you want to compute all/some/no Principal Components for the GUI. While it may be interesting if you are familiar with classical clustering and PCs, you should not consider exploring PCs for large datasets.

.. note:: 
	
	If you want to export the results that you have processed after the :doc:`Meta Merging GUI<../code/merging>`, you just need to specify the extension to choose for the export::

		>> spyking-circus path/mydata.extension -m converging -e merged
		>> circus-gui-python path/mydata.extension -e merged


.. _phy: https://github.com/cortex-lab/phy
.. _MATLAB: http://fr.mathworks.com/products/matlab/
.. _phylib: https://github.com/cortex-lab/phylib
