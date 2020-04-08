Launching the visualization GUIs
================================

You have several options and GUIs to visualize your results, just pick the one you are the most comfortable with!

Matlab GUI
----------

Installing MATLAB
~~~~~~~~~~~~~~~~~

SpyKING CIRUCS will assume that you have a valid installation of MATLAB, and that the matlab command can be found in the system $PATH. For windows user, please have a look to this `howto <https://helpdeskgeek.com/windows-10/add-windows-path-environment-variable/>`_. For unix users (mac or linux), simply add the following line to your .bash_profile or .bashrc file, in your $HOME directory::

    export $PATH=$PATH:/PATH_TO_YOUR_MATLAB/bin/matlab

Then relaunch the terminal

Launching the MATLAB GUI
~~~~~~~~~~~~~~~~~~~~~~~~

To launch the MATLAB_ GUI provided with the software, you need of course to have a valid installation of MATLAB_, and you should be able to simply do::

    >> circus-gui-matlab path/mydata.extension

Note that in a near future, we plan to integrate all the views of the MATLAB_ GUI into phy_

To reload a particular dataset, that have been saved with a special ``suffix``, you just need to do::

    >> circus-gui-matlab path/mydata.extension -e suffix

This allows you to load a sorting session that has been saved and not finished. Also, if you want to load the results obtained by the :doc:`Meta Merging GUI<../code/merging>`, you need to do::

	>> circus-gui-matlab path/mydata.extension -e merged


Phy GUI
-------

To launch the phy_ GUI (pure python based using opengl), you need a valid installation of phy_ 2.0 and phylib_.

Installing phy 2.0
~~~~~~~~~~~~~~~~~~

If you want to use the phy GUI to visualize your results, you may need to install phy_ 2.0. If you have installed SpyKING CIRCUS within a conda environment, first activate it::

    >> conda activate circus

Then, once you are in the environment, install phy_ 2.0::

    (circus) >> pip install colorcet pyopengl qtconsole requests traitlets tqdm joblib click mkdocs dask toolz mtscomp
    (circus) >> pip install --upgrade https://github.com/cortex-lab/phy/archive/master.zip
    (circus) >> pip install --upgrade https://github.com/cortex-lab/phylib/archive/master.zip

Launching the phy 2.0 GUI
~~~~~~~~~~~~~~~~~~~~~~~~~

If phy_ 2.0 is installed, you should be able to simply do::

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
