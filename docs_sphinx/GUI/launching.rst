Launching the GUI
=================

Matlab GUI
----------

To launch the MATLAB_ GUI provided with the software, you need of course to have a valid installation of MATLAB_, and you should be able to simply do::

    >> circus-gui path/data.extensions

Note that in a near future, we plan to integrate all the views of the MATLAB_ GUI into phy_

phy GUI
-------

This is not the default output of the SpyKING CIRCUS yet, but you can export your data into the kwik format, and being able to load them with phy_. To do so, at the end of the algorithm, simply do::

    >> spyking-circus path/data.extensions -m converting

This will create in the ``path`` folder a file name ``path/mydata.kwx``, and you can use phy_ to open it. To launch phy_ on the exported data, simply do::

    >> phy cluster-manual path/data.kwx

.. _phy: https://github.com/kwikteam/phy
.. _MATLAB: http://fr.mathworks.com/products/matlab/
