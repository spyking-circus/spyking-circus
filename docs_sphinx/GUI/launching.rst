Launching the GUI
-----------------

Matlab GUI
~~~~~~~~~~

To launch the MATLAB GUI provided with the software, you need of course to have a valid installtion of MATLAB, and you should be able to simply do::

    >> circus-gui path/data.extensions

Note that in a near future, we plan to integrate all the views of the MATLAB GUI into phy

phy GUI

This is not the default output of the SpyKING CIRCUS yet, but you can export your data into the kwik format, and being able to load them with phy. To do so, at the end of the algorithm, simply do::

    >> spyking-circus path/data.extensions -m converting

This will create in the **path** folder a file name **path/mydata.kwx**, and you can use phy to open it.
