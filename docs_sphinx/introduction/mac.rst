Installation for Mac OS
=======================

Here are some detailed instructions:

    1. Install Anaconda_ (Python 2 or 3)

    2. The best is to create a dedicated environment to install the software, and activate it::

        >> conda create -n circus python=3.6
        >> conda activate circus

    3. Install spiking-circus by executing::

        (circus) >> conda install -c conda-forge -c spyking-circus spyking-circus
    
    4. If you prefer, you can use pip (this is equivalent to step 3)::
    
        (circus) >> pip install spyking-circus

    5. Here you are! Now run the code, for example::

        (circus) >> spyking-circus pathtodata/data.dat -c 7

If you want to install phy_ 2.0 as a visulization GUI, you can install it in the same environment (:doc:`see here for detailed instructions <../GUI/launching>` 

.. _Anaconda: https://www.anaconda.com/distribution/
.. _phy: https://github.com/cortex-lab/phy
