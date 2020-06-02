Installation for Windows
========================

Here are some detailed instructions:

    1. Install Anaconda_ (Python 2 or 3)
    2. Launched Anaconda's command line
    3. The best is to create a dedicated environment to install the software, and activate it::

        >> conda create -n circus python=3.6
        >> conda activate circus

    4. Execute:: 

        (circus) >> conda install -c intel mpi4py

    5. You should see the following message:
        * mpi4py requires that Microsoft MPI be installed on the host system. That is a system wide installation that is currently not available through conda. In order to successfully use mpi4py you must install Microsoft MPI (both .exe and .msi) and then append the bin directory of the MPI installation to your PATH environment variable. To install Microsoft MPI see MPIv10_

    6. Install microsoft MPIv10_ (both .msi and .exe) as requested with the previous link
    7. Add the mpi installation's bin directory to the path (``C:\program files\microsoft mpi\bin``)

    8. Install spiking-circus by executing::

        (circus) >> conda install -c conda-forge -c intel -c spyking-circus spyking-circus
    
    9. If you prefer, you can use pip (this is equivalent to step 8)::
    
        (circus) >> pip install spyking-circus

    10. Here you are! Now run the code, for example::

        (circus) >> spyking-circus pathtodata/data.dat -c 7


If you want to install phy_ 2.0 as a visulization GUI, you can install it in the same environment (:doc:`see here for detailed instructions <../GUI/launching>` 

.. _Anaconda: https://www.anaconda.com/distribution/
.. _MPIv10: https://www.microsoft.com/en-us/download/details.aspx?id=57467
.. _phy: https://github.com/cortex-lab/phy