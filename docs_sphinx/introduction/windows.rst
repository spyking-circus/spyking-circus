Installation for Windows
========================

Here are some detailed instructions:

    1. Install Anaconda_ (Python 2.7 or 3.5)
    2. Launched Anaconda's command line
    3. Execute ``conda install mpi4py``
    4. You should see the following message:
        * mpi4py requires that Microsoft MPI 6 be installed on the host system. That is a system wide installation that is currently not available through conda. In order to successfully use mpi4py you must install Microsoft MPI and then append the bin directory of the MPI installation to your PATH environment variable. To install Microsoft MPI see MPIv6_

    5. Install microsoft MPIv6_ as requested with the previous link
    6. Add the mpi installation's bin directory to the path (``C:\program files\microsoft mpi\bin``)
    7. Install spiking-circus by executing::

        >> conda install -c conda-forge tqdm
        >> conda install -c mpi4py mpi4py
        >> conda install -c spyking-circus/label/dev spyking-circus
    
    8. If you prefer, you can use pip (this is equivalent to step 7)::
    
        >> pip install spyking-circus --process-dependency-links --pre

    9. [**optional**] Install CUDA_ for windows to activate GPU only
    10. [**optional**] If you want to enable CUDA_, you must have a valid ``nvcc`` install and do::
        
        >> pip install https://github.com/yger/cudamat/archive/master.zip#egg=cudamat-0.3circus
        
    11. Here you are! Now run the code, for example::

        >> spyking-circus pathtodata/data.dat -c 7 -g 0

    
.. _BitBucket: https://bitbucket.org
.. _Git: https://git-scm.com/
.. _SourceTree: https://www.sourcetreeapp.com/ 
.. _Anaconda: https://www.continuum.io/downloads
.. _Pycharm: https://www.jetbrains.com/pycharm/
.. _MPIv6: https://www.microsoft.com/en-us/download/details.aspx?id=47259
.. _mpi4py: http://www.lfd.uci.edu/~gohlke/pythonlibs/#mpi4py
.. _CUDA: https://developer.nvidia.com/cuda-downloads
.. _here: http://www.microsoft.com/en-us/download/details.aspx?id=44266