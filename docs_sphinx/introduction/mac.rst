Installation for Mac OS 10.10.5
===============================

Here are some detailed instructions:

    1. Install Anaconda_ (Python 2.7 or 3.5)
    2. Install mpi4py::

        >> conda install -c mpi4py mpi4py

    4. Install spiking-circus by executing::

        >> conda install -c spyking-circus/label/dev spyking-circus
    
    5. If you prefer, you can use pip (this is equivalent to step 4)::
    
        >> pip install spyking-circus --process-dependency-links --pre
    
    6. [**optional**] Download CUDA_ through the link provided (only for GPU)
    7. [**optional**] If you want to enable CUDA_, you must have a valid ``nvcc`` install and do::
    
        >> pip install https://github.com/yger/cudamat/archive/master.zip#egg=cudamat-0.3circus

    8. Add the following to ``$HOME/.bash_profile``::

        "export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH"
        "export PYTHONPATH=$HOME/anaconda/bin:$PYTHONPATH"

    9. Here you are! Now run the code, for example::

        >> spyking-circus pathtodata/data.dat -c 7 -g 0


.. _BitBucket: https://bitbucket.org
.. _Brew: http://brew.sh/
.. _Git: https://git-scm.com/
.. _SourceTree: https://www.sourcetreeapp.com/ 
.. _Anaconda: https://www.continuum.io/downloads
.. _CUDA: https://developer.nvidia.com/cuda-downloads
.. _OpenMPI: http://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.0.tar.gz
.. _help: https://wiki.helsinki.fi/display/HUGG/Installing+Open+MPI+on+Mac+OS+X
.. _Xcode: https://developer.apple.com/xcode/download/
