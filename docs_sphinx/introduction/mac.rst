Installation for Mac OS 10.10.5
===============================

Here are some detailed instructions:

    1. Install Anaconda_ (Python 2.7 or 3.5)

    2. Install spiking-circus by executing::

        >> conda install -c conda-forge -c spyking-circus spyking-circus
    
    3. If you prefer, you can use pip (this is equivalent to step 4)::
    
        >> pip install spyking-circus

    4. Add the following to ``$HOME/.bash_profile``::

        "export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH"
        "export PYTHONPATH=$HOME/anaconda/bin:$PYTHONPATH"

    5. Here you are! Now run the code, for example::

        >> spyking-circus pathtodata/data.dat -c 7


.. _Anaconda: https://www.anaconda.com/distribution/
