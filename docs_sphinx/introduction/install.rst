Installation
============

The SpyKING CIRCUS comes as a python package, and it has to be installed from sources. 
Note that currently, only unix systems are supported. 

Installation with pip
---------------------
You can install the SpyKING CIRCUS from the Python package index: https://pypi.python.org/pypi/spyking-circus

To do so, use the ``pip`` utility::

    pip install circus-0.1.tar.gz

By default, the package is installed without the CUDA dependencies. To install the code with the GPU support::

    pip install circus-0.1.tar.gz[cuda]

Note that you must have a valid CUDA installation, and **nvcc** installed.

You might want to add the ``--user`` flag, to install SpyKING CIRCUS for the local user
only, which means that you don't need administrator privileges for the
installation.

In principle, the above command also install SpyKING CIRCUS's dependencies.
Unfortunately, this does not work for ``numpy``, it has to be installed in a
separate step before all other dependencies (``pip install numpy``), if it is
not already installed.

Once the install is complete, you need to add the PATH where SpyKING CIRCUS has been installed into your local PATH, if not already the case. To do so, simply edit your ``/home/user/.bashrc`` and add the following line::

    export PATH=$PATH:$HOME/.local/bin

Then you have to relaunch the shell, and you should now have the SpyKING CIRCUS installed!


If you have an older version of pip, first update pip itself::

    # On Linux/MacOsX:
    pip install -U pip

If you don't have ``pip`` but you have the ``easy_install`` utility, you can use
it to install ``pip``::

    easy_install pip

If you have neither ``pip`` nor ``easy_install``, use the approach described
here to install ``pip``: https://pip.pypa.io/en/latest/installing.htm

Installation from source
------------------------

Alternatively, you can download the source package directly and uncompress it.
You can then simply run ``python setup.py install --user``.


Home Directory
--------------

During the install, the code will create a ``spyking-circus`` folder in ``/home/user`` where it will copy several probe designs, and a copy of the default parameter file. Note that if you are always using a similar setup, you can edit this template.


Requirements for CUDA
---------------------

Using CUDA_ is highly recommended since it can drastically increase the
speed of algorithm. To use it, you need to have a working CUDA_ environment installed onto the machine, and install the 
package as explained above.


.. _CUDA: https://developer.nvidia.com/cuda-downloads