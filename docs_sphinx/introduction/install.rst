Installation
============

The SpyKING CIRCUS comes as a python package, and it has to be installed from sources. 
Note that currently, only unix systems are supported. 

.. _installation_from_source:

Installation from source
------------------------
You can install the SpyKING CIRCUS from the Python package index: https://pypi.python.org/pypi/spyking-circus

To do so, use the ``pip`` utility. Newer versions of ``pip`` require you to use
the ``--pre`` option to install Brian 2 since it is not yet a final release::

    pip install --pre spyking-circus

You might want to add the ``--user`` flag, to install SpyKING CIRCUS for the local user
only, which means that you don't need administrator privileges for the
installation.

In principle, the above command also install SpyKING CIRCUS's dependencies.
Unfortunately, this does not work for ``numpy``, it has to be installed in a
separate step before all other dependencies (``pip install numpy``), if it is
not already installed.

If you have an older version of pip, first update pip itself::

    # On Linux/MacOsX:
    pip install -U pip

If you don't have ``pip`` but you have the ``easy_install`` utility, you can use
it to install ``pip``::

    easy_install pip

If you have neither ``pip`` nor ``easy_install``, use the approach described
here to install ``pip``: https://pip.pypa.io/en/latest/installing.htm

Alternatively, you can download the source package directly and uncompress it.
You can then either run ``python setup.py install`` or
``python setup.py develop`` to install it, or simply add
the source directory to your ``PYTHONPATH`` (this will only work for Python
2.x).


.. _installation_cuda:

Requirements for CUDA
---------------------

Using CUDA is highly recommended since it can drastically increase the
speed of algorithm (see :doc:`../user/computation` for details). To use it,
you need to have a working CUDA environment installed onto the machine.


Dependencies for the SpyKING CIRCUS
-----------------------------------

There are various packages that are useful but not necessary for working with
SpyKING CIRCUS. These include: matplotlib_ (for plotting). To install
them, simply do::

    pip install matplotlib


.. _matplotlib: http://matplotlib.org/
.. _ipython: http://ipython.org/
.. _travis: https://travis-ci.org/brian-team/brian2
.. _appveyor: https://ci.appveyor.com/project/brianteam/brian2
.. _nose: https://pypi.python.org/pypi/nose
.. _Cython: http://cython.org/
.. _weave: https://github.com/scipy/weave
