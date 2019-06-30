Parameters
==========

Display the helpers
-------------------

To know what are all the parameters of the software, just do::
    
    >> spyking-circus -h

To know what are all the file formats supported by the software, just do::
    
    >> spyking-circus help -i

To know more what are the parameter of a given file format *X*, just do ::

    >> spyking-circus X -i

Command line Parameters
-----------------------

The parameters to launch the program are:

* ``-m`` or ``--method``

What are the steps of the algorithm you would like to perform. Defaults steps are:

    1. filtering
    2. whitening
    3. clustering
    4. fitting
    5. merging


Note that filtering is performed only once, and if the code is relaunched on the same data, a flag in the parameter file will prevent the code to filter twice. You can specify only a subset of steps by doing::
    
    >> spyking-circus path/mydata.extension -m clustering,fitting

.. note::

    The results of the ``merging`` step are still saved with a different extension compared to the full results of the algorithm. This is because we don't claim that a full automation of the software can work out of the box for all dataset, areas, species, ... So if you want to work from merged results, use the ``-e merged`` extension while converting/displaying results. But otherwise, just look to the ra results, without the merging step (see the devoted section :doc:`documentation on Meta Merging <../code/merging>`), or even more (:doc:`documentation on extra steps <../advanced/extras>`).

* ``-c`` or ``--cpu``

The number of CPU that will be used by the code. For example, just do::

    >> spyking-circus path/mydata.extension -m clustering,fitting -c 10


* ``-H`` or ``--hostfile``

The CPUs used depends on your MPI configuration. If you wan to configure them, you must provide a specific hostfile and do::

    >> spyking-circus path/mydata.extension -c 10 -H nodes.hosts

To know more about the host file, see the MPI section :doc:`documentation on MPI <../introduction/mpi>`

* ``-b`` or ``--batch``

The code can accept a text file with several commands that will be executed one after the other, in a batch mode. This is interesting for processing several datasets in a row. An example of such a text file ``commands.txt`` would simply be::
    
    path/mydata1.extention -c 10
    path/mydata2.extention -c 10 -m fitting
    path/mydata3.extention -c 10 -m clustering,fitting,converting

Then simply launch the code by doing::

    >> spyking-circus commands.txt -b

.. warning::

    When processing files in a batch mode, be sure that the parameters file have been pre-generated. Otherwise, the code will hang asking you to generate them

* ``-p`` or ``--preview``

To be sure that data are properly loaded before filtering everything on site, the code will load only the first second of the data, computes thresholds, and show you an interactive GUI to visualize everything. Please see the :doc:`documentation on Python GUI <../GUI/python>`

.. note::

    The preview mode does not modify the data file!

* ``-r`` or ``--result``

Launch an interactive GUI to show you, superimposed, the activity on your electrodes and the reconstruction provided by the software. This has to be used as a sanity check. Please see the :doc:`documentation on Python GUI <../GUI/python>`

* ``-s`` or ``--second``

If the preview mode is activated, by default, it will show the first 2 seconds of the data. But you can specify an offset, in second, with this extra parameter such that the preview mode will display the signal in [second, second+2]

* ``-o`` or ``--output``

If you want to generate synthetic benchmarks from a dataset that you have already sorted, this allows you, using the ``benchmarking`` mode, to produce a new file ``output`` based on what type of benchmarks you want to do (see ``type``)

* ``-t`` or ``--type``

While generating synthetic datasets, you have to chose from one of those three possibilities: ``fitting``, ``clustering``, ``synchrony``. To know more about what those benchmarks are, see the :doc:`documentation on extra steps <../advanced/extras>`
    
.. note::

    Benchmarks will be better integrated soon into an automatic test suite, use them at your own risks for now. To know more about the additional extra steps, :doc:`documentation on extra steps <../advanced/extras>`

Configuration File
------------------

The code, when launched for the first time, generates a parameter file. The default template used for the parameter files is the one located in ``/home/user/spyking-circus/config.params``. You can edit it in advance if you are always using the same setup.

To know more about what is in the configuration file, :doc:`documentation on the configuration <../code/config>`