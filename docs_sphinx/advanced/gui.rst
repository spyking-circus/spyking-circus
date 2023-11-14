GUI without SpyKING CIRCUS
==========================

MATLAB
------

You may need to launch the MATLAB GUI on a personal laptop, where the data were not processed by the software itself, so where you only have MATLAB_ and SpyKING CIRCUS is not installed. This is feasible with the following procedure:

    * Copy the the result folder ``mydata`` on your computer
    * Create a MATLAB mapping for the probe you used, i.e. ``mapping.hdf5`` (see the following procedure below to create it)
    * Open MATLAB_
    * Set the folder ``circus/matlab_GUI`` as the default path
    * Launch the following command ``SortingGUI(sampling, 'mydata/mydata', '.mat', 'mapping.hdf5', 2)``


You just need to copy the following code snippet into a file ``generate_mapping.py``.

.. code:: python

    import sys, os, numpy, h5py

    probe_file = os.path.abspath(sys.argv[1])

    def generate_matlab_mapping(probe):
        p         = {}
        positions = []
        nodes     = []
        for key in probe['channel_groups'].keys():
            p.update(probe['channel_groups'][key]['geometry'])
            nodes     +=  probe['channel_groups'][key]['channels']
            positions += [p[channel] for channel in probe['channel_groups'][key]['channels']]
        idx       = numpy.argsort(nodes)
        positions = numpy.array(positions)[idx]

        t     = "mapping.hdf5"
        cfile = h5py.File(t, 'w')
        to_write = {'positions' : positions/10., 'permutation' : numpy.sort(nodes), 'nb_total' : numpy.array([probe['total_nb_channels']])}
        for key in ['positions', 'permutation', 'nb_total']:
            cfile.create_dataset(key, data=to_write[key])
        cfile.close()
        return t

    probe = {}
    with open(probe_file, 'r') as f:
        probetext = f.read()
        exec probetext in probe

    mapping = generate_matlab_mapping(probe)


And then simply launch::

    >> python generate_mapping.py yourprobe.prb

Once this is done, you should see a file ``mapping.hdf5`` in the directory where you launch the command. This is the MATLAB_ mapping.

.. note::
    
    If you do not have ``h5py`` installed on your machine, launch this script on the machine where SpyKING CIRCUS has been launched


phy
---

After the ``converting`` step, you must have a folder ``mydata/mydata.GUI``. You simply need to copy this folder onto a computer without SpyKING CIRCUS, but only phy_ and phylib_. In this folder, you should see a file ``params.py``, generated during the ``converting`` step. So in a terminal, you simply need to go to this folder, and launch from a terminal::
    
    >> phy template-gui params.py


If the raw data are not found, the Traceview will not be displayed. If you really want to see that view, remember that you need to get the raw data **filtered**, so  you must also copy them back from your sorting machine.

.. _phy: https://github.com/cortex-lab/phy
.. _phylib: https://github.com/cortex-lab/phylib
.. _MATLAB: http://fr.mathworks.com/products/matlab/

