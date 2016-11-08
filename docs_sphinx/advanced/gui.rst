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

After the ``converting`` step, you must have a folder ``mydata/mydata.GUI``. You simply need to copy this folder onto a computer without SpyKING CIRCUS, but only phy_. Then you just need to copy the following code snippet into a file ``phy_launcher.py``.


.. code:: python    
    
    from phy import add_default_handler
    from phy.utils._misc import _read_python
    from phy.gui import create_app, run_app
    from phycontrib.template import TemplateController
    
    gui_params                   = {}
    gui_params['dat_path']       = DATAPATH
    gui_params['n_channels_dat'] = TOTAL_NB_CHANNELS
    gui_params['n_features_per_channel'] = 5
    gui_params['dtype']          = DATATYPE
    gui_params['offset']         = DATA_OFFSET
    gui_params['sample_rate']    = SAMPLE_RATE
    gui_params['hp_filtered']    = True

    create_app()
    controller = TemplateController(**gui_params)
    gui = controller.create_gui()

    gui.show()
    run_app()
    gui.close()
    del gui



You need to edit the appropriate values in capital letters, and then simply copy it into the ``mydata.GUI`` folder. Now you can do, once in the ``mydata.GUI`` folder::

    >> python phy_launcher.py


If the raw data are not found, the Traceview will not be displayed. If you really want to see that view, remember that you need to get the raw data **filtered**, so  you must also copy them back from your sorting machine.

.. _phy: https://github.com/kwikteam/phy
.. _MATLAB: http://fr.mathworks.com/products/matlab/

