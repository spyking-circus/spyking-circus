Frequently Asked Questions
==========================

Here are some questions that are popping up regularly. You can ask some or get answers on our Google Group https://groups.google.com/forum/#!forum/spyking-circus-users

* **I can not install the software**

.. note::

	Be sure to have the latest version from the git folder. We are doing our best to improve the packaging and be sure that the code is working on all platforms, but please be in touch we us if you encounter any issues

* **Is is working with Python 3?**

.. note::

	Yes, the code is compatible with Python 3

* **The data displayed do not make any sense**

.. note::

	Are you sure that the data are properly loaded? (see ``data`` section of the parameter file, especially ``data_dtype``, ``data_header``). Test everything with the preview mode by doing::

		>> spyking-circus mydata.extension -p


* **Can I process single channel datasets, or coming from not so-dense electrodes?**

.. note::

	Yes, the code can handle spikes that will occur only on a single channel, and not on a large subset. However, you may want to set the ``cc_merge`` parameter in the ``[clustering]`` section to 1, to prevent any global merges. Those global merges are indeed performed automatically by the algorithm, before the fitting phase. It assumes that templates that are similar, up to a scaling factor, can be merged because they are likely to reflect bursting neurons. But for few channels, where spatial information can not really be used to disentangle templates, the amplitude is a key factor that you want to keep. Also, you may need to turn on the ``smart_search`` mode in the ``clustering`` section, because as you have few channels, you want to collect spikes efficiently.

* **Something is wrong with the filtering**

.. note::

	Be sure to check that you are not messing around with the ``filter_done`` flag, that should be automatically changed when you perform the filtering. You can read the troubleshooting section on the filtering  :doc:`here <../issues/filtering>`


* **I see too many clusters, at the end, that should have been split**

.. note::

	The main parameters that you can change will be ``cc_merge`` and ``sim_same_elec`` in the ``[clustering]`` section. They are controlling the number of *local* (i.e. per electrode) and *global* (i.e. across the whole probe layout) merges of templates that are performed before the fitting procedure is launched. By reducing ``sim_same_elec`` (can not be less than 0), you reduce the *local* merges, and by increasing ``cc_merge`` (can not be more than 1), you reduce the *global* merges. A first recommendation would be to set ``cc_merge`` to 1. You might also want to turn on the ``smart_search`` parameter in the ``clustering`` section. This will force a smarter collection of the spikes, based on rejection methods, and thus should improve the quality of the clustering.

* **Memory usage is saturating for thousands of channels**

.. note::

	If you have a very large number of channels (>1000), then the default size of 60s for all the data blocks loaded into memory during the different steps of the algorithm may be too big. In the ``whitening`` section, you can at least change it by setting ``chunk_size`` to a smaller value (for example 10s), but this may not be enough. If you want the code to always load smaller blocks during all steps of the algorithm ``clustering, filtering``, then you need to add this ``chunk_size`` parameter into the ``data`` section.

* **How do I read the templates in Python?**

.. note::

	Templates are saved as a sparse matrix, but you can easily get access to them. For example if you want to read the template *i*, you have to do

.. code:: python

	from circus.shared.files import *
	params    = load_parameters('yourdatafile.dat')
	N_e       = params.getint('data', 'N_e') # The number of channels
	N_t       = params.getint('data', 'N_t') # The temporal width of the template
	templates = load_data(params, 'templates') # To load the templates
	temp_i = templates[:, i].toarray().reshape(N_e, N_t) # To read the template i as a 2D matrix


To know more about how to play with the data, and build your own analysis, either in Python or MATLAB_ you can go to our :doc:`dedicated section on analysis <../advanced/analysis>`


* **After merging templates with the Meta Merging GUI, waveforms are not aligned**	

.. note::

	By default, the merges do not correct for the temporal lag that may exist between two templates. For example, if you are detecting both positive and negative peaks in your recordings, you may end up with time shifted copies of the same template. This is because if the template is large enough, crossing both positive and negative thresholds at the same time, the code will collect positive and negative spikes, leading to twice the same template, misaligned. We are doing our best, at the end of the clustering step, to automatically merge those duplicates based on the cross-correlation (see parameter ``cc_merge``). However, if the lag between the two extrema is too large, or if they are slightly different, the templates may not be fused. This situation will bring a graphical issue in the phy_ GUI, while reviewing the result: if the user decided in the Meta Merging GUI to merge the templates, the waveforms will not be properly aligned. To deal with that, you simply must to set the ``correct_lag`` parameter in the ``[merging]`` section to ``True``.	Note that such a correction can not be done for merges performed in phy_.


.. _MATLAB: http://fr.mathworks.com/products/matlab/
.. _phy: https://github.com/kwikteam/phy
.. _numpy: http://www.numpy.org/
.. _HDF5: https://www.hdfgroup.org/HDF5/