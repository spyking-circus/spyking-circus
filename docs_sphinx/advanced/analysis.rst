Example scripts
===============

On this page, you will be very simple example of scripts to load/play a bit with the raw results, either in Python or in Matlab. This is not exhaustive, this is simply an example to show you how you can integrate your own workflow on the results.

.. warning::

	Note that in Python templates (i.e. cells) indices start at 0, while they start at 1 in MATLAB.

Display a template
------------------

If you want to display the particular template *i*, as a 2D matrix of size :math:`N_e` x :math:`N_t` (respectively the number of channels and the temporal width of your template)

Python
~~~~~~

.. code:: python

	from circus.shared.parser import CircusParser
	from circus.shared.files import load_data
	from pylab import *
	params    = CircusParser('yourdatafile.dat')
	N_e       = params.getint('data', 'N_e') # The number of channels
	N_t       = params.getint('detection', 'N_t') # The temporal width of the template
	templates = load_data(params, 'templates') # To load the templates
	temp_i = templates[:, i].toarray().reshape(N_e, N_t) # To read the template i as a 2D matrix
	imshow(temp_i, aspect='auto')

Matlab
~~~~~~

.. code:: matlab

	tmpfile        = 'yourdata/yourdata.templates.hdf5';
	templates_size = double(h5read(tmpfile, '/temp_shape'));
	N_e = templates_size(2); 
	N_t = templates_size(1);
	temp_x = double(h5read(tmpfile, '/temp_x') + 1);
	temp_y = double(h5read(tmpfile, '/temp_y') + 1); 
	temp_z = double(h5read(tmpfile, '/temp_data'));
	templates = sparse(temp_x, temp_y, temp_z, templates_size(1)*templates_size(2), templates_size(3));
	templates_size = [templates_size(1) templates_size(2) templates_size(3)/2];
	temp_i = full(reshape(templates(:, tmpnum), templates_size(2), templates_size(1)))';
	imshow(temp_i)


Compute ISI
-----------

If you want to compute the inter-spike intervals of cell *i*

Python
~~~~~~

.. code:: python

	from circus.shared.parser import CircusParser
	from circus.shared.files import load_data
	from pylab import *
	params    = CircusParser('yourdatafile.dat')
	results   = load_data(params, 'results')
	spikes    = results['spiketimes']['temp_i']
	isis      = numpy.diff(spikes)
	hist(isis)


Matlab
~~~~~~

.. code:: matlab

	tmpfile = 'yourdata/yourdata.results.hdf5';
	spikes  = double(h5read(tmpfile, '/spiketimes/temp_i'));
	isis    = diff(spikes);
	hist(isis)


Display the amplitude over time for a given template
----------------------------------------------------

If you want to show a plot of cell *i* spike times vs. amplitudes

Python
~~~~~~

.. code:: python

	from circus.shared.parser import CircusParser
	from circus.shared.files import load_data
	from pylab import *
	params    = CircusParser('yourdatafile.dat')
	results   = load_data(params, 'results')
	spikes    = results['spiketimes']['temp_i']
	amps      = results['amplitudes']['temp_i'][:, 0] # The second column are amplitude for orthogonal, not needed
	plot(spikes, amps, '.')


Matlab
~~~~~~

.. code:: matlab

	tmpfile = 'yourdata/yourdata.results.hdf5';
	spikes  = double(h5read(tmpfile, '/spiketimes/temp_i'));
	amps    = double(h5read(tmpfile, '/amplitudes/temp_i')(:,1));
	plot(spikes, amps, '.')