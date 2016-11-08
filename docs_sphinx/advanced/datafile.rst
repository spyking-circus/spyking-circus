Writing your custom file wrapper
================================

Since 0.5, SpyKING CIRCUS can natively read/write several file formats, in order to ease your sorting workflow. By default, some generic
file formats are already implemented (see :doc:`the documentation on the file formats <../code/fileformat>`), but you can also write your own wrapper in order to read/write your own custom datafile.

Note that we did not used neo_, and we recommend not to do so, because your wrapper should have some functionalities not allowed yet by neo_::
    * it should allow memory mapping, i.e. to read only chunks of your data at a time, slicing either by time or by channels. 
    * it should read data in their native format, as they will internally be turned into ``float32``
    * it could allow streaming, if data are internally stored in several chunks

To do so, you simply need to create an object that will inherit from the ``DataFile`` object described in ``circus/files/datafile.py``. The easy thing
to understand the structure is to have a look to ``circus/files/raw_binary.py`` as an example of such a datafile object. If you have questions while writing your 
wrapper, do not hesitate to be in touch with us. 

The speed of the algorithm may slow down a little, depending on your wrapper. For example, currently, we provide an example of a wrapper based on neuroshare_ (mcd files). 
This wrapper is working, but slow and inefficient, because the neuroshare_ API is slow on its own. 


Mandatory attributes
--------------------

Here are the class attributes that you must define::

    description      = "mydatafile"     # Description of the file format
    extension        = [".myextension"] # extensions allowed
    parallel_write   = False            # can be written in parallel (using the comm object)
    is_writable      = False            # can be written
    is_streamable    = ['multi-files']  # If the file format can support streams of data ['multi-files' is a default, but can be something else]
    _shape           = None             # The total shape of the data (nb time steps, nb channels) across streams if any
    _t_start         = None             # The global t_start of the data (0 by default)
    _t_stop          = None             # The final t_stop of the data, across all streams if any
    _params          = {}               # The dictionary where all attributes will be saved


Note that the datafile objects has an internal dictionary ``_params`` that contains all the values provided by the Configuration Parser, i.e. read from the parameter file
in the data section. For a given file format, you can specify::

    # This is a dictionary of values that need to be provided to the constructor, with the corresponding type
    _required_fields = {}

This is the list of mandatory parameters, along with the type, that have to be specify in the parameter file, because they can not be inferred from the header of your data file. For example::

    _required_files = {'sampling_rate' : float, 'nb_channels' : int}


Then you can also specify some additional parameters, that may have a default value. If they are not provided in the parameter file, this default value is used. For example::

    # This is a dictionary of values that may have a default value, if not provided to the constructor
    _default_values  = {'gain' : 1.}


At the end, there are 5 mandatory attributes that the code will require for any given file format. Those should be stored in the ``_params`` dictionary:

    * ``nb_channels``
    * ``sampling_rate``
    * ``data_dtype``
    * ``dtype_offset``
    * ``gain``


Custom methods
--------------

Here is the list of the function that you should implement in order to have a valid wrapper

Basics IO
~~~~~~~~~

You must provide function to open/close the datafile::

    def _open(self, mode=''):
        ''' 
            This function should open the file
            - mode can be to read only 'r', or to write 'w'
        '''
        raise NotImplementedError('The open method needs to be implemented for file format %s' %self.description)


    def _close(self):
        '''
            This function closes the file
        '''
        raise NotImplementedError('The close method needs to be implemented for file format %s' %self.description)


Reading values from the header
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You need to provide a function that will read data from the header of your datafile::

    def _read_from_header(self):
        '''
            This function is called only if the file is not empty, and should fill the values in the constructor
            such as _shape. It returns a dictionary, that will be added to self._params based on the constrains given by
            required_fields and default_values
        '''
        raise NotImplementedError('The _read_from_header method needs to be implemented for file format %s' %self.description)


Such a function must:
    * set _shape to (duration, nb_channels)
    * set _t_start if not 0
    * return a dictionary of parameters that will be used, given the constrains obtained from values in _required_fields and _default_values, to create the DataFile


Reading chunks of data
~~~~~~~~~~~~~~~~~~~~~~

Then you need to provide a function to load a block of data, with a given size::    

    def read_chunk(self, idx, chunk_size, padding=(0, 0), nodes=None):
        '''
        Assuming the analyze function has been called before, this is the main function
        used by the code, in all steps, to get data chunks. More precisely, assuming your
        dataset can be divided in nb_chunks (see analyze) of temporal size (chunk_size), 

            - idx is the index of the chunk you want to load
            - chunk_size is the time of those chunks, in time steps
            - if the data loaded are data[idx:idx+1], padding should add some offsets, 
                in time steps, such that we can load data[idx+padding[0]:idx+padding[1]]
            - nodes is a list of nodes, between 0 and nb_channels            
        '''

        raise NotImplementedError('The get_data method needs to be implemented for file format %s' %self.description)


Note that for convenience, in such a function, you can obtained local t_start, t_stop by using the method ``t_start, t_stop = _get_t_start_t_stop(idx, chunk_size, padding)`` (see ``circus/files/raw_binary.py`` for an example). This may be easier to slice your datafile. At the end, data must be returned as ``float32``, and to do so, you can also use the internal method ``_scale_data_to_float32(local_chunk)``


Writing chunks of data
~~~~~~~~~~~~~~~~~~~~~~

This method is required only if your file format is allowing write access::

    def write_chunk(self, time, data):
        '''
            This function writes data at a given time.
            - time is expressed in time step
            - data must be a 2D matrix of size time_length x nb_channels
        '''
        raise NotImplementedError('The set_data method needs to be implemented for file format %s' %self.description)


Streams
-------

Depending on the complexity of your file format, you can allow several ways of streaming into your data. The way to define streams is rather simple, and by default, all files format can be streamed
with a mode called ``multi-files``. This is the former ``multi-files`` mode that we used to have in 0.4 versions (see MULTI)::

    def set_streams(self, stream_mode):
        '''
            This function is only used for file format supporting streams, and need to return a list of datafiles, with
            appropriate t_start for each of them. Note that the results will be using the times defined by the streams. 
            You can do anything regarding the keyword used for the stream mode, but multi-files is implemented by default
            This will allow every file format to be streamed from multiple sources, and processed as a single file.
        '''

        if stream_mode == 'multi-files':
            dirname         = os.path.abspath(os.path.dirname(self.file_name))
            all_files       = os.listdir(dirname)
            fname           = os.path.basename(self.file_name)
            fn, ext         = os.path.splitext(fname)
            head, sep, tail = fn.rpartition('_')
            mindigits       = len(tail)
            basefn, fnum    = head, int(tail)
            fmtstring       = '_%%0%dd%%s' % mindigits
            sources         = []
            to_write        = []
            global_time     = 0
            params          = self.get_description()

            while fname in all_files:
                new_data   = type(self)(os.path.join(os.path.abspath(dirname), fname), params)
                new_data._t_start = global_time
                global_time += new_data.duration
                sources     += [new_data]
                fnum        += 1
                fmtstring    = '_%%0%dd%%s' % mindigits
                fname        = basefn + fmtstring % (fnum, ext)
                to_write    += ['We found the datafile %s with t_start %s and duration %s' %(new_data.file_name, new_data.t_start, new_data.duration)]

            print_and_log(to_write, 'debug', logger)
            return sources


As you can see, set_streams is a function that given a ``stream_mode``, will read the parameters and return a list of DataFiles, created by slightly changing those parameters. In the case of ``multi-files``, this is just a change in the file names, but for some file formats, streams are embedded within the same data structure, and not spread over several files. For example, if you have a look to the file ``circus/files/kwd.py`` you can see that there is also a mode for streams call ``single-file``. If this mode is enabled, the code will process all chunks of data in the HDF5 file, sorted by their keys, as a single giant data file. This is a common situation in experiment. Chunks of data are recorded at several times, but in the same data file. Because they are originating from the same experiment, they better be processed as a whole.



Once those functions are implemented, you simply need to add your wrapper in the list defined in ``circus/files/__init__.py``. Or be in touch with us to make it available in the default trunk.


Parallelism
-----------

In all your wrappers, if you want to deal with parallelism and do read/write access that will depend on MPI, you have access to an object ``comm`` which is the MPI communicator. Simply add at the top of your python wrapper::

    from circus.shared.mpi import comm

And then have a look for example ``circus/files/hdf5.py`` to understand how this is used

Logs
----

In all your wrappers, if you want to log some informations to the log files (in addition to those logged by default in the DataFile class), you can use the ``print_and_log`` function. Simply add at the top of your wrapper::

    from circus.shared.messages import print_and_log
    import logging
    logger = logging.getLogger(__name__)


Then, if you want to log something, the syntax of such a function is::

    >> print_and_log(list_of_lines, 'debug', logger)


.. _neo: https://github.com/NeuralEnsemble/python-neo
.. _neuroshare: http://neuroshare.sourceforge.net/index.shtml