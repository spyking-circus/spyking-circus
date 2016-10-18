import h5py, numpy, re, sys, os, logging
from circus.shared.messages import print_and_log
from circus.shared.mpi import comm

logger = logging.getLogger(__name__)


def get_offset(data_dtype, dtype_offset):

    if dtype_offset == 'auto':
        if data_dtype in ['uint16', numpy.uint16]:
            dtype_offset = 32768
        elif data_dtype in ['int16', numpy.int16]:
            dtype_offset = 0
        elif data_dtype in ['float32', numpy.float32]:
            dtype_offset = 0
        elif data_dtype in ['int8', numpy.int8]:
            dtype_offset = 0        
        elif data_dtype in ['uint8', numpy.uint8]:
            dtype_offset = 127
        elif data_dtype in ['float64', numpy.float64]:
            dtype_offset = 0    
        if comm.rank == 0:
            print_and_log(['data type offset for %s is automatically set to %d' %(data_dtype, dtype_offset)], 'debug', logger)
    else:
        try:
            dtype_offset = int(dtype_offset)
        except Exception:
            if comm.rank == 0:
                print_and_log(["Offset %s is not valid" %dtype_offset], 'error', logger)
            sys.exit(1)

    return dtype_offset

class DataFile(object):

    '''
    A generic class that will represent how the program interacts with the data. Such an abstraction
    layer should allow people to write their own wrappers, for several file formats. Note that 
    depending on the complexity of the datastructure, this can slow down the code.

    The method belows are all methods that can be used, at some point, by the different steps of the code. 
    In order to provide a full compatibility with a given file format, they must all be implemented, but 
    feel free to reuse as much as possible those from the datafile main class.
    '''

    description      = "mydatafile"     # Description of the file format
    extension        = [".myextension"] # extensions
    parallel_write   = False            # can be written in parallel (using the comm object)
    is_writable      = False            # can be written
    is_streamable    = False            # If the file formats can support streams of data
    _shape           = None             # The total shape of the data (nb time steps, nb channels) accross streams if any
    _t_start         = None             # The global t_start of the data
    _t_stop          = None             # The final t_stop of the data, accross all streams if any
    
    # This is a dictionary of values that need to be provided to the constructor, with the corresponding type
    _required_fields = {}

    # This is a dictionary of values that may have a default value, if not provided to the constructor
    _default_values  = {}

    
    def __init__(self, file_name, params, is_empty=False):
        '''
        The constructor that will create the DataFile object. Note that by default, values are read from the header
        of the file. If not found in the header, they are read from the parameter file. If no values are found, the 
        code will trigger an error

        What you need to specify (usually be getting value in the _get_info function)
            - parallel_write : can the file be safely written in parallel ?
            - is_writable    : if the file can be written
            - is_streamable  : if the file format can support streaming data
            - _shape         : the size of the data, should be a tuple (duration in time bins, nb_channels)
            - is_empty is a flag to say if the file is created without data. It has no sense if the file is
             not writable
            - required_fields : what parameter must be specified for the file format, along with the type
            - default_values  : parameters that may have default values if not provided
        '''

        if not is_empty:
            self._check_filename(file_name)

        self._params   = {}
        self.file_name = file_name
        self.is_empty  = is_empty

        f_next, extension = os.path.splitext(self.file_name)
        
        self._check_extension(extension)
        self._fill_from_params(params)

        if not self.is_empty:
            #try:
            self._fill_from_header(self._read_from_header())
            #except Exception as ex:
            #    print_and_log(["There is an error in the _read_from_header method of the wrapper\n" + str(ex)], 'error', logger)
        else:
            self._shape = (0, 0)

        if self._shape is None:
            if self.is_master:
                print_and_log(["Shape of the data is not defined. Are you sure of the wrapper?"], 'error', logger)

        self._params['dtype_offset'] = get_offset(self.data_dtype, self.dtype_offset)


    ##################################################################################################################
    ##################################################################################################################
    #########                  Methods that need to be overwritten for a given fileformat                      #######
    ##################################################################################################################
    ##################################################################################################################


    def _read_from_header(self):
        '''
            This function is called only if the file is not empty, and should fill the values in the constructor
            such as _shape
        '''
        raise NotImplementedError('The _read_from_header method needs to be implemented for file format %s' %self.description)


    
    def get_data(self, idx, chunk_size, padding=(0, 0), nodes=None):
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


    def set_data(self, time, data):
        '''
            This function writes data at a given time.
            - time is expressed in timestep
            - data must be a 2D matrix of size time_length x nb_channels
        '''
        raise NotImplementedError('The set_data method needs to be implemented for file format %s' %self.description)


    def open(self, mode):
        ''' 
            This function should open the file
            - mode can be to read only 'r', or to write 'w'
        '''
        raise NotImplementedError('The open method needs to be implemented for file format %s' %self.description)


    def close(self):
        '''
            This function closes the file
        '''
        raise NotImplementedError('The close method needs to be implemented for file format %s' %self.description)


    def allocate(self, shape, data_dtype):
        '''
            This function may be used during benchmarking mode, or if multi-files mode is activated
            Starting from an empty file, it will allocates a given size:
                - shape is a tuple with (time length, nb_channels)
                - data_dtype is the data type
        '''
        raise NotImplementedError('The allocate method needs to be implemented for file format %s' %self.description)


    ################################## Optional, only if internal names are changed ##################################

    @property
    def sampling_rate(self):
        return self._params['sampling_rate']

    @property
    def data_dtype(self):
        return self._params['data_dtype']

    @property
    def dtype_offset(self):
        return self._params['dtype_offset']

    @property
    def data_offset(self):
        return self._params['data_offset']
    
    @property
    def nb_channels(self):
        return self._params['nb_channels']

    @property
    def gain(self):
        return self._params['gain']

    ##################################################################################################################
    ##################################################################################################################
    #########           End of methods that need to be overwritten for a given fileformat                      #######
    ##################################################################################################################
    ##################################################################################################################

    def _check_filename(self, file_name):
        if not os.path.exists(file_name):
            if self.is_master:
                print_and_log(["The file %s can not be found!" %file_name], 'error', logger)
            sys.exit(1)

    def _check_extension(self, extension):
        if self.extension is not None:
            if not extension in self.extension + [item.upper() for item in self.extension]:
                if self.is_master:
                    print_and_log(["The extension %s is not valid for a %s file" %(extension, self.description)], 'error', logger)
                sys.exit(1)

    def _fill_from_params(self, params):
    
        for key in self._required_fields:
            if not params.has_key(key):
                self._check_requirements_(params)
            else:
                self._params[key] = self._required_fields[key](params[key])
                if self.is_master:
                    print_and_log(['%s is read from the params with a value of %s' %(key, self._params[key])], 'debug', logger)

        for key in self._default_values:
            if not params.has_key(key):
                self._params[key] = self._default_values[key]
                if self.is_master:
                    print_and_log(['%s is not set and has the default value of %s' %(key, self._params[key])], 'debug', logger)
            else:
                self._params[key] = type(self._default_values[key])(params[key])
                if self.is_master:
                    print_and_log(['%s is read from the params with a value of %s' %(key, self._params[key])], 'debug', logger)


    def _fill_from_header(self, header):
       
        for key in header.keys():
            self._params[key] = header[key]
            if self.is_master:
                print_and_log(['%s is read from the header with a value of %s' %(key, self._params[key])], 'debug', logger)

    def _check_requirements_(self, params):

        missing = {}

        for key, value in self._required_fields.items():
            if key not in params.keys():
                missing[key] = value
                if self.is_master:
                    print_and_log(['%s must be specified as type %s in the [data] section!' %(key, str(value))], 'error', logger)
        
        if len(missing) > 0:
            self._display_requirements_()
            sys.exit(1)


    def _display_requirements_(self):

        to_write = ['The parameters for %s file format are:' %self.description.upper(), '']

        for key, value in self._required_fields.items():    
            mystring = '-- %s -- of type %s' %(key, str(value))
            mystring += ' [** mandatory **]'
            to_write += [mystring]

        for key, value in self._default_values.items():            
            mystring = '-- %s -- of type %s' %(key, str(type(value)))
            mystring += ' [default is %s]' %value
            to_write += [mystring]

        if self.is_master:
            print_and_log(to_write, 'error', logger)


    def _scale_data_to_float32(self, data):
        '''
            This function will convert data from local data dtype into float32, the default format of the algorithm
        '''

        if self.data_dtype != numpy.float32:
            data  = data.astype(numpy.float32)

        if self.dtype_offset != 0:
            data  -= self.dtype_offset

        if numpy.any(self.gain != 1):
            data *= self.gain

        return numpy.ascontiguousarray(data)

    def _unscale_data_from_from32(self, data):
        '''
            This function will convert data from float32 back to the original format of the file
        '''


        if numpy.any(self.gain != 1):
            data /= self.gain
        
        if self.dtype_offset != 0:
            data  += self.dtype_offset
        
        if data.dtype != self.data_dtype:
            data = data.astype(self.data_dtype)

        return data

    def get_snippet(self, time, length, nodes=None):
        '''
            This function should return a time snippet of size length x nodes
            - time is in timestep
            - length is in timestep
            - nodes is a list of nodes, between 0 and nb_channels
        '''
        return self.get_data(0, chunk_size=length, padding=(time, time), nodes=nodes)


    def analyze(self, chunk_size):
        '''
            This function should return two values: 
            - the number of temporal chunks of temporal size chunk_size that can be found 
            in the data. Note that even if the last chunk is not complete, it has to be 
            counted. chunk_size is expressed in time steps
            - the length of the last uncomplete chunk, in time steps
        '''
        nb_chunks      = self.duration // chunk_size
        last_chunk_len = self.duration - nb_chunks * chunk_size

        if self.is_master:
            print_and_log(['There are %d chunks of size %d' %(nb_chunks, chunk_size)], 'debug', logger)

        if last_chunk_len > 0:
            nb_chunks += 1

        if self.is_master:
            print_and_log(['The last chunk has size %d' %(last_chunk_len)], 'debug', logger)

        return nb_chunks, last_chunk_len

    def get_description(self):
        result = {}
        for key in ['sampling_rate', 'data_dtype', 'gain', 'nb_channels', 'dtype_offset']:
            result[key] = self._params[key]
        return result

    @property
    def shape(self):
        return self._shape  
             
    @property
    def duration(self):
        return numpy.int64(self._shape[0])

    @property
    def is_master(self):
        return comm.rank == 0

    @property
    def t_start(self):
        if self._t_start is None:
            self._t_start = 0
        return self._t_start

    @property
    def t_stop(self):
        if self._t_stop is None:
            self._t_stop = self.duration
        return self._t_stop
