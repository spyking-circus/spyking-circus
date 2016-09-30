import h5py, numpy, re, sys, os
from circus.shared.messages import print_error, print_and_log
from circus.shared.mpi import comm


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
    else:
        try:
            dtype_offset = int(dtype_offset)
        except Exception:
            print_error(["Offset %s is not valid" %dtype_offset])
            sys.exit(0)

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

    _description      = "mydatafile"     #Description of the file format
    _extension        = [".myextension"] #extensions
    _parallel_write   = False            #can be written in parallel (using the comm object)
    _is_writable      = False            #can be written
    _shape            = (0, 0)           #The shape of the data (nb time steps, nb channels)
    _t_start          = 0
    _t_stop           = 0


    # This is a dictionary of values that need to be provided to the constructor, with a specified type and
    # eventually default value. For example {'sampling_rate' : ['float' : 20000]}
    _requiered_fields = {}

    # Those are the attributes that need to be common in ALL file formats
    # Note that those values can be either infered from header, or otherwise read from the parameter file
    _mandatory        = ['sampling_rate', 'data_dtype', 'dtype_offset', 'gain', 'nb_channels']
    
    def __init__(self, file_name, is_empty=False, **kwargs):
        '''
        The constructor that will create the DataFile object. Note that by default, values are read from the header
        of the file. If not found in the header, they are read from the parameter file. If no values are found, the 
        code will trigger an error

        What you need to specify (usually be getting value in the _get_info function)
            - _parallel_write : can the file be safely written in parallel ?
            - _is_writable    : if the file can be written
            - _shape          : the size of the data, should be a tuple (duration in time bins, nb_channels)
            - is_empty is a flag to say if the file is created without data. It has no sense if the file is
             not writable
            - _requiered_fields : what parameter must be specified for the file format
        '''

        self.file_name = file_name
        self.is_empty  = is_empty

        f_next, extension = os.path.splitext(self.file_name)
        
        if self._extension is not None:
            if not extension in self._extension + [item.upper() for item in self._extension]:
                if self.is_master:
                    print_error(["The extension %s is not valid for a %s file" %(extension, self._description)])
                sys.exit(0)

        for key, value in kwargs.items():
            if key == 'nb_channels':
                self._shape = (0, value)
            else:
                self.__setattr__(key, value)

        if not self.is_empty:
            self._get_info_()
            self._check_valid_()

        self._check_requierements_(**kwargs)


    def get_description(self):
        result = {}
        for key in self._mandatory:
            result[key] = self.__getattribute__(key)
        return result

    def _check_valid_(self):
        for key in self._mandatory:
            if not hasattr(self, key):
                print_error(['%s is a needed attribute of a datafile, and it is not defined' %key])

    def _check_requierements_(self, **kwargs):

        missing = {}

        for key, value in self._requiered_fields.items():
            if key not in kwargs.keys():
                missing[key] = value
                print_error(['%s must be specified as type %s in the [data] section!' %(key, value[0])])
        

        if len(missing) > 0:
            self._display_requierements_()
            sys.exit(0)


    def _display_requierements_(self):

        to_write = ['The parameters for %s file format are:' %self._description.upper(), '']
        for key, values in self._requiered_fields.items():
                
            mystring = '-- %s -- of type %s' %(key, values[0])

            if values[1] is None:
                mystring += ' [** mandatory **]'
            else:
                mystring += ' [default is %s]' %values[1]

            to_write += [mystring]

        print_error(to_write)


    def _get_info_(self):
        '''
            This function is called only if the file is not empty, and should fill the values in the constructor
            such as max_offset, _shape, ...
        '''
        pass  


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

        pass

    def get_snippet(self, time, length, nodes=None):
        '''
            This function should return a time snippet of size length x nodes
            - time is in timestep
            - length is in timestep
            - nodes is a list of nodes, between 0 and nb_channels
        '''
        return self.get_data(0, chunk_size=length, padding=(time, time), nodes=nodes)


    def set_data(self, time, data):
        '''
            This function writes data at a given time.
            - time is expressed in timestep
            - data must be a 2D matrix of size time_length x nb_channels
        '''
        print_error(['No write support is implemented for %s file' %self._description])


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

        if last_chunk_len > 0:
            nb_chunks += 1

        return nb_chunks, last_chunk_len


    def open(self, mode):
        ''' 
            This function should open the file
            - mode can be to read only 'r', or to write 'w'
        '''
        pass


    def close(self):
        '''
            This function closes the file
        '''
        pass


    def allocate(self, shape, data_dtype):
        '''
            This function may be used during benchmarking mode, or if multi-files mode is activated
            Starting from an empty file, it will allocates a given size:
                - shape is a tuple with (time lenght, nb_channels)
                - data_dtype is the data type
        '''
        if self.master:
            print_error(["The method is not implemented for file format %s" %self._description])
        sys.exit(0)


    @property
    def shape(self):
        return self._shape  
         
    @property
    def nb_channels(self):
        return self._shape[1]
    
    @property
    def duration(self):
        return numpy.int64(self._shape[0])

    @property
    def is_master(self):
    	return comm.rank == 0

    @property
    def t_start(self):
        return self._t_start

    @property
    def t_stop(self):
        return self._t_stop