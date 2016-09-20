import h5py, numpy, re, sys, os
import ConfigParser as configparser

from circus.shared.messages import print_error, print_and_log

def _check_requierements_(description, fields, params, **kwargs):

    for key, values in fields.items():
        if key not in kwargs.keys():
            try:
                value, default = values
                if default is not None:
                    kwargs[key] = default
                else:
                    if value == 'int':
                        kwargs[key] = params.getint('data', key)
                    elif value == 'string':
                        kwargs[key] = params.get('data', key)
                    elif value == 'float':
                        kwargs[key] = params.getfloat('data', key)
                    elif value == 'bool':
                        kwargs[key] = params.getboolean('data', key)
            except Exception:
                _display_requierements_(description, params, fields)
                print_error(['%s must be specified as type %s in the [data] section!' %(key, value)])
                sys.exit(0)
    return kwargs


def _display_requierements_(description, params, fields):

    to_write = ['The parameters for %s file format are:' %description.upper(), '']
    for key, values in fields.items():
            
        mystring = '-- %s -- of type %s' %(key, values[0])

        if values[1] is None:
            mystring += ' [mandatory]'
        else:
            mystring += ' [default is %s]' %values[1]

        to_write += [mystring]

    print_and_log(to_write, 'info', params)


class DataFile(object):

    '''
    A generic class that will represent how the program interacts with the data. Such an abstraction
    layer should allow people to write their own wrappers, for several file formats. Note that 
    depending on the complexity of the datastructure, this can slow down the code.

    The method belows are all methods that can be used, at some point, by the different steps of the code. 
    In order to provide a full compatibility with a given file format, they must all be implemented.

    Note also that you must specify if your file format allows parallel write calls, as this is used in
    the filtering and benchmarking steps.
    '''

    _description      = "mydatafile"    
    _extension        = [".myextension"]
    _parallel_write   = False
    _is_writable      = False
    _requiered_fields = {}
    # Note that those values can be either infered from header, or otherwise read from the parameter file

    def __init__(self, file_name, params, empty=False, comm=None, **kwargs):
        '''
        The constructor that will create the DataFile object. Note that by default, values are read from
        the parameter file, but you could completly fill them based on values that would be obtained
        from the datafile itself. 
        What you need to specify
            - _parallel_write : can the file be safely written in parallel ?
            - _is_writable    : if the file can be written
            - _shape          : the size of the data, should be a tuple (max_offset, N_tot)
            - max_offset      : the time length of the data, in time steps
            - comm is a MPI communicator ring, if the file is created in a MPI environment
            - empty is a flag to say if the file is created without data

        Note that you can overwrite values such as N_e, rate from the header in your data. Those will then be
        used in the code, instead of the ones from the parameter files.

        Note also that the code can create empty files [multi-file, benchmarking], this is why there is an empty
        flag to warn the constructor about the fact that the file may be empty
        '''

        self.file_name = file_name
        self.empty     = empty
        self.comm      = comm

        assert isinstance(params, configparser.ConfigParser)
        self.params = params

        f_next, extension = os.path.splitext(self.file_name)
        
        if self._extension is not None:
            if not extension in self._extension + [item.upper() for item in self._extension]:
                print_error(["The extension %s is not valid for a %s file" %(extension, self._description)])
                sys.exit(0)

        requiered_values = {'rate'  : ['data', 'sampling_rate'], 
                            'N_e'   : ['data', 'N_e'],
                            'N_tot' : ['data', 'N_total']}

        for key, value in kwargs.items():
            self.__setattr__(key, value)

        for key, value in requiered_values.items():
            if not hasattr(self, key):
                to_be_set = numpy.int64(self.params.getint(value[0], value[1]))
                self.__setattr__(key, to_be_set)
                print_and_log(['%s is read from the params with a value of %s' %(key, to_be_set)], 'debug', self.params)
            else:
                print_and_log(['%s is infered from the data file with a value of %s' %(key, value)], 'debug', self.params)


        self.max_offset  = 0
        self._shape      = None
        print_and_log(["The datafile %s with type %s has been created" %(self.file_name, self._description)], 'debug', self.params)

        if not self.empty:
            self._get_info_()
            
            chunk_size  = self.params.getint('data', 'chunk_size')
            self.params.set('data', 'chunk_size', str(chunk_size*self.rate))            
            chunk_size = self.params.getint('whitening', 'chunk_size')
            self.params.set('whitening', 'chunk_size', str(chunk_size*self.rate))

            try:
                N_t             = self.params.getfloat('detection', 'N_t')
            except Exception:
                N_t             = self.params.getfloat('data', 'N_t')

            N_t = int(self.rate*N_t*1e-3)

            for key in ['whitening', 'clustering']:
                safety_time = self.params.get(key, 'safety_time')
                if safety_time == 'auto':
                    self.params.set(key, 'safety_time', '%g' %(N_t//3.))

            if numpy.mod(N_t, 2) == 0:
                N_t += 1
            self.params.set('detection', 'N_t', str(N_t))
            self.params.set('detection', 'template_shift', str(int((N_t-1)//2)))
            self.params.set('detection', 'dist_peaks', str(N_t)) # Get only isolated spikes for a single electrode (whitening, clustering, basis) 


    def _get_info_(self):
        '''
            This function is called only if the file is not empty, and should fill the values in the constructor
            such as max_offset, _shape, ...
        '''
        pass


    def _get_chunk_size_(self, chunk_size=None):
        '''
            This function returns a default size for the data chunks
        '''
        if chunk_size is None:
            chunk_size = self.params.getint('data', 'chunk_size')
        
        return chunk_size     


    def get_data(self, idx, chunk_size=None, padding=(0, 0), nodes=None):
        '''
        Assuming the analyze function has been called before, this is the main function
        used by the code, in all steps, to get data chunks. More precisely, assuming your
        dataset can be divided in nb_chunks (see analyze) of temporal size (chunk_size), 

            - idx is the index of the chunk you want to load
            - chunk_size is the time of those chunks, in time steps
            - if the data loaded are data[idx:idx+1], padding should add some offsets, 
                in time steps, such that we can load data[idx+padding[0]:idx+padding[1]]
            - nodes is a list of nodes, between 0 and N_total            
        '''

        pass

    def get_snippet(self, time, length, nodes=None):
        '''
            This function should return a time snippet of size length x nodes
            - time is in timestep
            - length is in timestep
            - nodes is a list of nodes, between 0 and N_total
        '''
        return self.get_data(0, chunk_size=length, padding=(time, time), nodes=nodes)


    def set_data(self, time, data):
        '''
            This function writes data at a given time.
            - time is expressed in timestep
            - data must be a 2D matrix of size time_length x N_total
        '''
        pass


    def analyze(self, chunk_size=None):
        '''
            This function should return two values: 
            - the number of temporal chunks of temporal size chunk_size that can be found 
            in the data. Note that even if the last chunk is not complete, it has to be 
            counted. chunk_size is expressed in time steps
            - the length of the last uncomplete chunk, in time steps
        '''
        chunk_size     = self._get_chunk_size_(chunk_size)
        nb_chunks      = numpy.int64(self.shape[0]) // chunk_size
        last_chunk_len = numpy.int64(self.shape[0]) - nb_chunks * chunk_size

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
                - shape is a tuple with (time lenght, N_total)
                - data_dtype is the data type
        '''
        pass


    @property
    def shape(self):
        return self._shape   


    @property
    def is_master(self):
    	if self.comm == None:
            return True
    	else:
            return self.comm.rank == 0