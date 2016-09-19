import h5py, numpy, re, sys, os
import ConfigParser as configparser

from circus.shared.messages import print_error, print_and_log

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

    def __init__(self, file_name, params, empty=False, comm=None, **kwargs):
        '''
        The constructor that will create the DataFile object. Note that by default, values are read from
        the parameter file, but you could completly fill them based on values that would be obtained
        from the datafile itself. 
        What you need to specify
            - _parallel_write : can the file be safely written in parallel ?
            - max_offset : the time length of the data, in time steps
            - _shape : the size of the data, should be a tuple (max_offset, N_tot)
            - comm is a MPI communicator ring, if the file is created in a MPI environment

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
                self.__setattr__(key, self.params.getint(value[0], value[1]))

        self.max_offset  = 0
        self._shape      = None
        print_and_log(["The datafile %s with type %s has been created" %(self.file_name, self._description)], 'debug', self.params)

        if not self.empty:
            self._get_info_()

    def _get_info_(self):
        '''
            This function is called only if the file is not empty, and should fill the values in the constructor
            such as max_offset, _shape, ...
        '''
        pass

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
        if chunk_size is None:
            chunk_size = self.params.getint('data', 'chunk_size')

        nb_chunks      = numpy.int64(self.shape[0]) // chunk_size
        last_chunk_len = self.shape[0] - nb_chunks * chunk_size

        if last_chunk_len > 0:
            nb_chunks += 1

        return nb_chunks, last_chunk_len


    def open(self, mode):
        ''' 
            This function should open the file
            - mode can be to read only 'r', or to write 'w'
        '''
        pass

    def copy_header(self, file_out):
        '''
            If the data has some header that need to be copied when a new file of the same
            time is created. Note that in the code, this is always followed by a all to the
            allocate() function for that new file
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