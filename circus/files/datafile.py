import h5py, numpy, re, sys
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

    def __init__(self, file_name, params, empty=False, comm=None):
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
        assert isinstance(params, configparser.ConfigParser)
        self.params = params
        self.N_e    = params.getint('data', 'N_e')
        self.N_tot  = params.getint('data', 'N_total')
        self.rate   = params.getint('data', 'sampling_rate')
        self.template_shift = params.getint('data', 'template_shift')
        self.max_offset  = 0
        self.empty = empty
        self._shape = None
        self.comm = comm
        print_and_log(["The datafile %s with type %s has been created" %(self.file_name, self._description)], 'debug', self.params)


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
        pass

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

        pass


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

    def set_dtype_offset(self, data_dtype):
        self.dtype_offset = self.params.get('data', 'dtype_offset')
        if self.dtype_offset == 'auto':
            if self.data_dtype == 'uint16':
                self.dtype_offset = 32767
            elif self.data_dtype == 'int16':
                self.dtype_offset = 0
            elif self.data_dtype == 'float32':
                self.dtype_offset = 0
            elif self.data_dtype == 'int8':
                self.dtype_offset = 0        
            elif self.data_dtype == 'uint8':
                self.dtype_offset = 127
        else:
            try:
                self.dtype_offset = int(self.dtype_offset)
            except Exception:
                print_error(["Offset %s is not valid" %self.dtype_offset])
                sys.exit(0)


class RawBinaryFile(DataFile):

    _description = "raw_binary"    
    _parrallel_write = True

    def __init__(self, file_name, params, empty=False, comm=None):
        DataFile.__init__(self, file_name, params, empty, comm)
        try:
            self.data_offset = self.params.getint('data', 'data_offset')
        except Exception:
            self.data_offset = 0
        self.data_dtype  = self.params.get('data', 'data_dtype')
        self.set_dtype_offset(self.data_dtype)
        if not self.empty:
            self._get_info_()    

    def _get_info_(self):
        self.empty = False
        self.open()
        self.N      = len(self.data)
        self._shape = (self.N//self.N_tot, self.N_tot)
        self.max_offset = self._shape[0] 
        self.close()

    def allocate(self, shape, data_dtype=None):
        if data_dtype is None:
            data_dtype = self.data_dtype
        self.data = numpy.memmap(self.file_name, offset=self.data_offset, dtype=self.data_dtype, mode='w+', shape=shape)
        self._get_info_()
        del self.data

    def get_data(self, idx, chunk_size=None, padding=(0, 0), nodes=None):
    	
        if chunk_size is None:
            chunk_size = self.params.getint('data', 'chunk_size')

        chunk_len    = chunk_size * self.N_tot 
        padding      = numpy.array(padding) * self.N_tot

        self.open()
        local_chunk  = self.data[idx*numpy.int64(chunk_len)+padding[0]:(idx+1)*numpy.int64(chunk_len)+padding[1]]
        local_shape  = len(local_chunk)//self.N_tot
        local_chunk  = local_chunk.reshape(local_shape, self.N_tot)
        local_chunk  = local_chunk.astype(numpy.float32)
        local_chunk -= self.dtype_offset
        self.close()

        if nodes is not None:
            if not numpy.all(nodes == numpy.arange(self.N_tot)):
                local_chunk = numpy.take(local_chunk, nodes, axis=1)

        return numpy.ascontiguousarray(local_chunk), local_shape


    def get_snippet(self, time, length, nodes=None):
        
        self.open()
        local_chunk  = self.data[time*self.N_tot:time*self.N_tot + length*self.N_tot]
        local_chunk  = local_chunk.reshape(length, self.N_tot)
        local_chunk  = local_chunk.astype(numpy.float32)
        local_chunk -= self.dtype_offset
        self.close()
        
        if nodes is not None:
            if not numpy.all(nodes == numpy.arange(self.N_tot)):
                local_chunk = numpy.take(local_chunk, nodes, axis=1)

        return numpy.ascontiguousarray(local_chunk)

    def set_data(self, time, data):
        self.open(mode='r+')
        data  += self.dtype_offset
        data   = data.astype(self.data_dtype)
        data   = data.ravel()
        self.data[self.N_tot*time:self.N_tot*time+len(data)] = data
        self.close()

    def analyze(self, chunk_size=None):

        if chunk_size is None:
            chunk_size = self.params.getint('data', 'chunk_size')
	    
        chunk_len      = numpy.int64(self.N_tot) * chunk_size
        borders        = self.template_shift
        nb_chunks      = numpy.int64(self.N) // chunk_len
        last_chunk_len = self.N - (nb_chunks * chunk_len)
        last_chunk_len = last_chunk_len//self.N_tot
        
        if last_chunk_len > 0:
            nb_chunks += 1

        return nb_chunks, last_chunk_len

    def copy_header(self, file_out):
        fin  = open(self.file_name, 'rb')
        fout = open(file_out, 'wb')
        data = fin.read(self.data_offset)
        fout.write(data)
        fin.close()
        fout.close()

    def open(self, mode='r'):
        self.data = numpy.memmap(self.file_name, offset=self.data_offset, dtype=self.data_dtype, mode=mode)

    def close(self):
        self.data = None

		

class RawMCSFile(RawBinaryFile):

    _description = "mcs_raw_binary"
    _parrallel_write = True

    def __init__(self, file_name, params, empty=False, comm=None):
        RawBinaryFile.__init__(self, file_name, params, empty, comm)
        a, b = self.detect_header()
        self.data_offset = a
        self.nb_channels = b

        if self.nb_channels != self.N_tot:
            print_and_log(["MCS file: mismatch between number of electrodes and data header"], 'error', params)
        self.empty = empty
        if not self.empty:
            self._get_info_()


    def detect_header(self):
        try:
            header      = 0
            stop        = False
            fid         = open(self.file_name, 'rb')
            header_text = ''
            regexp      = re.compile('El_\d*')

            while ((stop is False) and (header <= 5000)):
                header      += 1
                char         = fid.read(1)
                header_text += char.decode('Windows-1252')
                if (header > 2):
                    if (header_text[header-3:header] == 'EOH'):
                        stop = True
            fid.close()
            if stop is False:
                print_error(['Wrong MCS header: file is not exported with MCRack'])
                sys.exit(0) 
            else:
                header += 2
            return header, len(regexp.findall(header_text))
        except Exception:
            print_error(["Wrong MCS header: file is not exported with MCRack"])
            sys.exit(0)




class H5File(DataFile):

    _description = "hdf5"    
    _parrallel_write = h5py.get_config().mpi

    def __init__(self, file_name, params, empty=False, comm=None):

        DataFile.__init__(self, file_name, params, empty, comm)
        self.h5_key      = self.params.get('data', 'hdf5_key_data')
        self.compression = ''
        if not self.empty:
            self._get_info_()

    def _get_info_(self):
        self.empty = False
        self.open()
        self.data_dtype  = self.my_file.get(self.h5_key).dtype
        self.compression = self.my_file.get(self.h5_key).compression

        # HDF5 does not support parallel writes with compression
        if self.compression != '':
        	self._parrallel_write = False
        
        self.size        = self.my_file.get(self.h5_key).shape
        self.set_dtype_offset(self.data_dtype)
        
        assert (self.size[0] == self.N_tot) or (self.size[1] == self.N_tot)
        if self.size[0] == self.N_tot:
            self.time_axis = 1
            self._shape = (self.size[1], self.size[0])
        else:
            self.time_axis = 0
            self._shape = self.size

        self.max_offset = self._shape[0]
        self.data_offset = 0
        self.close()

    def allocate(self, shape, data_dtype=None):
        if data_dtype is None:
            data_dtype = self.data_dtype

        if self._parrallel_write and (self.comm is not None):
            self.my_file = h5py.File(self.file_name, 'w', driver='mpio', comm=self.comm)
        else:
            self.my_file = h5py.File(self.file_name, mode='w')
    	
    	self.my_file.create_dataset(self.h5_key, dtype=data_dtype, shape=shape, compression=self.compression, chunks=True)
        self.my_file.close()
        self._get_info_()

    def get_data(self, idx, chunk_len, chunk_size=None, padding=(0, 0), nodes=None):

        if chunk_size is None:
            chunk_size = self.params.getint('data', 'chunk_size')

        if self.time_axis == 0:
            local_chunk = self.data[idx*numpy.int64(chunk_len)+padding[0]:(idx+1)*numpy.int64(chunk_len)+padding[1], :]
        elif self.time_axis == 1:
            local_chunk = self.data[:, idx*numpy.int64(chunk_len)+padding[0]:(idx+1)*numpy.int64(chunk_len)+padding[1]].T

        local_chunk  = local_chunk.astype(numpy.float32)
        local_chunk -= self.dtype_offset

        if nodes is not None:
            if not numpy.all(nodes == numpy.arange(self.N_tot)):
                local_chunk = numpy.take(local_chunk, nodes, axis=1)

        return numpy.ascontiguousarray(local_chunk), len(local_chunk)

    def get_snippet(self, time, length, nodes=None):

        if self.time_axis == 0:
            local_chunk = self.data[time:time+length, :]
        elif self.time_axis == 1:
            local_chunk = self.data[:, time:time+length].T

        local_chunk  = local_chunk.astype(numpy.float32)
        local_chunk -= self.dtype_offset

        if nodes is not None:
            if not numpy.all(nodes == numpy.arange(self.N_tot)):
                local_chunk = numpy.take(local_chunk, nodes, axis=1)
        
        return numpy.ascontiguousarray(local_chunk)


    def set_data(self, time, data):
        
    	data += self.dtype_offset
    	data  = data.astype(self.data_dtype)

        if self.time_axis == 0:
            local_chunk = self.data[time:time+data.shape[0], :] = data
        elif self.time_axis == 1:
            local_chunk = self.data[:, time:time+data.shape[0]] = data.T

    def analyze(self, chunk_size=None):

        if chunk_size is None:
            chunk_size = self.params.getint('data', 'chunk_size')
	    
        nb_chunks      = numpy.int64(self.shape[0]) // chunk_size
        last_chunk_len = self.shape[0] - nb_chunks * chunk_size

        if last_chunk_len > 0:
            nb_chunks += 1
        return nb_chunks, last_chunk_len

    def open(self, mode='r'):
        if self._parrallel_write and (self.comm is not None):
            self.my_file = h5py.File(self.file_name, mode=mode, driver='mpio', comm=self.comm)
        else:
            self.my_file = h5py.File(self.file_name, mode=mode)

        self.data = self.my_file.get(self.h5_key)
        
    def close(self):
        self.my_file.close()

