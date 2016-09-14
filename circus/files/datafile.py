import h5py, numpy, re
import ConfigParser as configparser

class DataFile(object):

    '''
    A generic class that will represent how the program interacts with the data. Such an abstraction
    layer should allow people to write their own wrappers, for several file format. Note that 
    depending on the complexity of the datastructure, this can slow down the code.

    The method belows are all methods can be used, at some point, by the different steps of the code. 
    In order to provide a full compatibility with a given file format, they must all be implemented

    '''

    def __init__(self, file_name, params, empty=False):
        self.file_name = file_name
        assert isinstance(params, configparser.ConfigParser)
        self.params = params
        self.N_e    = params.getint('data', 'N_e')
        self.N_tot  = params.getint('data', 'N_total')
        self.rate   = params.getint('data', 'sampling_rate')
        self.template_shift = params.getint('data', 'template_shift')
        self.max_offset  = 0
        self.empty = empty
        self._parrallel_write = False

    def get_data(self, idx, chunk_len, chunk_size=None, padding=(0, 0), nodes=None):
        pass

    def get_snippet(self, time, length, nodes=None):
        pass

    def set_data(self, time, data):
        pass

    def analyze(self):
        pass

    def prepare_preview(self):
    	pass

    def open(self, mode):
        pass

    def copy_header(self, file_out):
        pass

    def close(self):
        pass

    def allocate(self, shape, data_dtype):
        pass

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
                self.dtype_offset = self.params.getint('data', 'dtype_offset')
            except Exception:
                print "Offset not valid"


class RawBinaryFile(DataFile):

    def __init__(self, file_name, params, empty=False):
        DataFile.__init__(self, file_name, params, empty)
        try:
            self.data_offset = self.params.getint('data', 'data_offset')
        except Exception:
            self.data_offset = 0
        self.data_dtype  = self.params.get('data', 'data_dtype')
        self.set_dtype_offset(self.data_dtype)
        self._parrallel_write = True
        if not self.empty:
            self._get_info_()    

    def _get_info_(self):
        self.empty = False
        self.open()
        self.N    = len(self.data)
        self.size = (self.N//self.N_tot, self.N_tot)
        self.max_offset = self.size[0] 
        self.close()

    def allocate(self, shape, data_dtype=None):
        if data_dtype is None:
            data_dtype = self.data_dtype
        self.data = numpy.memmap(self.file_name, offset=self.data_offset, dtype=self.data_dtype, mode='w+', shape=shape)
        self._get_info_()
        del self.data

    def get_data(self, idx, chunk_len, chunk_size=None, padding=(0, 0), nodes=None):
    	
        if chunk_size is None:
            chunk_size = self.params.getint('data', 'chunk_size') * self.N_tot

        self.open()
        local_chunk  = self.data[idx*numpy.int64(chunk_len)+padding[0]*self.N_tot:(idx+1)*numpy.int64(chunk_len)+padding[1]*self.N_tot]
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
        return local_chunk

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
        return borders, nb_chunks, chunk_len, last_chunk_len

    def copy_header(self, file_out):
        fin  = open(self.file_name, 'rb')
        fout = open(file_out, 'wb')
        data = fin.read(self.data_offset)
        fout.write(data)
        fin.close()
        fout.close()

    def prepare_preview(self, preview_filename):
        chunk_size   = 2*self.rate
        chunk_len    = numpy.int64(self.N_tot) * chunk_size
	    
        self.open()
        local_chunk  = self.data[0:chunk_len]
        self.close()

        output = open(preview_filename, 'wb')
        fid    = open(self.file_name, 'rb')
        for i in xrange(self.data_offset):
            output.write(fid.read(1))
        fid.close()

	    #Then the datafile
        local_chunk.tofile(output)
        output.close()

    def open(self, mode='r'):
        self.data = numpy.memmap(self.file_name, offset=self.data_offset, dtype=self.data_dtype, mode=mode)

    def close(self):
        self.data = None

		

class RawMCSFile(RawBinaryFile):

    def __init__(self, file_name, params, empty=False):
        RawBinaryFile.__init__(self, file_name, params, empty=True)
        a, b = self.detect_header()
        self.data_offset = a
        self.nb_channels = b

        if self.nb_channels != self.N_tot:
            print_and_log(["MCS file: mismatch between number of electrodes and data header"], 'error', params, show)
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
	        #    print_error(['Wrong MCS header: file is not exported with MCRack'])
                sys.exit(0) 
            else:
                header += 2
            return header, len(regexp.findall(header_text))
        except Exception:
	        #print_error(["Wrong MCS header: file is not exported with MCRack"])
            sys.exit(0)




class H5File(DataFile):

    def __init__(self, file_name, params, empty):
        DataFile.__init__(self, file_name, params, empty)
        self.h5_key = self.params.get('data', 'hdf5_key_data')
        self._parrallel_write = h5py.get_config().mpi
        if not self.empty:
            self._get_info_()

    def _get_info_(self):
        self.empty = False
        self.open()
        self.data_dtype = self.my_file.get(self.h5_key).dtype
        self.set_dtype_offset(self.data_dtype)
        self.size      = self.my_file.get(self.h5_key).shape
        
        assert (self.size[0] == self.N_tot) or (self.size[1] == self.N_tot)
        if self.size[0] == self.N_tot:
            self.time_axis = 1
        else:
            self.time_axis = 0
        self.max_offset = self.size[self.time_axis]
        self.data_offset = 0
        self.close()

    def allocate(self, shape, data_dtype=None):
        if data_dtype is None:
            data_dtype = self.data_dtype

        self.my_file = h5py.File(self.file_name, mode='w')
        self.my_file.create_dataset(self.h5_key, dtype=data_dtype, shape=shape, chunks=True)
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
        local_shape  = len(local_chunk)

        if nodes is not None:
            if not numpy.all(nodes == numpy.arange(self.N_tot)):
                local_chunk = numpy.take(local_chunk, nodes, axis=1)

        return numpy.ascontiguousarray(local_chunk), local_shape

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
        return local_chunk


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
	    
	    chunk_len      = chunk_size
	    borders        = self.template_shift
	    nb_chunks      = numpy.int64(self.size[self.time_axis]) // chunk_len
	    last_chunk_len = self.size[self.time_axis] - nb_chunks * chunk_len
	    
	    return borders, nb_chunks, chunk_len, last_chunk_len

    def prepare_preview(self, preview_filename):
        chunk_size   = 2*self.rate
        if self.time_axis == 0:
            local_chunk = self.data[:2*self.rate, :]
        elif self.time_axis == 1:
            local_chunk = self.data[:, :2*self.rate]

        output = h5py.File(preview_filename, mode='w')
        output.create_dataset(self.h5_key, data=local_chunk)
        output.close()

    def open(self, mode='r'):
        self.my_file = h5py.File(self.file_name, mode=mode)
        self.data = self.my_file.get(self.h5_key)
        

    def close(self):
        self.my_file.close()

