import h5py, numpy, re, sys, os
import ConfigParser as configparser
from circus.shared.messages import print_error, print_and_log
from datafile import DataFile

class OpenEphysFile(DataFile):

    _description    = "openephys"    
    _extension      = [".openephys"]
    _parallel_write = True
    _is_writable    = True

    # constants
    NUM_HEADER_BYTES   = 1024L
    SAMPLES_PER_RECORD = 1024L
    RECORD_SIZE        = 8 + 2*2 + SAMPLES_PER_RECORD*2 + 10 # size of each continuous record in bytes
    OFFSET_PER_BLOCK   = ((8 + 2*2)/2, 10/2)

    def _get_sorted_channels_(self, folderpath):
        return sorted([int(f.split('_CH')[1].split('.')[0]) for f in os.listdir(folderpath) 
                    if '.continuous' in f and '_CH' in f]) 

    def _read_header_(self, file):
        header = { }
        f = open(file, 'rb')
        h = f.read(self.NUM_HEADER_BYTES).replace('\n','').replace('header.','')
        for i,item in enumerate(h.split(';')):
            if '=' in item:
                header[item.split(' = ')[0]] = item.split(' = ')[1]
        f.close()
        return header

    def __init__(self, file_name, params, empty=False, comm=None):

        kwargs = {}
        kwargs['data_dtype']   = 'float32'
        kwargs['dtype_offset'] = 0
        kwargs['data_offset']  = self.NUM_HEADER_BYTES

        if not empty:
            folder_path     = os.path.dirname(os.path.realpath(file_name))
            self.all_channels = self._get_sorted_channels_(folder_path)
            self.all_files  = [os.path.join(folder_path, '100_CH' + x + '.continuous') for x in map(str,self.all_channels)]
            self.header     = self._read_header_(self.all_files[0])
            kwargs['rate']  = float(self.header['sampleRate'])        
            kwargs['N_tot'] = len(self.all_files)
            kwargs['gain']  = float(self.header['bitVolts'])        

        DataFile.__init__(self, file_name, params, empty, comm, **kwargs)


    def _get_info_(self):
        self.empty       = False
        self.open()
        g = open(self.all_files[0], 'rb')
        self.size        = ((os.fstat(g.fileno()).st_size - self.NUM_HEADER_BYTES)//self.RECORD_SIZE) * self.SAMPLES_PER_RECORD
        g.close()
        self._shape      = (self.size, self.N_tot)
        self.max_offset  = self._shape[0]
        self.close()

    def allocate(self, shape, data_dtype=None):
        pass

    def copy_header(self, file_out):
        if self.is_master:
            fin  = open(self.file_name, 'rb')
            fout = open(file_out, 'wb')
            data = fin.read(self.data_offset)
            fout.write(data)
            fin.close()
            fout.close()

    def _get_slice_(self, t_start, t_stop):

        x_beg = numpy.int64(t_start // self.SAMPLES_PER_RECORD)
        r_beg = numpy.mod(t_start, self.SAMPLES_PER_RECORD)
        x_end = numpy.int64(t_stop // self.SAMPLES_PER_RECORD)
        r_end = numpy.mod(t_stop, self.SAMPLES_PER_RECORD)

        data_slice  = []
        
        if x_beg == x_end:
            g_offset = x_beg * self.SAMPLES_PER_RECORD + self.OFFSET_PER_BLOCK[0]*(x_beg + 1) + self.OFFSET_PER_BLOCK[1]*x_beg
            data_slice = numpy.arange(g_offset + r_beg, g_offset + r_end)
        else:
            for count, nb_blocks in enumerate(numpy.arange(x_beg, x_end + 1)):
                g_offset = nb_blocks * self.SAMPLES_PER_RECORD + self.OFFSET_PER_BLOCK[0]*(nb_blocks + 1) + self.OFFSET_PER_BLOCK[1]*nb_blocks
                if count == 0:
                    data_slice += numpy.arange(g_offset + r_beg, g_offset + self.SAMPLES_PER_RECORD).tolist()
                elif (count == (x_end - x_beg)):
                    data_slice += numpy.arange(g_offset, g_offset + r_end).tolist()
                else:
                    data_slice += numpy.arange(g_offset, g_offset + self.SAMPLES_PER_RECORD).tolist()

        return data_slice 


    def get_data(self, idx, chunk_size=None, padding=(0, 0), nodes=None):
        
        chunk_size  = self._get_chunk_size_(chunk_size)
        t_start     = idx*numpy.int64(chunk_size)+padding[0]
        t_stop      = (idx+1)*numpy.int64(chunk_size)+padding[1]
        local_shape = t_stop - t_start

        if (t_start + local_shape) > self.max_offset:
            local_shape = self.max_offset - t_start
            t_stop      = self.max_offset

        if nodes is None:
            nodes = numpy.arange(self.N_tot)

        local_chunk = numpy.zeros((local_shape, len(nodes)), dtype='>i2')
        data_slice  = self._get_slice_(t_start, t_stop) 

        self.open()
        for count, i in enumerate(nodes):
            local_chunk[:, count] = self.data[i][data_slice]
        self.close()

        local_chunk  = local_chunk.astype(numpy.float32)
        local_chunk *= self.gain

        return numpy.ascontiguousarray(local_chunk)


    def set_data(self, time, data):

        t_start     = time
        t_stop      = time + data.shape[0]
        local_shape = t_stop - t_start

        if (t_start + local_shape) > self.max_offset:
            local_shape = self.max_offset - t_start
            t_stop      = self.max_offset

        data_slice  = self._get_slice_(t_start, t_stop) 

        data /= self.gain
        data  = data.astype('>i2')
        
        self.open(mode='r+')
        for i in xrange(self.N_tot):
            self.data[i][data_slice] = data[:, i]
        self.close()

    def open(self, mode='r'):
        self.data = [numpy.memmap(self.all_files[i], offset=self.data_offset, dtype='>i2', mode=mode) for i in xrange(self.N_tot)]
        
    def close(self):
        self.data = None
