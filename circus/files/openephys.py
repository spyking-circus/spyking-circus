import h5py, numpy, re, sys, os
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

    def __init__(self, file_name, is_empty=False, **kwargs):

        kwargs['data_dtype']   = 'int16'
        kwargs['dtype_offset'] = 0
        kwargs['data_offset']  = self.NUM_HEADER_BYTES

        if not is_empty:
            folder_path     = os.path.dirname(os.path.realpath(file_name))
            self.all_channels = self._get_sorted_channels_(folder_path)
            self.all_files  = [os.path.join(folder_path, '100_CH' + x + '.continuous') for x in map(str,self.all_channels)]
            self.header     = self._read_header_(self.all_files[0])
            kwargs['sampling_rate'] = float(self.header['sampleRate'])        
            kwargs['nb_channels']   = len(self.all_files)
            kwargs['gain']          = float(self.header['bitVolts'])        

        DataFile.__init__(self, file_name, is_empty, **kwargs)


    def _get_info_(self):
        self.open()
        g = open(self.all_files[0], 'rb')
        self.size        = ((os.fstat(g.fileno()).st_size - self.NUM_HEADER_BYTES)//self.RECORD_SIZE) * self.SAMPLES_PER_RECORD
        g.close()
        self._shape      = (self.size, self.nb_channels)
        self.close()

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


    def get_data(self, idx, chunk_size, padding=(0, 0), nodes=None):
        
        t_start     = idx*numpy.int64(chunk_size)+padding[0]
        t_stop      = (idx+1)*numpy.int64(chunk_size)+padding[1]
        local_shape = t_stop - t_start

        if (t_start + local_shape) > self.duration:
            local_shape = self.duration - t_start
            t_stop      = self.duration

        if nodes is None:
            nodes = numpy.arange(self.nb_channels)

        local_chunk = numpy.zeros((local_shape, len(nodes)), dtype=self.data_dtype)
        data_slice  = self._get_slice_(t_start, t_stop) 

        self.open()
        for count, i in enumerate(nodes):
            local_chunk[:, count] = self.data[i][data_slice]
        self.close()

        return self._scale_data_to_float32(local_chunk)


    def set_data(self, time, data):

        t_start     = time
        t_stop      = time + data.shape[0]
        local_shape = t_stop - t_start

        if (t_start + local_shape) > self.duration:
            local_shape = self.duration - t_start
            t_stop      = self.duration

        data_slice  = self._get_slice_(t_start, t_stop) 
        
        self.open(mode='r+')
        for i in xrange(self.nb_channels):
            self.data[i][data_slice] = self._unscale_data_from_from32(data)[:, i]
        self.close()

    def open(self, mode='r'):
        self.data = [numpy.memmap(self.all_files[i], offset=self.data_offset, dtype=self.data_dtype, mode=mode) for i in xrange(self.nb_channels)]
        
    def close(self):
        self.data = None
