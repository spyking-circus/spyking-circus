import h5py, numpy, re, sys, os
from datafile import DataFile
import xml.etree.ElementTree as ET

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text) ]

class OpenEphysFile(DataFile):

    description    = "openephys"    
    extension      = [".openephys"]
    parallel_write = True
    is_writable    = True

    # constants
    NUM_HEADER_BYTES   = 1024L
    SAMPLES_PER_RECORD = 1024L
    RECORD_SIZE        = 8 + 2*2 + SAMPLES_PER_RECORD*2 + 10 # size of each continuous record in bytes
    OFFSET_PER_BLOCK   = ((8 + 2*2)/2, 10/2)

    _params            = {'data_dtype'   : '>i2',
                          'dtype_offset' : 0,
                          'data_offset'  : NUM_HEADER_BYTES}


    def _get_sorted_channels_(self):
        tree = ET.parse(self.file_name)
        root = tree.getroot()
        # find all channels with gain matching intan's non-aux channel gain
        chans = root.findall("./RECORDING/PROCESSOR/CHANNEL[@bitVolts]")
        lfp_chans = [x for x in chans if x.attrib['bitVolts'].startswith('0.1949')]
        # return list of channel file names
        alist = [x.attrib['filename'] for x in lfp_chans]
        alist.sort(key=natural_keys)
        return alist

    def _read_header_(self, file):
        header = { }
        f = open(file, 'rb')
        h = f.read(self.NUM_HEADER_BYTES).replace('\n','').replace('header.','')
        for i,item in enumerate(h.split(';')):
            if '=' in item:
                header[item.split(' = ')[0]] = item.split(' = ')[1]
        f.close()
        return header


    def _read_from_header(self):

        folder_path     = os.path.dirname(os.path.realpath(self.file_name))
        self.all_channels = self._get_sorted_channels_()
        self.all_files  = [os.path.join(folder_path, x) for x in self.all_channels]
        self.header     = self._read_header_(self.all_files[0])
        
        header                  = {}
        header['sampling_rate'] = float(self.header['sampleRate'])        
        header['nb_channels']   = len(self.all_files)
        header['gain']          = float(self.header['bitVolts'])        

        g = open(self.all_files[0], 'rb')
        self.size        = ((os.fstat(g.fileno()).st_size - self.NUM_HEADER_BYTES)//self.RECORD_SIZE - 1) * self.SAMPLES_PER_RECORD
        self._shape      = (self.size, header['nb_channels'])
        g.close()
        
        return header


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


    def read_chunk(self, idx, chunk_size, padding=(0, 0), nodes=None):
        
        t_start, t_stop = self._get_t_start_t_stop(idx, chunk_size, padding)
        local_shape     = t_stop - t_start

        if nodes is None:
            nodes = numpy.arange(self.nb_channels)

        local_chunk = numpy.zeros((local_shape, len(nodes)), dtype=self.data_dtype)
        data_slice  = self._get_slice_(t_start, t_stop) 

        self._open()
        for count, i in enumerate(nodes):
            local_chunk[:, count] = self.data[i][data_slice]
        self._close()

        return self._scale_data_to_float32(local_chunk)


    def write_chunk(self, time, data):

        t_start     = time
        t_stop      = time + data.shape[0]

        if t_stop > self.duration:
            t_stop  = self.duration

        data_slice  = self._get_slice_(t_start, t_stop) 
        
        self._open(mode='r+')
        for i in xrange(self.nb_channels):
            self.data[i][data_slice] = self._unscale_data_from_from32(data)[:, i]
        self._close()

    def _open(self, mode='r'):
        self.data = [numpy.memmap(self.all_files[i], offset=self.data_offset, dtype=self.data_dtype, mode=mode) for i in xrange(self.nb_channels)]
        
    def _close(self):
        self.data = None
