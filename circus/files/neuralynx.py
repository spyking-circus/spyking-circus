import numpy
import re
import sys
import os
import datetime
import warnings
import logging
from .datafile import DataFile
from circus.shared.messages import print_and_log


logger = logging.getLogger(__name__)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split('(\d+)', text)]


def filter_per_extension(files, extension):
    results = []
    for file in files:
        fn, ext = os.path.splitext(file)
        if ext == extension:
            results += [file]
    return results


def filter_name_duplicates(tmp_all_files, ncs_pattern=''):
    all_files = []
    for file in tmp_all_files:

        all_parts = file.split('_')
        name = '_'.join(all_parts[:-1])

        if ncs_pattern != '':
            pattern = name.find(ncs_pattern) > 0
        else:
            pattern = True

        if not pattern:
            to_consider = False
        else:
            to_consider = True

        for f in all_files:
            already_present = '_'.join(f.split('_')[:-1])
            if name == already_present:
                to_consider = False

        if to_consider:
            all_files += ['_'.join([name, all_parts[-1]])]

    return all_files


class NeuraLynxFile(DataFile):

    description = "neuralynx"
    extension = [".ncs"]
    parallel_write = True
    is_writable = True
    is_streamable = ['multi-files', 'multi-folders']

    # constants
    NUM_HEADER_BYTES = 16 * 1024  # 16 kilobytes of header
    SAMPLES_PER_RECORD = 512
    RECORD_SIZE = 8 + 4 + 4 + 4 + SAMPLES_PER_RECORD * 2  # size of each continuous record in bytes
    OFFSET_PER_BLOCK = ((8 + 4 + 4 + 4) / 2, 0)

    _params = {
        'data_dtype': 'int16',
        'dtype_offset': 0,
        'data_offset': NUM_HEADER_BYTES
    }

    _default_values = {
        'ncs_pattern': ''
    }

    def set_streams(self, stream_mode):

        # We assume that all names are in the forms XXXX_channel.ncs

        if stream_mode == 'multi-files':
            dirname = os.path.abspath(os.path.dirname(self.file_name))
            fname = os.path.basename(self.file_name)
            fn, ext = os.path.splitext(fname)
            tmp_all_files = os.listdir(dirname)
            tmp_all_files = filter_per_extension(tmp_all_files, ext)
            tmp_all_files.sort(key=natural_keys)

            all_files = filter_name_duplicates(tmp_all_files, self.params['ncs_pattern'])

            sources = []
            to_write = []
            global_time = 0
            params = self.get_description()

            for fname in all_files:
                params['ncs_pattern'] = '_'.join(fname.split('_')[:-1])
                new_data = type(self)(os.path.join(os.path.abspath(dirname), fname), params)
                new_data._t_start = global_time
                global_time += new_data.duration
                sources += [new_data]
                to_write += [
                    "We found the datafile %s with t_start %s and duration %s"
                    % (new_data.file_name, new_data.t_start, new_data.duration)
                ]

            print_and_log(to_write, 'debug', logger)
            return sources

        elif stream_mode == 'multi-folders':
            dirname = os.path.abspath(os.path.dirname(self.file_name))
            upper_dir = os.path.dirname(dirname)
            fname = os.path.basename(self.file_name)

            all_directories = os.listdir(upper_dir)
            all_files = []

            for local_dir in all_directories:
                local_dir = os.path.join(upper_dir, local_dir)
                if os.path.isdir(local_dir):
                    all_local_files = os.listdir(local_dir)
                    for local_file in all_local_files:
                        ncs_file = os.path.join(upper_dir, local_dir, local_file)
                        is_valid = len(re.findall(".*_%s_1.ncs" % self.params['ncs_pattern'], ncs_file)) > 0
                        if is_valid and ncs_file not in all_files:
                            all_files += [ncs_file]

            all_files.sort(key=natural_keys)

            sources = []
            to_write = []
            global_time = 0
            params = self.get_description()

            for fname in all_files:
                params['ncs_pattern'] = self.params['ncs_pattern']
                new_data = type(self)(os.path.join(os.path.abspath(dirname), fname), params)
                new_data._t_start = global_time
                global_time += new_data.duration
                sources += [new_data]
                to_write += ['We found the datafile %s with t_start %s and duration %s' % (new_data.file_name, new_data.t_start, new_data.duration)]

            print_and_log(to_write, 'debug', logger)
            return sources

    def parse_neuralynx_time_string(self, time_string):
        # Parse a datetime object from the idiosyncratic time string in Neuralynx file headers
        try:
            tmp_date = [int(x) for x in time_string.split()[4].split('/')]
            tmp_time = [int(x) for x in time_string.split()[-1].replace('.', ':').split(':')]
            tmp_microsecond = tmp_time[3] * 1000
        except:
            print_and_log(['Unable to parse time string from Neuralynx header: ' + time_string], 'debug', logger)
            return None
        else:
            return datetime.datetime(tmp_date[2], tmp_date[0], tmp_date[1],  # Year, month, day
                                     tmp_time[0], tmp_time[1], tmp_time[2],  # Hour, minute, second
                                     tmp_microsecond)

    def _get_sorted_channels_(self):
        
        directory = os.path.dirname(self.file_name)
        all_files = os.listdir(directory)
        alist = []

        for f in all_files:
            if self.params['ncs_pattern'] != '':
                test = f.find('.ncs') > -1 and f.find(self.params['ncs_pattern']) > -1
            else:
                test = f.find('.ncs') > -1
            if test:
                alist += [os.path.join(directory, f)]
        alist.sort(key=natural_keys)
        return alist

    def _read_header_(self, file):
        header = {}

        f = open(file, 'rb')
        raw_hdr = f.read(self.NUM_HEADER_BYTES).strip(b'\0')
        f.close()

        raw_hdr = raw_hdr.decode('iso-8859-1')

        # Neuralynx headers seem to start with a line identifying the file, so
        # let's check for it
        hdr_lines = [line.strip() for line in raw_hdr.split('\r\n') if line != '']
        if hdr_lines[0] != '######## Neuralynx Data File Header':
            print_and_log(['Unexpected start to header: ' + hdr_lines[0]], 'debug', logger)

        # Try to read the original file path
        try:
            assert hdr_lines[1].split()[1:3] == ['File', 'Name']
            header[u'FileName'] = ' '.join(hdr_lines[1].split()[3:])
            # hdr['save_path'] = hdr['FileName']
        except:
            print_and_log(['Unable to parse original file path from Neuralynx header: ' + hdr_lines[1]], 'debug', logger)

        # Process lines with file opening and closing times
        header[u'TimeOpened'] = hdr_lines[2][3:]
        header[u'TimeOpened_dt'] = self.parse_neuralynx_time_string(hdr_lines[2])
        header[u'TimeClosed'] = hdr_lines[3][3:]
        header[u'TimeClosed_dt'] = self.parse_neuralynx_time_string(hdr_lines[3])

        # Read the parameters, assuming "-PARAM_NAME PARAM_VALUE" format
        for line in hdr_lines[4:]:
            try:
                name, value = line[1:].split()  # Ignore the dash and split PARAM_NAME and PARAM_VALUE
                header[name] = value
            except:
                print_and_log(['Unable to parse parameter line from Neuralynx header: ' + line], 'debug', logger)

        return header

    def _read_from_header(self):

        folder_path = os.path.dirname(os.path.abspath(self.file_name))
        tmp_all_files = self._get_sorted_channels_()
        regexpr = re.compile('\d+')
        all_files = filter_name_duplicates(tmp_all_files, self.params['ncs_pattern'])

        name = '_'.join(all_files[0].split('_')[:-1])
        self.all_channels = []
        self.all_files = []

        for f in tmp_all_files:
            if f.find(name) > -1:
                self.all_channels += [int(regexpr.findall(f)[0])]
                self.all_files += [f]

        self.header = self._read_header_(self.all_files[0])
        
        header = {}
        header['sampling_rate'] = float(self.header['SamplingFrequency'])        
        header['nb_channels'] = len(self.all_channels)
        header['gain'] = float(self.header['ADBitVolts']) * 1000000

        self.inverse = self.header.has_key('InputInverted') and (self.header['InputInverted'] == 'True')
        if self.inverse:
            header['gain'] *= -1

        g = open(self.all_files[0], 'rb')
        self.size = ((os.fstat(g.fileno()).st_size - self.NUM_HEADER_BYTES)//self.RECORD_SIZE - 1) * self.SAMPLES_PER_RECORD
        self._shape = (self.size, header['nb_channels'])
        g.close()

        return header

    def _get_slice_(self, t_start, t_stop):

        x_beg = numpy.int64(t_start // self.SAMPLES_PER_RECORD)
        r_beg = numpy.mod(t_start, self.SAMPLES_PER_RECORD)
        x_end = numpy.int64(t_stop // self.SAMPLES_PER_RECORD)
        r_end = numpy.mod(t_stop, self.SAMPLES_PER_RECORD)

        data_slice = []

        if x_beg == x_end:
            g_offset = x_beg * self.SAMPLES_PER_RECORD + self.OFFSET_PER_BLOCK[0]*(x_beg + 1) + self.OFFSET_PER_BLOCK[1]*x_beg
            data_slice = numpy.arange(g_offset + r_beg, g_offset + r_end, dtype=numpy.int64)
        else:
            for count, nb_blocks in enumerate(numpy.arange(x_beg, x_end + 1, dtype=numpy.int64)):
                g_offset = nb_blocks * self.SAMPLES_PER_RECORD + self.OFFSET_PER_BLOCK[0]*(nb_blocks + 1) + self.OFFSET_PER_BLOCK[1]*nb_blocks
                if count == 0:
                    data_slice += numpy.arange(g_offset + r_beg, g_offset + self.SAMPLES_PER_RECORD, dtype=numpy.int64).tolist()
                elif count == (x_end - x_beg):
                    data_slice += numpy.arange(g_offset, g_offset + r_end, dtype=numpy.int64).tolist()
                else:
                    data_slice += numpy.arange(g_offset, g_offset + self.SAMPLES_PER_RECORD, dtype=numpy.int64).tolist()
        return data_slice

    def read_chunk(self, idx, chunk_size, padding=(0, 0), nodes=None):
        
        t_start, t_stop = self._get_t_start_t_stop(idx, chunk_size, padding)
        local_shape = t_stop - t_start

        if nodes is None:
            nodes = numpy.arange(self.nb_channels)

        local_chunk = numpy.zeros((local_shape, len(nodes)), dtype=self.data_dtype)
        data_slice = self._get_slice_(t_start, t_stop)

        self._open()
        for count, i in enumerate(nodes):
            local_chunk[:, count] = self.data[i][data_slice]
        self._close()

        return self._scale_data_to_float32(local_chunk)

    def write_chunk(self, time, data):

        t_start = time
        t_stop = time + data.shape[0]

        if t_stop > self.duration:
            t_stop = self.duration

        data_slice = self._get_slice_(t_start, t_stop)
        data = self._unscale_data_from_float32(data)

        self._open(mode='r+')
        for i in range(self.nb_channels):
            self.data[i][data_slice] = data[:, i]
        self._close()

    def _open(self, mode='r'):
        self.data = [
            numpy.memmap(self.all_files[i], offset=self.data_offset, dtype=self.data_dtype, mode=mode)
            for i in range(self.nb_channels)
        ]
        
    def _close(self):
        self.data = None
