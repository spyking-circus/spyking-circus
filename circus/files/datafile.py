import numpy
import re
import sys
import os
import logging
from circus.shared.messages import print_and_log
from circus.shared.mpi import comm


logger = logging.getLogger(__name__)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split('(\d+)', text) ]


def filter_per_extension(files, extension):
    results = []
    for file in files:
        fn, ext = os.path.splitext(file)
        if ext == extension:
            results += [file]
    return results


def get_offset(data_dtype, dtype_offset):

    if dtype_offset == 'auto':
        if data_dtype in ['uint16', numpy.uint16, '<u2', '>u2']:
            dtype_offset = 32768
        elif data_dtype in ['int16', numpy.int16, '<i2', '>i2']:
            dtype_offset = 0
        elif data_dtype in ['int32', numpy.int32, '<i4', '>i4']:
            dtype_offset = 0
        elif data_dtype in ['int64', numpy.int64, '<i8', '>i8']:
            dtype_offset = 0
        elif data_dtype in ['float32', numpy.float32]:
            dtype_offset = 0
        elif data_dtype in ['int8', numpy.int8]:
            dtype_offset = 0
        elif data_dtype in ['uint8', numpy.uint8, '<u1', '>u1']:
            dtype_offset = 127
        elif data_dtype in ['float64', numpy.float64]:
            dtype_offset = 0
        if comm.rank == 0:
            print_and_log(['data type offset for %s is automatically set to %d' % (data_dtype, dtype_offset)], 'debug', logger)
    else:
        try:
            dtype_offset = int(dtype_offset)
        except Exception:
            if comm.rank == 0:
                print_and_log(["Offset %s is not valid" % dtype_offset], 'error', logger)
            sys.exit(1)

    return dtype_offset


class DataFile(object):
    """
    A generic class that will represent how the program interacts with the data. Such an abstraction
    layer should allow people to write their own wrappers, for several file formats, with or without
    parallel write, streams, and so on. Note that depending on the complexity of the datastructure,
    this extra layer can slow down the code.
    """

    description = "mydatafile"  # Description of the file format
    extension = [".myextension"]  # extensions
    parallel_write = False  # can be written in parallel (using the comm object)
    is_writable = False  # can be written
    is_streamable = ['multi-files']  # If the file format can support streams of data ['multi-files' is a default, but can be something else]
    _shape = None  # The total shape of the data (nb time steps, nb channels) accross streams if any
    _t_start = None  # The global t_start of the data
    _t_stop = None  # The final t_stop of the data, accross all streams if any

    # This is a dictionary of values that need to be provided to the constructor, with the corresponding type
    _required_fields = {}

    # This is a dictionary of values that may have a default value, if not provided to the constructor
    _default_values = {}

    _params = {}

    def __init__(self, file_name, params, is_empty=False, stream_mode=None):
        """
        The constructor that will create the DataFile object. Note that by default, values are read from the header
        of the file. If not found in the header, they are read from the parameter file. If no values are found, the
        code will trigger an error

        What you need to specify at a generic level (for a given file format)
            - parallel_write  : can the file be safely written in parallel ?
            - is_writable     : if the file can be written
            - is_streamable   : if the file format can support streaming data
            - required_fields : what parameter must be specified for the file format, along with the type
            - default_values  : parameters that may have default values if not provided

        What you need to specify at a low level (maybe by getting specific values with _read_from_header)
            - _shape          : the size of the data, should be a tuple (duration in time bins, nb_channels)
            - _t_start        : the time (in time steps) of the recording (0 by default)
        """

        self.params = {}
        self.params.update(self._params)

        if not is_empty:
            self._check_filename(file_name)

        if stream_mode is not None:
            self.is_stream = True
            if not stream_mode in self.is_streamable:
                if self.is_master:
                    print_and_log(["The file format %s does not support stream mode %s" % (self.description, stream_mode)], 'error', logger)
                sys.exit(1)
            if is_empty:
                if self.is_master:
                    print_and_log(["A datafile can not have streams and be empty!"], 'error', logger)
                sys.exit(1)
        else:
            self.is_stream = False

        self.file_name = file_name
        self.is_empty = is_empty
        self.stream_mode = stream_mode
        self._is_open = False

        f_next, extension = os.path.splitext(self.file_name)

        self._check_extension(extension)
        self._fill_from_params(params)

        if not self.is_empty:
            # try:
            self._fill_from_header(self._read_from_header())
            # except Exception as ex:
            #     print_and_log(["There is an error in the _read_from_header method of the wrapper\n" + str(ex)], 'error', logger)
        else:
            self._shape = (0, 0)

        if self._shape is None:
            if self.is_master:
                print_and_log(["Shape of the data is not defined. Are you sure of the wrapper?"], 'error', logger)
            sys.exit(1)

        self.params['dtype_offset'] = get_offset(self.data_dtype, self.dtype_offset)

        if self.stream_mode:
            self._sources = self.set_streams(self.stream_mode)
            self._times = []
            for source in self._sources:
                self._times += [source.t_start]
            print_and_log([
                "The file is composed of %d streams" % len(self._sources),
                "Times are between %d and %d" % (self._sources[0].t_start, self._sources[-1].t_stop)
            ], 'debug', logger)

    ##################################################################################################################
    ##################################################################################################################
    #########                  Methods that need to be overwritten for a given fileformat                      #######
    ##################################################################################################################
    ##################################################################################################################

    def _read_from_header(self):
        """
        This function is called only if the file is not empty, and should fill the values in the constructor
        such as _shape. It returns a dictionnary, that will be added to self._params based on the constrains given by
        required_fields and default_values
        """
        raise NotImplementedError('The _read_from_header method needs to be implemented for file format %s' % self.description)

    def _open(self, mode=''):
        """
        This function should open the file
        - mode can be to read only 'r', or to write 'w'
        """
        raise NotImplementedError('The open method needs to be implemented for file format %s' % self.description)

    def _close(self):
        """
        This function closes the file
        """
        raise NotImplementedError('The close method needs to be implemented for file format %s' % self.description)

    def read_chunk(self, idx, chunk_size, padding=(0, 0), nodes=None):
        """
        Assuming the analyze function has been called before, this is the main function
        used by the code, in all steps, to get data chunks. More precisely, assuming your
        dataset can be divided in nb_chunks (see analyze) of temporal size (chunk_size),

            - idx is the index of the chunk you want to load
            - chunk_size is the time of those chunks, in time steps
            - if the data loaded are data[idx:idx+1], padding should add some offsets,
                in time steps, such that we can load data[idx+padding[0]:idx+padding[1]]
            - nodes is a list of nodes, between 0 and nb_channels
        """

        raise NotImplementedError('The get_data method needs to be implemented for file format %s' % self.description)

    def write_chunk(self, time, data):
        """
        This function writes data at a given time.
        - time is expressed in timestep
        - data must be a 2D matrix of size time_length x nb_channels
        """
        raise NotImplementedError('The set_data method needs to be implemented for file format %s' % self.description)

    def set_streams(self, stream_mode):
        """
        This function is only used for file format supporting streams, and need to return a list of datafiles, with
        appropriate t_start for each of them. Note that the results will be using the times defined by the streams.
        You can do anything regarding the keyword used for the stream mode, but multi-files is immplemented by default
        This will allow every file format to be streamed from multiple sources, and processed as a single file.
        """

        if stream_mode == 'multi-files':
            dirname = os.path.abspath(os.path.dirname(self.file_name))
            fname = os.path.basename(self.file_name)
            fn, ext = os.path.splitext(fname)
            all_files = os.listdir(dirname)
            all_files = filter_per_extension(all_files, ext)
            all_files.sort(key=natural_keys)

            sources = []
            to_write = []
            global_time = 0
            params = self.get_description()

            for fname in all_files:
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

    ################################## Optional, only if internal names are changed ##################################

    @property
    def sampling_rate(self):
        return self.params['sampling_rate']

    @property
    def data_dtype(self):
        return self.params['data_dtype']

    @property
    def dtype_offset(self):
        return self.params['dtype_offset']

    @property
    def data_offset(self):
        return self.params['data_offset']

    @property
    def nb_channels(self):
        return int(self.params['nb_channels'])

    @property
    def gain(self):
        return self.params['gain']

    ##################################################################################################################
    ##################################################################################################################
    #########           End of methods that need to be overwritten for a given fileformat                      #######
    ##################################################################################################################
    ##################################################################################################################

    def get_file_names(self):
        res = []
        if self.stream_mode == 'multi-files':
            for source in self._sources:
                res += [source.file_name]
        return res

    def _check_filename(self, file_name):
        if not os.path.exists(file_name):
            if self.is_master:
                print_and_log(["The file %s can not be found!" % file_name], 'error', logger)
            sys.exit(1)

    def _check_extension(self, extension):
        if len(self.extension) > 0:
            if not extension in self.extension + [item.upper() for item in self.extension]:
                if self.is_master:
                    print_and_log(["The extension %s is not valid for a %s file" % (extension, self.description)], 'error', logger)
                sys.exit(1)

    def _fill_from_params(self, params):

        for key in self._required_fields:
            if key not in params:
                self._check_requirements_(params)
            else:
                self.params[key] = self._required_fields[key](params[key])
                if self.is_master:
                    print_and_log(['%s is read from the params with a value of %s' % (key, self.params[key])], 'debug', logger)

        for key in self._default_values:
            if key not in params:
                self.params[key] = self._default_values[key]
                if self.is_master:
                    print_and_log(['%s is not set and has the default value of %s' % (key, self.params[key])], 'debug', logger)
            else:
                self.params[key] = type(self._default_values[key])(params[key])
                if self.is_master:
                    print_and_log(['%s is read from the params with a value of %s' % (key, self.params[key])], 'debug', logger)

    def _fill_from_header(self, header):

        for key in header.keys():
            self.params[key] = header[key]
            if self.is_master:
                print_and_log(['%s is read from the header with a value of %s' % (key, self.params[key])], 'debug', logger)

    def _check_requirements_(self, params):

        missing = {}

        for key, value in self._required_fields.items():
            if key not in params.keys():
                missing[key] = value

        if len(missing) > 0:
            self._display_requirements_()
            sys.exit(1)

    def _display_requirements_(self):

        to_write = ['The parameters for %s file format are:' % self.description.upper(), '']
        nb_params = 0

        for key, value in self._required_fields.items():
            mystring = '-- %s -- %s' % (key, str(value))
            mystring += ' [** mandatory **]'
            to_write += [mystring]
            nb_params += 1

        to_write += ['']

        for key, value in self._default_values.items():
            mystring = '-- %s -- %s' % (key, str(type(value)))
            mystring += ' [default is %s]' % value
            to_write += [mystring]
            nb_params += 1

        if self.is_master:
            if nb_params > 0:
                print_and_log(to_write, 'info', logger)
            else:
                print_and_log(['You do not need to specify anything for file format %s' % self.description.upper()], 'info', logger)

    def _scale_data_to_float32(self, data):
        """
        This function will convert data from local data dtype into float32, the default format of the algorithm
        """
        if self.data_dtype != numpy.float32:
            data = data.astype(numpy.float32)

        if self.dtype_offset != 0:
            data -= self.dtype_offset

        if numpy.any(self.gain != 1):
            data *= self.gain

        if not data.flags['C_CONTIGUOUS']:
            data = numpy.ascontiguousarray(data)

        return data

    def _unscale_data_from_float32(self, data):
        """
        This function will convert data from float32 back to the original format of the file
        """

        if numpy.any(self.gain != 1):
            data /= self.gain

        if self.dtype_offset != 0:
            data += self.dtype_offset

        if (data.dtype != self.data_dtype) and (self.data_dtype != numpy.float32):
            data = data.astype(self.data_dtype)

        return data

    def _count_chunks(self, chunk_size, duration, strict=False):
        """
        This function will count how many block of size chunk_size can be found within a certain duration
        This returns the number of blocks, plus the remaining part
        """
        nb_chunks = duration // chunk_size
        last_chunk_len = duration - nb_chunks * chunk_size

        if self.is_master:
            print_and_log(['There are %d chunks of size %d' % (nb_chunks, chunk_size)], 'debug', logger)

        if not strict and last_chunk_len > 0:
            nb_chunks += 1

        if self.is_master:
            print_and_log(['The last chunk has size %d' % last_chunk_len], 'debug', logger)

        return nb_chunks, last_chunk_len

    def _get_t_start_t_stop(self, idx, chunk_size, padding=(0, 0)):

        t_start = max(0, idx * numpy.int64(chunk_size) + padding[0])
        t_stop = min(self.duration, (idx + 1) * numpy.int64(chunk_size) + padding[1])

        return t_start, t_stop

    def _get_streams_index_by_time(self, local_time):
        if self.is_stream:
            cidx  = numpy.searchsorted(self._times, local_time, 'right') - 1
            return cidx

    def is_first_chunk(self, idx, nb_chunks):

        if self.is_stream:
            cidx = numpy.searchsorted(self._chunks_in_sources, idx, 'right') - 1
            idx -= self._chunks_in_sources[cidx]
            if idx == 0:
                return True
        else:
            if idx == 0:
                return True
        return False

    def is_last_chunk(self, idx, nb_chunks):

        if self.is_stream:
            if (idx > 0) and (idx in self._chunks_in_sources - 1):
                return True
        else:
            if idx == nb_chunks:
                return True
        return False

    def get_snippet(self, global_time, length, nodes=None):
        """
        This function should return a time snippet of size length x nodes
        - time is in timestep
        - length is in timestep
        - nodes is a list of nodes, between 0 and nb_channels
        """
        if self.is_stream:
            cidx = self._get_streams_index_by_time(global_time)
            return self._sources[cidx].get_snippet(global_time, length, nodes)
        else:
            local_time = global_time - self.t_start
            return self.get_data(0, chunk_size=length, padding=(local_time, local_time), nodes=nodes)[0]

    def get_data(self, idx, chunk_size, padding=(0, 0), nodes=None):

        if self.is_stream:
            cidx = numpy.searchsorted(self._chunks_in_sources, idx, 'right') - 1
            idx -= self._chunks_in_sources[cidx]
            return self._sources[cidx].read_chunk(idx, chunk_size, padding, nodes), self._sources[cidx].t_start + idx * chunk_size
        else:
            return self.read_chunk(idx, chunk_size, padding, nodes), self.t_start + idx*chunk_size

    def set_data(self, global_time, data):

        if self.is_stream:
            cidx = self._get_streams_index_by_time(global_time)
            local_time = global_time - self._sources[cidx].t_start
            return self._sources[cidx].write_chunk(local_time, data)
        else:
            local_time = global_time - self.t_start
            return self.write_chunk(local_time, data)

    def analyze(self, chunk_size, strict=False):
        """
        This function should return two values:
        - the number of temporal chunks of temporal size chunk_size that can be found
          in the data. Note that even if the last chunk is not complete, it has to be
          counted. chunk_size is expressed in time steps
        - the length of the last uncomplete chunk, in time steps
        """
        if self.is_stream:
            nb_chunks = 0
            last_chunk_len = 0
            self._chunks_in_sources = [0]

            for source in self._sources:
                a, b = self._count_chunks(chunk_size, source.duration, strict)
                nb_chunks += a
                last_chunk_len += b

                self._chunks_in_sources += [nb_chunks]

            self._chunks_in_sources = numpy.array(self._chunks_in_sources)

            return nb_chunks, last_chunk_len
        else:
            return self._count_chunks(chunk_size, self.duration, strict)

    def get_description(self):
        result = {}
        for key in ['sampling_rate', 'data_dtype', 'gain', 'nb_channels', 'dtype_offset'] + self._default_values.keys() + self._required_fields.keys():
            result[key] = self.params[key]
        return result

    @property
    def shape(self):
        return self.duration, int(self.nb_channels)

    @property
    def duration(self):
        if self.is_stream:
            duration = 0
            for source in self._sources:
                duration += source.duration
            return duration
        else:
            return numpy.int64(self._shape[0])

    @property
    def is_master(self):
        return comm.rank == 0

    @property
    def t_start(self):
        if self.is_stream:
            return self._sources[0].t_start
        else:
            if self._t_start is None:
                self._t_start = 0
            return self._t_start

    @property
    def t_stop(self):
        if self.is_stream:
            return self._sources[-1].t_stop
        else:
            if self._t_stop is None:
                self._t_stop = self.t_start + self.duration
            return self._t_stop

    @property
    def nb_streams(self):
        if self.is_stream:
            return len(self._sources)
        else:
            return 1

    def open(self, mode='r'):
        if self.is_stream:
            for source in self._sources:
                source._open(mode)
        else:
            self._open(mode)
        self._is_open = True
        self._mode = mode


    def close(self):
        if self.is_stream:
            for source in self._sources:
                source._close()
        else:
            self._close()

        self._is_open = False