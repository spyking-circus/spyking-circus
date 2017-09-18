import h5py, numpy, re, sys, logging
from datafile import DataFile
from circus.shared.messages import print_and_log
import sys, struct, os

logger = logging.getLogger(__name__)


def read_header(fid):
    """Reads the Intan File Format header from the given file."""

    # Check 'magic number' at beginning of file to make sure this is an Intan
    # Technologies RHD2000 data file.
    magic_number, = struct.unpack('<I', fid.read(4)) 
    if magic_number != int('c6912702', 16): raise Exception('Unrecognized file type.')

    header = {}
    # Read version number.
    version = {}
    (version['major'], version['minor']) = struct.unpack('<hh', fid.read(4)) 
    header['version'] = version

    freq = {}

    # Read information of sampling rate and amplifier frequency settings.
    header['sample_rate'], = struct.unpack('<f', fid.read(4))
    (freq['dsp_enabled'], freq['actual_dsp_cutoff_frequency'], freq['actual_lower_bandwidth'], freq['actual_upper_bandwidth'], 
    freq['desired_dsp_cutoff_frequency'], freq['desired_lower_bandwidth'], freq['desired_upper_bandwidth']) = struct.unpack('<hffffff', fid.read(26))


    # This tells us if a software 50/60 Hz notch filter was enabled during
    # the data acquisition.
    notch_filter_mode, = struct.unpack('<h', fid.read(2))
    header['notch_filter_frequency'] = 0
    if notch_filter_mode == 1:
        header['notch_filter_frequency'] = 50
    elif notch_filter_mode == 2:
        header['notch_filter_frequency'] = 60
    freq['notch_filter_frequency'] = header['notch_filter_frequency']

    (freq['desired_impedance_test_frequency'], freq['actual_impedance_test_frequency']) = struct.unpack('<ff', fid.read(8))

    note1 = read_qstring(fid)
    note2 = read_qstring(fid)
    note3 = read_qstring(fid)
    header['notes'] = { 'note1' : note1, 'note2' : note2, 'note3' : note3}

    # If data file is from GUI v1.1 or later, see if temperature sensor data was saved.
    header['num_temp_sensor_channels'] = 0
    if (version['major'] == 1 and version['minor'] >= 1) or (version['major'] > 1) :
        header['num_temp_sensor_channels'], = struct.unpack('<h', fid.read(2))


    # If data file is from GUI v1.3 or later, load eval board mode.
    header['eval_board_mode'] = 0
    if ((version['major'] == 1) and (version['minor'] >= 3)) or (version['major'] > 1) :
        header['eval_board_mode'], = struct.unpack('<h', fid.read(2))

    # Place frequency-related information in data structure. (Note: much of this structure is set above)
    freq['amplifier_sample_rate'] = header['sample_rate']
    freq['aux_input_sample_rate'] = header['sample_rate'] / 4
    freq['supply_voltage_sample_rate'] = header['sample_rate'] / 60
    freq['board_adc_sample_rate'] = header['sample_rate']
    freq['board_dig_in_sample_rate'] = header['sample_rate']

    header['frequency_parameters'] = freq

    # Create structure arrays for each type of data channel.
    header['spike_triggers'] = []
    header['amplifier_channels'] = []
    header['aux_input_channels'] = []
    header['supply_voltage_channels'] = []
    header['board_adc_channels'] = []
    header['board_dig_in_channels'] = []
    header['board_dig_out_channels'] = []

    # Read signal summary from data file header.

    if (header['version']['major'] > 1):
        header['reference_channel'] = read_qstring(fid)

    number_of_signal_groups, = struct.unpack('<h', fid.read(2))

    for signal_group in range(0, number_of_signal_groups):
        signal_group_name = read_qstring(fid)
        signal_group_prefix = read_qstring(fid)
        (signal_group_enabled, signal_group_num_channels, signal_group_num_amp_channels) = struct.unpack('<hhh', fid.read(6))

        if (signal_group_num_channels > 0) and (signal_group_enabled > 0):
            for signal_channel in range(0, signal_group_num_channels):
                new_channel = {'port_name' : signal_group_name, 'port_prefix' : signal_group_prefix, 'port_number' : signal_group}
                new_channel['native_channel_name'] = read_qstring(fid)
                new_channel['custom_channel_name'] = read_qstring(fid)
                (new_channel['native_order'], new_channel['custom_order'], signal_type, channel_enabled, new_channel['chip_channel'], new_channel['board_stream']) = struct.unpack('<hhhhhh', fid.read(12))
                new_trigger_channel = {}
                (new_trigger_channel['voltage_trigger_mode'], new_trigger_channel['voltage_threshold'], new_trigger_channel['digital_trigger_channel'], new_trigger_channel['digital_edge_polarity'])  = struct.unpack('<hhhh', fid.read(8))
                (new_channel['electrode_impedance_magnitude'], new_channel['electrode_impedance_phase']) = struct.unpack('<ff', fid.read(8))

                if channel_enabled:
                    if signal_type == 0:
                        header['amplifier_channels'].append(new_channel)
                        header['spike_triggers'].append(new_trigger_channel)
                    elif signal_type == 1:
                        header['aux_input_channels'].append(new_channel)
                    elif signal_type == 2:
                        header['supply_voltage_channels'].append(new_channel)
                    elif signal_type == 3:
                        header['board_adc_channels'].append(new_channel)
                    elif signal_type == 4:
                        header['board_dig_in_channels'].append(new_channel)
                    elif signal_type == 5:
                        header['board_dig_out_channels'].append(new_channel)
                    else:
                        raise Exception('Unknown channel type.')


    # Summarize contents of data file.
    header['num_amplifier_channels'] = len(header['amplifier_channels'])
    header['num_aux_input_channels'] = len(header['aux_input_channels'])
    header['num_supply_voltage_channels'] = len(header['supply_voltage_channels'])
    header['num_board_adc_channels'] = len(header['board_adc_channels'])
    header['num_board_dig_in_channels'] = len(header['board_dig_in_channels'])
    header['num_board_dig_out_channels'] = len(header['board_dig_out_channels'])

    return header


def get_bytes_per_data_block(header):
    """Calculates the number of bytes in each 60-sample datablock."""

    if (header['version']['major'] == 1):
        num_samples_per_data_block = 60
    else:
        num_samples_per_data_block = 128

    # Each data block contains 60 amplifier samples.
    bytes_per_block = num_samples_per_data_block * 4  # timestamp data
    bytes_per_block = bytes_per_block + num_samples_per_data_block * 2 * header['num_amplifier_channels']

    # Auxiliary inputs are sampled 4x slower than amplifiers
    bytes_per_block = bytes_per_block + (num_samples_per_data_block / 4) * 2 * header['num_aux_input_channels']

    # Supply voltage is sampled 60x slower than amplifiers
    bytes_per_block = bytes_per_block + 1 * 2 * header['num_supply_voltage_channels']

    # Board analog inputs are sampled at same rate as amplifiers
    bytes_per_block = bytes_per_block + num_samples_per_data_block * 2 * header['num_board_adc_channels']

    # Board digital inputs are sampled at same rate as amplifiers
    if header['num_board_dig_in_channels'] > 0:
        bytes_per_block = bytes_per_block + num_samples_per_data_block * 2

    # Board digital outputs are sampled at same rate as amplifiers
    if header['num_board_dig_out_channels'] > 0:
        bytes_per_block = bytes_per_block + num_samples_per_data_block * 2

    # Temp sensor is sampled 60x slower than amplifiers
    if header['num_temp_sensor_channels'] > 0:
        bytes_per_block = bytes_per_block + 1 * 2 * header['num_temp_sensor_channels']

    return bytes_per_block



def read_qstring(fid):
    """Read Qt style QString.  

    The first 32-bit unsigned number indicates the length of the string (in bytes).  
    If this number equals 0xFFFFFFFF, the string is null.

    Strings are stored as unicode.
    """

    length, = struct.unpack('<I', fid.read(4))
    if length == int('ffffffff', 16): return ""

    if length > (os.fstat(fid.fileno()).st_size - fid.tell() + 1) :
        print(length)
        raise Exception('Length too long.')

    # convert length from bytes to 16-bit Unicode words
    length = int(length / 2)

    data = []
    for i in range(0, length):
        c, = struct.unpack('<H', fid.read(2))
        data.append(c)

    if sys.version_info >= (3,0):
        a = ''.join([chr(c) for c in data])
    else:
        a = ''.join([unichr(c) for c in data])
    
    return a
  

class RHDFile(DataFile):

    description    = "rhd"    
    extension      = [".rhd"]
    parallel_write = True
    is_writable    = True
    is_streamable  = ['multi-files']

    _required_fields = {}
    _default_values  = {}

    _params          = {'dtype_offset' : 'auto',
                        'data_dtype'   : 'uint16',
                        'gain'         : 0.195}

    

    def _read_from_header(self):

        header = {}

        self.file  = open(self.file_name)
        full_header = read_header(self.file)
        header['nb_channels']   = full_header['num_amplifier_channels']  
        header['sampling_rate'] = full_header['sample_rate']
        
        if full_header['version']['major'] == 1:
            self.SAMPLES_PER_RECORD = 60
        else:
            self.SAMPLES_PER_RECORD = 128

        header['data_offset']   = self.file.tell()
        data_present         = False
        filesize             = os.path.getsize(self.file_name)
        self.bytes_per_block = get_bytes_per_data_block(full_header)
        self.block_offset    = self.SAMPLES_PER_RECORD * 4
        self.block_size      = 2 * self.SAMPLES_PER_RECORD * header['nb_channels']
        bytes_remaining      = filesize - self.file.tell()

        self.bytes_per_block_div = self.bytes_per_block / 2
        self.block_offset_div    = self.block_offset / 2
        self.block_size_div      = self.block_size / 2

        if bytes_remaining > 0:
            data_present = True
        if bytes_remaining % self.bytes_per_block != 0:
            print_and_log(['Something is wrong with file size : should have a whole number of data blocks'], 'error', logger)

        num_data_blocks = int(bytes_remaining / self.bytes_per_block)
        self.num_amplifier_samples = self.SAMPLES_PER_RECORD * num_data_blocks

        self.size        = self.num_amplifier_samples
        self._shape      = (self.size, header['nb_channels'])
        self.file.close()

        return header


    def _get_slice_(self, t_start, t_stop):

        x_beg = numpy.int64(t_start // self.SAMPLES_PER_RECORD)
        r_beg = numpy.mod(t_start, self.SAMPLES_PER_RECORD)
        x_end = numpy.int64(t_stop // self.SAMPLES_PER_RECORD)
        r_end = numpy.mod(t_stop, self.SAMPLES_PER_RECORD)

        data_slice  = [[]]

        if x_beg == x_end:
            g_offset = x_beg * self.bytes_per_block_div + self.block_offset_div
            data_slice = [numpy.arange(g_offset + r_beg * self.nb_channels, g_offset + r_end * self.nb_channels, dtype=numpy.int64)]
        else:
            for count, nb_blocks in enumerate(numpy.arange(x_beg, x_end + 1, dtype=numpy.int64)):
                g_offset = nb_blocks * self.bytes_per_block_div + self.block_offset_div
                if count == 0:
                    data_slice += [numpy.arange(g_offset + r_beg * self.nb_channels, g_offset + self.block_size_div, dtype=numpy.int64).tolist()]
                elif (count == (x_end - x_beg)):
                    data_slice += [numpy.arange(g_offset, g_offset + r_end * self.nb_channels, dtype=numpy.int64).tolist()]
                else:
                    data_slice += [numpy.arange(g_offset, g_offset + self.block_size_div, dtype=numpy.int64).tolist()]

        return data_slice 


    def read_chunk(self, idx, chunk_size, padding=(0, 0), nodes=None):
        
        t_start, t_stop = self._get_t_start_t_stop(idx, chunk_size, padding)
        local_shape     = t_stop - t_start

        local_chunk = numpy.zeros((self.nb_channels, local_shape), dtype=self.data_dtype)
        data_slice  = self._get_slice_(t_start, t_stop) 

        self._open()
        count = 0
        for s in data_slice:
            t_slice = len(s)/self.nb_channels
            local_chunk[:, count:count + t_slice] = self.data[s].reshape(self.nb_channels, len(s)/self.nb_channels)
            count += t_slice

        local_chunk = local_chunk.T
        self._close()

        if nodes is not None:
            if not numpy.all(nodes == numpy.arange(self.nb_channels)):
                local_chunk = numpy.take(local_chunk, nodes, axis=1)

        return self._scale_data_to_float32(local_chunk)

    def write_chunk(self, time, data):

        t_start     = time
        t_stop      = time + data.shape[0]

        if t_stop > self.duration:
            t_stop  = self.duration


        data = self._unscale_data_from_float32(data)

        data_slice  = self._get_slice_(t_start, t_stop) 
        
        self._open(mode='r+')
        count = 0
        for s in data_slice:
            t_slice      = len(s)/self.nb_channels
            self.data[s] = data[count:count + t_slice, :].T.ravel()
            count += t_slice

        self._close()

    def _open(self, mode='r'):
        self.data = numpy.memmap(self.file_name, offset=self.data_offset, dtype=self.data_dtype, mode=mode)

    def _close(self):
        self.file.close()