import h5py, numpy, re, sys
from circus.shared.messages import print_error, print_and_log
from raw_binary import RawBinaryFile

class RawMCSFile(RawBinaryFile):

    _description    = "mcs_raw_binary"
    _extension      = [".raw", ".dat"]
    _parallel_write = True
    _is_writable    = True

    def __init__(self, file_name, params, empty=False, comm=None):

        kwargs = {}

        if not empty:
            self.file_name = file_name
            a, b, c = self._read_header()
            self.header            = a 
            kwargs['data_offset']  = b
            kwargs['N_tot']        = c
            kwargs['dtype_offset'] = int(self.header['ADC zero'])
            kwargs['gain']         = float(self.header['El'].replace('\xb5V/AD', ''))
            if kwargs['dtype_offset'] == 32768:
                kwargs['data_dtype'] = 'uint16'
            elif kwargs['dtype_offset'] == 0:
                kwargs['data_dtype'] = 'int16'

        RawBinaryFile.__init__(self, file_name, params, empty, comm, **kwargs)

    def _read_header(self):
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

            full_header = {}
            f = open(self.file_name, 'rb')
            h = f.read(header).replace('\r','')
            for i,item in enumerate(h.split('\n')):
                if '=' in item:
                    full_header[item.split(' = ')[0]] = item.split(' = ')[1]
            f.close()

            return full_header, header, len(regexp.findall(full_header['Streams']))
        except Exception:
            print_error(["Wrong MCS header: file is not exported with MCRack"])
            sys.exit(0)