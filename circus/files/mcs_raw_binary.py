import h5py, numpy, re, sys, re, logging
from circus.shared.messages import print_and_log
from raw_binary import RawBinaryFile

logger = logging.getLogger(__name__)

class RawMCSFile(RawBinaryFile):

    description    = "mcs_raw_binary"
    extension      = [".raw", ".dat"]
    parallel_write = True
    is_writable    = True

    def _get_header(self):
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
                print_and_log(['Wrong MCS header: file is not exported with MCRack'], 'error', logger)
                sys.exit(1) 
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
            print_and_log(["Wrong MCS header: file is not exported with MCRack"], 'error', logger)
            sys.exit(1)

    def _read_from_header(self):

        a, b, c                = self._get_header()
        header                 = a 
        header['data_offset']  = b
        header['nb_channels']  = c
        header['dtype_offset'] = int(header['ADC zero'])
        header['gain']         = float(re.findall("\d+\.\d+", header['El'])[0])
        
        if header['dtype_offset'] > 0:
            header['data_dtype'] = 'uint16'
        elif header['dtype_offset'] == 0:
             header['data_dtype'] = 'int16'

        return header