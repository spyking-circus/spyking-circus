import h5py, numpy, re, sys
import ConfigParser as configparser
from circus.shared.messages import print_error, print_and_log
from raw_binary import RawBinaryFile

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