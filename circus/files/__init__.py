from raw_binary import RawBinaryFile
from mcs_raw_binary import RawMCSFile
from hdf5 import H5File
from kwd import KwdFile
from openephys import OpenEphysFile
from nwb import NWBFile
from arf import ARFFile
from brw import BRWFile
from npy import NumpyFile
from nix import NixFile

try:
    import neuroshare
    HAVE_NEUROSHARE = True
except ImportError:
    HAVE_NEUROSHARE = False

__supported_data_files__ = {
    RawBinaryFile.description : RawBinaryFile,
    RawMCSFile.description : RawMCSFile,
    H5File.description : H5File,
    OpenEphysFile.description : OpenEphysFile,
    KwdFile.description : KwdFile,
    NWBFile.description : NWBFile,
    NixFile.description : NixFile,
    ARFFile.description : ARFFile,
    BRWFile.description : BRWFile,
    NumpyFile.description : NumpyFile
}


if HAVE_NEUROSHARE:
    from mcd import MCDFile
    __supported_data_files__[MCDFile.description] = MCDFile

    from blackrock import BlackRockFile
    __supported_data_files__[BlackRockFile.description] = BlackRockFile

    from plexon import PlexonFile
    __supported_data_files__[PlexonFile.description] = PlexonFile



def list_all_file_format():
    to_write = ['The file formats that are supported are:', '']
    for file in __supported_data_files__:
        if __supported_data_files__[file].is_writable:
            if __supported_data_files__[file].parallel_write:
                rw = '(read/parallel write)'
            else:
                rw = '(read/write)'
        else:
            rw = '(read only)'    

        streams    = ", ".join(__supported_data_files__[file].is_streamable)
        extensions = ", ".join(__supported_data_files__[file].extension)
        to_write += ['-- ' + file.upper() + ' ' + rw]
        to_write += ['      Extensions       : ' + extensions]
        to_write += ['      Supported streams: ' + streams]
    return to_write