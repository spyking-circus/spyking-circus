from datafile import *

from raw_binary import RawBinaryFile
from mcs_raw_binary import RawMCSFile
from hdf5 import H5File
from kwik import KwikFile
from kwd import KwdFile
from openephys import OpenEphysFile

try:
	import nixio
	HAVE_NIX_SUPPORT = True
except ImportError:
	HAVE_NIX_SUPPORT = False

__supported_data_files__ = {
	RawBinaryFile._description : RawBinaryFile,
	RawMCSFile._description : RawMCSFile,
	H5File._description : H5File,
	KwikFile._description : KwikFile,
	OpenEphysFile._description : OpenEphysFile,
	KwdFile._description : KwdFile
}

#if HAVE_NIX_SUPPORT:
#	from nixfile import NixFile
#	__supported_data_files__['nix'] = NixFile