from datafile import *

from raw_binary import RawBinaryFile
from mcs_raw_binary import RawMCSFile
from hdf5 import H5File
from kwik import KwikFile

try:
	import nixio
	HAVE_NIX_SUPPORT = True
except ImportError:
	HAVE_NIX_SUPPORT = False

__supported_data_files__ = {
	'raw_binary' : RawBinaryFile,
	'mcs_raw_binary' : RawMCSFile,
	'hdf5' : H5File,
	'kwik' : KwikFile
}

#if HAVE_NIX_SUPPORT:
#	from nixfile import NixFile
#	__supported_data_files__['nix' : NixFile]