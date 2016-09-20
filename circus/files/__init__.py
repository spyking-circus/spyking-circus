from datafile import *

from raw_binary import RawBinaryFile
from mcs_raw_binary import RawMCSFile
from hdf5 import H5File
from kwd import KwdFile
from openephys import OpenEphysFile
from nwb import NWBFile
from arf import ARFFile

try:
	import nixio
	HAVE_NIX_SUPPORT = True
except ImportError:
	HAVE_NIX_SUPPORT = False

try:
	import neuroshare
	HAVE_NEUROSHARE = True
except ImportError:
	HAVE_NEUROSHARE = False

__supported_data_files__ = {
	RawBinaryFile._description : RawBinaryFile,
	RawMCSFile._description : RawMCSFile,
	H5File._description : H5File,
	OpenEphysFile._description : OpenEphysFile,
	KwdFile._description : KwdFile,
	NWBFile._description : NWBFile,
	ARFFile._description : ARFFile,
	NIXFile._description : NixFile
}

if HAVE_NIX_SUPPORT:
	from nixfile import NixFile
	__supported_data_files__[NIXFile._description] = NixFile

if HAVE_NEUROSHARE:
	from mcd import MCDFile
	__supported_data_files__[MCDFile._description] = MCDFile


