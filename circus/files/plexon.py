import h5py, numpy, re, sys
from neuroshare import NeuroShareFile
import neuroshare as ns

class PlexonFile(NeuroShareFile):

    description    = "plexon"    
    extension      = [".plx"]