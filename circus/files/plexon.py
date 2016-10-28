import h5py, numpy, re, sys
from neurofile import NeuroShareFile

class PlexonFile(NeuroShareFile):

    description    = "plexon"    
    extension      = [".plx"]