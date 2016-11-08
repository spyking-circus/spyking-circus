import h5py, numpy, re, sys
from neurofile import NeuroShareFile

class BlackRockFile(NeuroShareFile):

    description    = "blackrock"    
    extension      = [".nev"]