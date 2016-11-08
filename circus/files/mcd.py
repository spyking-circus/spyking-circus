import h5py, numpy, re, sys
from neurofile import NeuroShareFile

class MCDFile(NeuroShareFile):

    description    = "mcd"    
    extension      = [".mcd"]