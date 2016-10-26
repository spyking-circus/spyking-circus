import h5py, numpy, re, sys
from neuroshare import NeuroShareFile
import neuroshare as ns

class MCDFile(NeuroShareFile):

    description    = "mcd"    
    extension      = [".mcd"]