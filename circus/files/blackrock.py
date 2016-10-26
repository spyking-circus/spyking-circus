import h5py, numpy, re, sys
from neuroshare import NeuroShareFile
import neuroshare as ns

class BlackRockFile(NeuroShareFile):

    description    = "blackrock"    
    extension      = [".nev"]