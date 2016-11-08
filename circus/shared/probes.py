import numpy, os, sys, logging
from messages import print_and_log
from mpi import comm

logger = logging.getLogger(__name__)

def read_probe(parser):
    probe    = {}
    filename = os.path.abspath(os.path.expanduser(parser.get('data', 'mapping')))
    if comm.rank == 0:
        print_and_log(["Reading the probe file %s" %filename], 'debug', logger)
    if not os.path.exists(filename):
        if comm.rank == 0:
            print_and_log(["The probe file %s can not be found" %filename], 'error', logger)
        sys.exit(1)
    try:
        with open(filename, 'r') as f:
            probetext = f.read()
            exec(probetext, probe)
    except Exception as ex:
        if comm.rank == 0:
            print_and_log(["Something wrong with the syntax of the probe file:\n" + str(ex)], 'error', logger)
        sys.exit(1)

    key_flags = ['total_nb_channels', 'radius', 'channel_groups']
    for key in key_flags:
        if not probe.has_key(key):
            if comm.rank == 0:
                print_and_log(["%s is missing in the probe file" %key], 'error', logger)
            sys.exit(1)
    return probe


def get_nodes_and_edges(parser, validating=False):
    """
    Retrieve the topology of the probe.
    
    Other parameters
    ----------------
    radius : integer
    
    Returns
    -------
    nodes : ndarray of integers
        Array of channel ids retrieved from the description of the probe.
    edges : dictionary
        Dictionary which link each channel id to the ids of the channels whose
        distance is less or equal than radius.
    
    """
    
    edges  = {}
    nodes  = []
    radius = parser.getint('detection', 'radius')

    if validating:
        radius_factor = parser.getfloat('validating', 'radius_factor')
        radius = int(radius_factor * float(radius))

    def get_edges(i, channel_groups):
        edges = []
        pos_x, pos_y = channel_groups['geometry'][i]
        for c2 in channel_groups['channels']:
            pos_x2, pos_y2 = channel_groups['geometry'][c2]
            if (((pos_x - pos_x2)**2 + (pos_y - pos_y2)**2) <= radius**2):
                edges += [c2]
        return edges

    for key in parser.probe['channel_groups'].keys():
        for i in parser.probe['channel_groups'][key]['channels']:
            edges[i] = get_edges(i, parser.probe['channel_groups'][key])
            nodes   += [i]

    return numpy.sort(numpy.array(nodes, dtype=numpy.int32)), edges


def get_averaged_n_edges(parser):
    nodes, edges = get_nodes_and_edges(parser)
    n = 0
    for key, value in edges.items():
        n += len(value)
    return n/float(len(edges.values()))