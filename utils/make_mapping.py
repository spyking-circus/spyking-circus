import numpy, hdf5storage, os

def generate_matlab_mapping(probe_file):

    probe     = {}
    probetext = file(probe_file, 'r')
    exec probetext in probe
    probetext.close()

    p         = {}
    positions = []
    nodes     = []
    for key in probe['channel_groups'].keys():
        p.update(probe['channel_groups'][key]['geometry'])
        nodes     +=  probe['channel_groups'][key]['channels']
        positions += [p[channel] for channel in probe['channel_groups'][key]['channels']]
    idx       = numpy.argsort(nodes)
    positions = numpy.array(positions)[idx]
    
    t, ext = os.path.splitext(probe_file)
    hdf5storage.savemat(t + '.mat', {'Positions' : positions/10.})

    message = 
    '''
    To launch the GUI without installing Spyking Circus

    1. Go in spyking-circus/circus/matlab_GUI folder
    2. Launch the GUI with the following command
    
    >> SortingGUI(sampling_rate, 'path/mydata/mydata', '.mat', 'probe_mapping', refactory_period)
    
    sampling_rate, refactory_period are integers

    '''
    print message

    return t