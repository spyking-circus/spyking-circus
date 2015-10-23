import numpy, hdf5storage, h5py, os, progressbar, platform
import ConfigParser as configparser
from termcolor import colored

def purge(file, pattern):
    if platform.system() == 'Windows':
        dir = "\\".join(file.split("\\")[:-1])
    else:
        dir = "/".join(file.split("/")[:-1])
    for f in os.listdir(dir):
        if f.find(pattern) > -1:
            os.remove(os.path.join(dir, f))

def detect_header(filename, value='MCS'):

    if value == 'MCS':
        header      = 0
        stop        = False
        fid         = open(filename, 'r')
        header_text = ''

        while ((stop is False) and (header <= 2000)):
            header      += 1
            char         = fid.read(1)
            header_text += str(char)
            if (header > 2):
                if (header_text[header-3:header] == 'EOH'):
                    stop = True
        fid.close()
        if stop is False:
            print_info(['File is not exported with MCRack: header is set to 0'])
            header  = 0 
        else:
            header += 2
        return header
    else:
        return value

def change_flag(file_name, flag, value, avoid_flag=None):
    extension       = '.' + file_name.split('.')[-1]
    file_params     = file_name.replace(extension, '.params')
    f     = open(file_params, 'r')
    lines = f.readlines()
    f.close()
    f     = open(file_params, 'w')
    to_write = '%s      = %s              #!! AUTOMATICALLY EDITED: DO NOT MODIFY !!\n' %(flag, value)
    for line in lines:
        if avoid_flag is not None:
            mytest = (line.find(flag) > -1) and (line.find(avoid_flag) == -1)
        else:
            mytest = (line.find(flag) > -1)
        if mytest:
            f.write(to_write)
        else:
            f.write(line)
    f.close()

def load_parameters(file_name):

    extension       = '.' + file_name.split('.')[-1]
    file_params     = file_name.replace(extension, '.params')
    parser          = configparser.SafeConfigParser()
    parser.read(file_params)

    sections = ['data', 'whitening', 'extracting', 'clustering', 'fitting', 'filtering', 'noedits']
    for section in sections:
        for (key, value) in parser.items(section):
            parser.set(section, key, value.split('#')[0].replace(' ', '')) 

    if platform.system() == 'Windows':
        file_path   = "\\".join(file_name.split("\\")[:-1])
    else:
        file_path   = "/".join(file_name.split("/")[:-1])
    file_name       = file_name.replace(extension, '')

    N_t             = parser.getfloat('data', 'N_t')
    sampling_rate   = parser.getint('data', 'sampling_rate')
    N_t             = int(sampling_rate*N_t*1e-3)
    if numpy.mod(N_t, 2) == 0:
        N_t += 1
    parser.set('data', 'N_t', str(N_t))
    parser.set('data', 'chunk_size', str(60*sampling_rate))
    parser.set('data', 'template_shift', str(int((N_t-1)/2)))

    data_offset = parser.get('data', 'data_offset')
    parser.set('data', 'data_offset', str(detect_header(file_name+extension, data_offset)))
    
    probe     = {}
    try:
        probetext = file(parser.get('data', 'mapping'), 'r')
        exec probetext in probe
        probetext.close()
    except Exception:
        print_error(["Something wrong with the probe file!"])
    #mapping_elec = hdf5storage.loadmat(parser.get('data', 'mapping'))['Mapping']
    parser.set('data', 'N_total', str(probe['total_nb_channels']))   
    N_e = 0
    for key in probe['channel_groups'].keys():
        N_e += len(probe['channel_groups'][key]['channels'])
    parser.set('data', 'N_e', str(N_e))   

    for section in ['whitening', 'clustering']:
        test = (parser.getfloat(section, 'nb_elts') > 0) and (parser.getfloat(section, 'nb_elts') <= 1)
        assert test, colored("nb_elts in %s should be in [0,1]" %section, 'red')

    test = (parser.getfloat('clustering', 'nclus_min') > 0) and (parser.getfloat(section, 'nclus_min') <= 1)
    assert test, colored("nclus_min in clustering should be in [0,1]", 'red')

    try:
        os.makedirs(file_name)
    except Exception:
        pass

    if platform.system() == 'Windows':
        file_out = file_name + '\\'+ file_name.split('\\')[-1]
    else:
        file_out = file_name + '/'+ file_name.split('/')[-1]
    parser.set('data', 'file_name', file_name)
    parser.set('data', 'file_out', file_out) # Output file without suffix
    parser.set('data', 'file_out_suff', file_out  + parser.get('data', 'suffix')) # Output file with suffix
    parser.set('data', 'data_file', file_name + extension)   # Data file (assuming .filtered at the end)
    parser.set('data', 'data_file_noext', file_name)   # Data file (assuming .filtered at the end)
    parser.set('data', 'dist_peaks', str(N_t)) # Get only isolated spikes for a single electrode (whitening, clustering, basis)    
    
    parser.set('fitting', 'space_explo', '0.75')
    parser.set('fitting', 'nb_chances', '3')

    try: 
        parser.get('data', 'radius')
    except Exception:
        parser.set('data', 'radius', 'auto')
    try:
        parser.getint('data', 'radius')
    except Exception:
        parser.set('data', 'radius', str(int(probe['radius'])))

    new_values = [['fitting', 'amp_auto', 'bool', 'True'], 
                  ['fitting', 'spike_range', 'float', '0'],
                  ['data', 'spikedetekt', 'bool', 'False'],
                  ['data', 'global_tmp', 'bool', 'True'],
                  ['clustering', 'max_clusters', 'int', '10'],
                  ['clustering', 'nb_repeats', 'int', '3'],
                  ['clustering', 'make_plots', 'bool', 'True'],
                  ['clustering', 'test_clusters', 'bool', 'False'],
                  ['clustering', 'sim_same_elec', 'float', '3'],
                  ['clustering', 'smart_search', 'float', '3'],
                  ['clustering', 'safety_space', 'bool', 'True'],
                  ['clustering', 'noise_thr', 'float', '0.8'],
                  ['clustering', 'cc_merge', 'float', '0.975'],
                  ['clustering', 'cc_delay', 'float', '0'],
                  ['extracting', 'cc_merge', 'float', '0.95'],
                  ['extracting', 'noise_thr', 'float', '0.8']]

    for item in new_values:
        section, name, val_type, value = item
        try:
            if val_type is 'bool':
                parser.getboolean(section, name)
            elif val_type is 'int':
                parser.getint(section, name)
            elif val_type is 'float':
                parser.getfloat(section, name)
        except Exception:
            parser.set(section, name, value)

    return parser


def data_stats(params):
    data_file      = params.get('data', 'data_file')
    data_offset    = params.getint('data', 'data_offset')
    data_dtype     = params.get('data', 'data_dtype')
    N_total        = params.getint('data', 'N_total')
    N_e            = params.getint('data', 'N_e')
    sampling_rate  = params.getint('data', 'sampling_rate')
    datablock      = numpy.memmap(data_file, offset=data_offset, dtype=data_dtype)
    N              = len(datablock)
    chunk_len      = N_total * (60 * sampling_rate)
    nb_chunks      = N / chunk_len
    last_chunk_len = (N - nb_chunks * chunk_len)/(N_total*sampling_rate)

    N_t             = params.getint('data', 'N_t')
    N_t             = numpy.round(1000.*N_t/sampling_rate, 1)

    lines = ["Number of recorded channels : %d" %N_total,
             "Number of analyzed channels : %d" %N_e,
             "Data type                   : %s" %str(data_dtype),
             "Header offset for the data  : %d" %data_offset,
             "Duration of the recording   : %d min %s s" %(nb_chunks, last_chunk_len),
             "Width of the templates      : %d ms" %N_t,
             "Spatial radius considered   : %d um" %params.getint('data', 'radius')]
    print_info(lines)

def print_info(lines):
    print colored("-----------------------  Informations  -----------------------", 'yellow')
    for line in lines:
        print colored("| " + line, 'yellow')
    print colored("--------------------------------------------------------------", 'yellow')

def print_error(lines):
    print colored("--------------------------  Error  ---------------------------", 'red')
    for line in lines:
        print colored("| " + line, 'red')
    print colored("--------------------------------------------------------------", 'red')


def load_chunk(params, idx, chunk_len, chunk_size=None, padding=(0, 0), nodes=None):
    
    if chunk_size is None:
        chunk_size = params.getint('data', 'chunk_size')
    data_file    = params.get('data', 'data_file')
    data_offset  = params.getint('data', 'data_offset')
    dtype_offset = params.getint('data', 'dtype_offset')
    data_dtype   = params.get('data', 'data_dtype')
    N_total      = params.getint('data', 'N_total')
    gain         = params.getfloat('data', 'gain')
    datablock    = numpy.memmap(data_file, offset=data_offset, dtype=data_dtype)
    local_chunk  = datablock[idx*chunk_len+padding[0]:(idx+1)*chunk_len+padding[1]]
    local_shape  = chunk_size + (padding[1]-padding[0])/N_total
    local_chunk  = local_chunk.reshape(local_shape, N_total)
    local_chunk  = local_chunk.astype(numpy.float32)
    local_chunk -= dtype_offset
    local_chunk *= gain
    if nodes is not None:
        local_chunk  = local_chunk[:, nodes]
    return local_chunk, local_shape

def analyze_data(params, chunk_size=None):

    if chunk_size is None:
        chunk_size = params.getint('data', 'chunk_size')
    data_file      = params.get('data', 'data_file')
    data_offset    = params.getint('data', 'data_offset')
    data_dtype     = params.get('data', 'data_dtype')
    N_total        = params.getint('data', 'N_total')
    template_shift = params.getint('data', 'template_shift')
    datablock      = numpy.memmap(data_file, offset=data_offset, dtype=data_dtype)
    N              = len(datablock)
    chunk_len      = N_total * chunk_size
    nb_chunks      = N / chunk_len
    last_chunk_len = N - nb_chunks * chunk_len
    last_chunk_len = N_total * int(last_chunk_len/N_total)
    borders        = N_total * template_shift
    return borders, nb_chunks, chunk_len, last_chunk_len

def get_nodes_and_edges(parameters):
    
    edges     = {}
    nodes     = []
    probe     = {}
    probetext = file(parameters.get('data', 'mapping'), 'r')

    try:
        exec probetext in probe
    except Exception:
        print_error(["Something wrong with the probe file!"])
    probetext.close()

    radius = parameters.getint('data', 'radius')

    def get_edges(i, channel_groups):
        edges = []
        pos_x, pos_y = channel_groups['geometry'][i]
        for c2 in channel_groups['channels']:
            pos_x2, pos_y2 = channel_groups['geometry'][c2]
            if (((pos_x - pos_x2)**2 + (pos_y - pos_y2)**2) <= radius**2):
                edges += [c2]
        return edges

    for key in probe['channel_groups'].keys():
        for i in probe['channel_groups'][key]['channels']:
            edges[i] = get_edges(i, probe['channel_groups'][key])
            nodes   += [i]

    return numpy.sort(numpy.array(nodes, dtype=numpy.int32)), edges

def load_data(params, data, extension=''):

    file_out      = params.get('data', 'file_out')
    file_out_suff = params.get('data', 'file_out_suff')
    data_file_noext = params.get('data', 'data_file_noext')

    if data == 'thresholds':
        spike_thresh = params.getfloat('data', 'spike_thresh')
        N_e          = params.getint('data', 'N_e') 
        if os.path.exists(file_out + '.thresholds.npy'):
            thresholds = spike_thresh * numpy.load(file_out + '.thresholds.npy')
        else:
            thresholds = spike_thresh * numpy.ones(N_e, dtype=numpy.float32)
        return thresholds
    elif data == 'spatial_whitening':
        if os.path.exists(file_out + '.whitening.mat'):
            return hdf5storage.loadmat(file_out + '.whitening.mat')['spatial']
        else:
            raise Exception('Whitening matrix has to be computed first!')
    elif data == 'temporal_whitening':
        if os.path.exists(file_out + '.whitening.mat'):
            return hdf5storage.loadmat(file_out + '.whitening.mat')['temporal']
        else:
            raise Exception('Whitening matrix has to be computed first!')
    elif data == 'basis':
        N_t = params.getfloat('data', 'N_t')
        if os.path.exists(file_out + '.basis.npz'):
            basis      = numpy.load(file_out + '.basis.npz')
            basis_proj = basis['proj']
            basis_rec  = basis['rec']
        else:
            basis_proj = numpy.identity(N_t)
            basis_rec  = numpy.identity(N_t)
        return basis_proj, basis_rec
    elif data == 'templates':
        if os.path.exists(file_out_suff + '.templates%s.mat' %extension):
           return hdf5storage.loadmat(file_out_suff + '.templates%s.mat' %extension)['templates']
        else:
            raise Exception('No templates found! Check suffix?')
    elif data == 'spike-cluster':
        file_name = params.get('data', 'data_file_noext') + '.spike-cluster.mat'
        if os.path.exists(file_name):
            data       = hdf5storage.loadmat(file_name)
            clusters   = data['clusters'].flatten()
            N_clusters = len(numpy.unique(clusters))
            spiketimes = data['spikes'].flatten()
            return clusters, spiketimes, N_clusters
        else:
            raise Exception('Need to provide a spike-cluster file!')
    elif data == 'spikedetekt':
        file_name = params.get('data', 'data_file_noext') + ".kwik"
        if os.path.exists(file_name):
            return h5py.File(file_name).get('channel_groups/1/spikes/time_samples')[:].astype(numpy.int32)
        else:
            raise Exception('No clusters found! Check suffix or run clustering?')
    elif data == 'clusters':
        if os.path.exists(file_out_suff + '.clusters%s.mat' %extension):
            return hdf5storage.loadmat(file_out_suff + '.clusters%s.mat' %extension)
        else:
            raise Exception('No clusters found! Check suffix or run clustering?')
    elif data == 'results':
        try:
            return get_results(params, extension)
        except Exception:
            raise Exception('No results found! Check suffix or run the fitting?')
    elif data == 'overlaps':
        try:
            return get_overlaps(params, extension)
        except Exception:
            raise Exception('No overlaps found! Check suffix or run the fitting?')
    elif data == 'limits':
        try:
            return hdf5storage.loadmat(file_out_suff + '.limits.mat')['limits']
        except Exception:
            return None
    elif data == 'injected_spikes':
        try:
            spikes = hdf5storage.loadmat(data_file_noext + '/injected/spiketimes.mat')
            elecs  = numpy.load(data_file_noext + '/injected/elecs.npy')
            N_tm   = len(spikes)
            count  = 0
            result = {}
            for i in xrange(N_tm):
                key = 'temp_' + str(i)
                if len(spikes[key]) > 0:
                    result['spikes_' + str(elecs[count])] = spikes[key]
                    count += 1
            return result
        except Exception:
            return None


def save_data(params, data, extension=''):
    file_out = params.get('data', 'file_out')
    file     = h5py.File(file_out + '.hdf5' %extension)
    for key, value in data:
        file.create_dataset(key, data=value)
    file.close()


def collect_data(nb_threads, params, erase=False, with_real_amps=False, with_voltages=False):

    file_out_suff  = params.get('data', 'file_out_suff')
    templates      = load_data(params, 'templates')
    N_e, N_t, N_tm = templates.shape

    print "Gathering data from %d nodes..." %nb_threads

    result = {'spiketimes' : {}, 'amplitudes' : {}}
    if with_real_amps:
        result['real_amps'] = {}
    if with_voltages:
        result['voltages'] = {}    
    for i in xrange(N_tm/2):
        result['spiketimes']['temp_' + str(i)]  = numpy.empty(shape=0)
        result['amplitudes']['temp_' + str(i)]  = numpy.empty(shape=(0, 2))
        if with_real_amps:
            result['real_amps']['temp_' + str(i)] = numpy.empty(shape=0)
        if with_voltages:
            result['voltages']['temp_' + str(i)] = numpy.empty(shape=0)

    pbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()], maxval=nb_threads).start()

    for count, node in enumerate(xrange(nb_threads)):
        spiketimes_file = file(file_out_suff + '.spiketimes-%d.data' %node, 'r')
        amplitudes_file = file(file_out_suff + '.amplitudes-%d.data' %node, 'r')
        templates_file  = file(file_out_suff + '.templates-%d.data' %node, 'r')
        if with_real_amps:
            real_amps_file = file(file_out_suff + '.real_amps-%d.data' %node, 'r')
            real_amps      = numpy.fromfile(real_amps_file, dtype=numpy.float32)
        if with_voltages:
            voltages_file  = file(file_out_suff + '.voltages-%d.data' %node, 'r')
            voltages       = numpy.fromfile(voltages_file, dtype=numpy.float32)

        amplitudes = numpy.fromfile(amplitudes_file, dtype=numpy.float32)
        spiketimes = numpy.fromfile(spiketimes_file, dtype=numpy.int32)
        templates  = numpy.fromfile(templates_file, dtype=numpy.int32)
        N          = len(amplitudes)
        amplitudes = amplitudes.reshape(N/2, 2)
        min_size   = min([amplitudes.shape[0], spiketimes.shape[0], templates.shape[0]])
        amplitudes = amplitudes[:min_size]
        spiketimes = spiketimes[:min_size]
        templates  = templates[:min_size]
        if with_real_amps:
            real_amps = real_amps[:min_size]
        if with_voltages:
            voltages = voltages[:min_size]

        local_temp = numpy.unique(templates)

        for j in local_temp:
            idx = numpy.where(templates == j)[0]
            result['amplitudes']['temp_' + str(j)] = numpy.concatenate((amplitudes[idx], result['amplitudes']['temp_' + str(j)]))
            result['spiketimes']['temp_' + str(j)] = numpy.concatenate((result['spiketimes']['temp_' + str(j)], spiketimes[idx])) 
            if with_real_amps:
                result['real_amps']['temp_' + str(j)] = numpy.concatenate((result['real_amps']['temp_' + str(j)], real_amps[idx]))
            if with_voltages:
                result['voltages']['temp_' + str(j)] = numpy.concatenate((result['voltages']['temp_' + str(j)], voltages[idx])) 

        pbar.update(count)

    pbar.finish()

    for key in result['spiketimes']:
        result['spiketimes'][key] = numpy.array(result['spiketimes'][key], dtype=numpy.int32)
        idx                       = numpy.argsort(result['spiketimes'][key])
        result['amplitudes'][key] = numpy.array(result['amplitudes'][key], dtype=numpy.float32)
        result['spiketimes'][key] = result['spiketimes'][key][idx]
        result['amplitudes'][key] = result['amplitudes'][key][idx]
        if with_real_amps:
            result['real_amps'][key] = result['real_amps'][key][idx]
        if with_voltages:
            result['voltages'][key] = result['voltages'][key][idx]

    keys = ['spiketimes', 'amplitudes']
    if with_real_amps:
        keys += ['real_amps']
    if with_voltages:
        keys += ['voltages']
    for key in keys:
        if os.path.exists(file_out_suff + '.%s.mat' %key):
            purge(file_out_suff, '.%s.mat' %key)
        hdf5storage.savemat(file_out_suff + '.%s' %key, result[key])

    count = 0
    for item in result['spiketimes'].keys():
        count += len(result['spiketimes'][item])

    print_info(["Number of spikes fitted : %d" %count])

    if erase:
        purge(file_out_suff, '.data')

def get_results(params, extension=''):
    file_out_suff        = params.get('data', 'file_out_suff')
    result               = {}
    result['amplitudes'] = hdf5storage.loadmat(file_out_suff + '.amplitudes%s.mat' %extension)
    result['spiketimes'] = hdf5storage.loadmat(file_out_suff + '.spiketimes%s.mat' %extension)
    return result

def get_overlaps(params, extension='', erase=False):

    file_out_suff = params.get('data', 'file_out_suff')   
    N_e           = params.getint('data', 'N_e')
    N_t           = params.getint('data', 'N_t')    

    if os.path.exists(file_out_suff + '.overlap%s.hdf5' %extension) and not erase:
        return h5py.File(file_out_suff + '.overlap%s.hdf5' %extension).get('overlap')[:]
    else:
        if os.path.exists(file_out_suff + '.overlap%s.hdf5' %extension) and erase:
            os.remove(file_out_suff + '.overlap%s.hdf5' %extension)
            
        templates      = load_data(params, 'templates', extension=extension)
        N_e, N_t, N_tm = templates.shape
        cuda_string    = 'with 1 CPU...'

        try:
            import cudamat as cmt    
            HAVE_CUDA = True
            cmt.cuda_set_device(0)
            cmt.init()
            cmt.cuda_sync_threads()
            cuda_string    = 'with 1 GPU...'
        except Exception:
            HAVE_CUDA = False

        #print "Normalizing the templates..."
        norm_templates = numpy.sqrt(numpy.mean(numpy.mean(templates**2,0),0))
        templates     /= norm_templates

        print "Computing the templates overlaps", cuda_string
        pbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()], maxval=N_t).start()

        file = h5py.File(file_out_suff + '.overlap%s.hdf5' %extension, 'w')
        file.create_dataset('overlap', shape=(N_tm, N_tm, 2*N_t - 1), dtype=numpy.float32, chunks=True)

        for idelay in xrange(1, N_t+1):
            tmp_1 = templates[:, :idelay, :] 
            tmp_2 = templates[:, -idelay:, :]
            size  = N_e*idelay
            if HAVE_CUDA:
                tmp_1 = cmt.CUDAMatrix(tmp_1.reshape(size, N_tm))
                tmp_2 = cmt.CUDAMatrix(tmp_2.reshape(size, N_tm))
                data  = cmt.dot(tmp_1.T, tmp_2).asarray()
            else:
                tmp_1 = tmp_1.reshape(size, N_tm)
                tmp_2 = tmp_2.reshape(size, N_tm)
                data  = numpy.dot(tmp_1.T, tmp_2)

            file.get('overlap')[:, :, idelay-1]           = data
            file.get('overlap')[:, :, 2*N_t - idelay - 1] = numpy.transpose(data)
            pbar.update(idelay)
        file.close()
        pbar.finish()

        max_overlaps = numpy.zeros((N_tm, N_tm), dtype=numpy.float32)
        file = h5py.File(file_out_suff + '.overlap%s.hdf5' %extension)
        for i in xrange(N_tm):
            max_overlaps[i] = numpy.max(file.get('overlap')[i], 1)

        hdf5storage.savemat(file_out_suff + '.overlap.mat%s' %extension, {'maxoverlap' : max_overlaps})

        return h5py.File(file_out_suff + '.overlap%s.hdf5' %extension).get('overlap')[:]

