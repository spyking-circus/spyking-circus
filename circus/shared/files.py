import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)
import numpy, h5py, os, progressbar, platform, re, sys, scipy
import ConfigParser as configparser
import colorama
colorama.init(autoreset=True)
from colorama import Fore, Back, Style

from circus.shared.mpi import gather_array
from circus.shared.utils import smooth
import logging

def purge(file, pattern):
    dir = os.path.dirname(os.path.abspath(file))
    for f in os.listdir(dir):
        if f.find(pattern) > -1:
            os.remove(os.path.join(dir, f))

def set_logger(params):
    f_next, extension = os.path.splitext(params.get('data', 'data_file'))
    log_file          = f_next + '.log'
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', 
        filename=log_file,
        level=logging.DEBUG, 
        datefmt='%m/%d/%Y %I:%M:%S %p')

def write_to_logger(params, to_write, level='info'):
    set_logger(params)
    for line in to_write:
        if level == 'info':
            logging.info(line)
        elif level in ['debug', 'default']:
            logging.debug(line)
        elif level == 'warning':
            logging.warning(line)

def detect_header(filename, value='MCS'):

    if value == 'MCS':
        header      = 0
        stop        = False
        fid         = open(filename, 'rb')
        header_text = ''
        regexp      = re.compile('El_\d*')

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
        return header, len(regexp.findall(header_text))
    else:
        return value, None

def copy_header(header, file_in, file_out):
    fin  = open(file_in, 'r')
    fout = open(file_out, 'wb')
    data = fin.read(header)
    fout.write(data)
    fin.close()
    fout.close()


def get_multi_files(params):
    file_name   = params.get('data', 'data_multi_file')
    dirname     = os.path.abspath(os.path.dirname(file_name))
    all_files   = os.listdir(dirname)
    pattern     = os.path.basename(file_name)
    to_process  = []
    count       = 0

    while pattern in all_files:
        to_process += [os.path.join(os.path.abspath(dirname), pattern)]
        pattern     = pattern.replace(str(count), str(count+1))
        count      += 1

    print_and_log(['Multi-files:'] + to_process, 'debug', params)
    return to_process


def change_flag(file_name, flag, value, avoid_flag=None):
    f_next, extension = os.path.splitext(os.path.abspath(file_name))
    file_params       = os.path.abspath(file_name.replace(extension, '.params'))
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

def read_probe(parser):
    probe = {}
    if not os.path.exists(parser.get('data', 'mapping')):
        print_error(["The probe file can not be found"])
        sys.exit(0)
    try:
        probetext = open(parser.get('data', 'mapping'), 'r')
        exec probetext in probe
        probetext.close()
    except Exception:
        print_error(["Something wrong with the syntax of the probe file!"])
        sys.exit(0)

    key_flags = ['total_nb_channels', 'radius', 'channel_groups']
    for key in key_flags:
        if not probe.has_key(key):
            print_error(["%s is missing in the probe file" %key])
            sys.exit(0)
    return probe

def load_parameters(file_name):

    file_name         = os.path.abspath(file_name)
    f_next, extension = os.path.splitext(file_name)
    file_path         = os.path.dirname(file_name)
    file_params       = f_next + '.params'
    parser            = configparser.ConfigParser()
    if not os.path.exists(file_params):
        print_error(["%s does not exist" %file_params])
        sys.exit(0)
    parser.read(file_params)

    sections = ['data', 'whitening', 'extracting', 'clustering', 'fitting', 'filtering', 'merging', 'noedits']
    for section in sections:
        if parser.has_section(section):
            for (key, value) in parser.items(section):
                parser.set(section, key, value.split('#')[0].replace(' ', '').replace('\t', '')) 
        else:
            parser.add_section(section)

    N_t             = parser.getfloat('data', 'N_t')

    for key in ['whitening', 'clustering']:
        safety_time = parser.get(key, 'safety_time')
        if safety_time == 'auto':
            parser.set(key, 'safety_time', '%g' %(N_t/3.))

    sampling_rate   = parser.getint('data', 'sampling_rate')
    N_t             = int(sampling_rate*N_t*1e-3)
    if numpy.mod(N_t, 2) == 0:
        N_t += 1
    parser.set('data', 'N_t', str(N_t))
    parser.set('data', 'template_shift', str(int((N_t-1)/2)))

    data_offset              = parser.get('data', 'data_offset')
    if data_offset == 'MCS':
        parser.set('data', 'MCS', 'True')
    else:
        parser.set('data', 'MCS', 'False')
    data_offset, nb_channels = detect_header(file_name, data_offset)
    parser.set('data', 'data_offset', str(data_offset))
    
    probe = read_probe(parser)

    parser.set('data', 'N_total', str(probe['total_nb_channels']))   
    N_e = 0
    for key in probe['channel_groups'].keys():
        N_e += len(probe['channel_groups'][key]['channels'])

    if nb_channels is not None:
        if N_e != nb_channels:
            print_and_log(["MCS file: mistmatch between number of electrodes and data header"], 'error', parser)
            #sys.exit(0)

    parser.set('data', 'N_e', str(N_e))   
    parser.set('fitting', 'space_explo', '1')
    parser.set('fitting', 'nb_chances', '3')

    dtype_offset = parser.get('data', 'dtype_offset')
    if dtype_offset == 'auto':
        if parser.get('data', 'data_dtype') == 'uint16':
            parser.set('data', 'dtype_offset', '32767')
        elif parser.get('data', 'data_dtype') == 'int16':
            parser.set('data', 'dtype_offset', '0')
        elif parser.get('data', 'data_dtype') == 'float32':
            parser.set('data', 'dtype_offset', '0')
        elif parser.get('data', 'data_dtype') == 'int8':
            parser.set('data', 'dtype_offset', '0')        
        elif parser.get('data', 'data_dtype') == 'uint8':
            parser.set('data', 'dtype_offset', '127')

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
                  ['fitting', 'min_rate', 'float', '0'],
                  ['data', 'spikedetekt', 'bool', 'False'],
                  ['data', 'global_tmp', 'bool', 'True'],
                  ['data', 'chunk_size', 'int', '10'],
                  ['data', 'stationary', 'bool', 'True'],
                  ['data', 'alignment', 'bool', 'True'],
                  ['data', 'skip_artefact', 'bool', 'False'],
                  ['data', 'multi-files', 'bool', 'False'],
                  ['whitening', 'chunk_size', 'int', '60'],
                  ['clustering', 'max_clusters', 'int', '15'],
                  ['clustering', 'nb_repeats', 'int', '3'],
                  ['clustering', 'make_plots', 'bool', 'True'],
                  ['clustering', 'test_clusters', 'bool', 'False'],
                  ['clustering', 'sim_same_elec', 'float', '2'],
                  ['clustering', 'smart_search', 'float', '0'],
                  ['clustering', 'safety_space', 'bool', 'True'],
                  ['clustering', 'noise_thr', 'float', '0.8'],
                  ['clustering', 'cc_merge', 'float', '0.95'],
                  ['clustering', 'extraction', 'string', 'median-raw'],
                  ['clustering', 'remove_mixture', 'bool', 'True'],
                  ['extracting', 'cc_merge', 'float', '0.95'],
                  ['extracting', 'noise_thr', 'float', '1.'],
                  ['merging', 'cc_overlap', 'float', '0.25'],
                  ['merging', 'cc_bin', 'float', '2']]

    for item in new_values:
        section, name, val_type, value = item
        try:
            if val_type is 'bool':
                parser.getboolean(section, name)
            elif val_type is 'int':
                parser.getint(section, name)
            elif val_type is 'float':
                parser.getfloat(section, name)
            elif val_type is 'string':
                parser.get(section, name)
        except Exception:
            parser.set(section, name, value)
  
    chunk_size = parser.getint('data', 'chunk_size')
    parser.set('data', 'chunk_size', str(chunk_size*sampling_rate))
    chunk_size = parser.getint('whitening', 'chunk_size')
    parser.set('whitening', 'chunk_size', str(chunk_size*sampling_rate))

    test = (parser.get('clustering', 'extraction') in ['quadratic', 'median-raw', 'median-pca', 'mean-raw', 'mean-pca'])
    if not test:
        print_and_log(["Only 5 extraction modes: quadratic, median-raw, median-pca, mean-raw or mean-pca!"], 'error', parser)
        sys.exit(0)

    if parser.getboolean('data', 'multi-files'):
        parser.set('data', 'data_multi_file', file_name)
        pattern     = os.path.basename(file_name).replace('0', 'all')
        multi_file  = os.path.join(file_path, pattern)
        parser.set('data', 'data_file', multi_file)
        f_next, extension = os.path.splitext(multi_file)
    else:
        parser.set('data', 'data_file', file_name)

    try:
        os.makedirs(f_next)
    except Exception:
        pass

    file_out = os.path.join(f_next, os.path.basename(f_next))
    parser.set('data', 'file_out', file_out) # Output file without suffix
    parser.set('data', 'file_out_suff', file_out  + parser.get('data', 'suffix')) # Output file with suffix
    parser.set('data', 'data_file_noext', f_next)   # Data file (assuming .filtered at the end)
    parser.set('data', 'dist_peaks', str(N_t)) # Get only isolated spikes for a single electrode (whitening, clustering, basis)    

    for section in ['whitening', 'clustering']:
        test = (parser.getfloat(section, 'nb_elts') > 0) and (parser.getfloat(section, 'nb_elts') <= 1)
        if not test: 
            print_and_log(["nb_elts in %s should be in [0,1]" %section], 'error', parser)
            sys.exit(0)

    test = (parser.getfloat('clustering', 'nclus_min') >= 0) and (parser.getfloat('clustering', 'nclus_min') <= 1)
    if not test:
        print_and_log(["nclus_min in clustering should be in [0,1]"], 'error', parser)
        sys.exit(0)
 
    test = (parser.getfloat('clustering', 'smart_search') >= 0) and (parser.getfloat('clustering', 'smart_search') <= 1)
    if not test:
        print_and_log(["smart_search in clustering should be in [0,1]"], 'error', parser)
        sys.exit(0)

    test = (parser.getfloat('clustering', 'noise_thr') >= 0) and (parser.getfloat('clustering', 'noise_thr') <= 1)
    if not test:
        print_and_log(["noise_thr in clustering should be in [0,1]"], 'error', parser)
        sys.exit(0)

    return parser


def data_stats(params, show=True, export_times=False):
    data_file      = params.get('data', 'data_file')
    data_offset    = params.getint('data', 'data_offset')
    data_dtype     = params.get('data', 'data_dtype')
    N_total        = params.getint('data', 'N_total')
    N_e            = params.getint('data', 'N_e')
    sampling_rate  = params.getint('data', 'sampling_rate')
    multi_files    = params.getboolean('data', 'multi-files')
    chunk_len      = N_total * (60 * sampling_rate)
        
    if not multi_files:
        datablock      = numpy.memmap(data_file, offset=data_offset, dtype=data_dtype, mode='r')
        N              = len(datablock)
        nb_chunks      = N / chunk_len
        last_chunk_len = (N - nb_chunks * chunk_len)/(N_total*sampling_rate)
    else:
        all_files      = get_multi_files(params)
        N              = 0
        nb_chunks      = 0
        last_chunk_len = 0
        t_start        = 0
        times          = []
        for f in all_files:
            if params.getboolean('data', 'MCS'):
                data_offset, nb_channels = detect_header(f, 'MCS')
            datablock       = numpy.memmap(f, offset=data_offset, dtype=data_dtype, mode='r')
            loc_N           = len(datablock)
            loc_nb_chunks   = loc_N / chunk_len
            nb_chunks      += loc_nb_chunks
            last_chunk_len += (loc_N - loc_nb_chunks * chunk_len)/(N_total*sampling_rate)
            times   += [[t_start, t_start + len(datablock)/N_total]]
            t_start  = t_start + len(datablock)/N_total

    N_t = params.getint('data', 'N_t')
    N_t = numpy.round(1000.*N_t/sampling_rate, 1)

    nb_extra        = last_chunk_len/60
    nb_chunks      += nb_extra
    last_chunk_len -= nb_extra*60

    lines = ["Number of recorded channels : %d" %N_total,
             "Number of analyzed channels : %d" %N_e,
             "Data type                   : %s" %str(data_dtype),
             "Sampling rate               : %d kHz" %(sampling_rate/1000.),
             "Header offset for the data  : %d" %data_offset,
             "Duration of the recording   : %d min %s s" %(nb_chunks, last_chunk_len),
             "Width of the templates      : %d ms" %N_t,
             "Spatial radius considered   : %d um" %params.getint('data', 'radius'),
             "Stationarity                : %s" %params.getboolean('data', 'stationary'),
             "Waveform alignment          : %s" %params.getboolean('data', 'alignment'),
             "Skip strong artefacts       : %s" %params.getboolean('data', 'skip_artefact'),
             "Template Extraction         : %s" %params.get('clustering', 'extraction')]
    
    if multi_files:
        lines += ["Multi-files activated       : %s files" %len(all_files)]    

    print_and_log(lines, 'info', params, show)

    if not export_times:
        return nb_chunks*60 + last_chunk_len
    else:
        return times

def print_and_log(to_print, level='info', logger=None, display=True):
    if display:
        if level == 'default':
            for line in to_print:
                print line
        if level == 'info':
            print_info(to_print)
        elif level == 'error':
            print_error(to_print)

    if logger is not None:
        write_to_logger(logger, to_print, level)

def print_info(lines):
    print Fore.YELLOW + "-------------------------  Informations  -------------------------"
    for line in lines:
        print Fore.YELLOW + "| " + line
    print Fore.YELLOW + "------------------------------------------------------------------"

def print_error(lines):
    print Fore.RED + "----------------------------  Error  -----------------------------"
    for line in lines:
        print Fore.RED + "| " + line
    print Fore.RED + "------------------------------------------------------------------"


def get_stas(params, times_i, labels_i, src, neighs, nodes=None, mean_mode=False, all_labels=False):

    
    N_t          = params.getint('data', 'N_t')
    if not all_labels:
        if not mean_mode:
            stas = numpy.zeros((len(times_i), len(neighs), N_t), dtype=numpy.float32)
        else:
            stas = numpy.zeros((len(neighs), N_t), dtype=numpy.float32)
    else:
        nb_labels = numpy.unique(labels_i)
        stas      = numpy.zeros((len(nb_labels), len(neighs), N_t), dtype=numpy.float32)

    data_file    = params.get('data', 'data_file')
    data_offset  = params.getint('data', 'data_offset')
    dtype_offset = params.getint('data', 'dtype_offset')
    data_dtype   = params.get('data', 'data_dtype')
    N_total      = params.getint('data', 'N_total')
    alignment    = params.getboolean('data', 'alignment')
    datablock    = numpy.memmap(data_file, offset=data_offset, dtype=data_dtype, mode='r')

    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    template_shift        = params.getint('data', 'template_shift')

    if do_spatial_whitening:
        spatial_whitening  = load_data(params, 'spatial_whitening')
    if do_temporal_whitening:     
        temporal_whitening = load_data(params, 'temporal_whitening')

    if alignment:
        cdata = numpy.linspace(-template_shift, template_shift, 5*N_t)
        xdata = numpy.arange(-2*template_shift, 2*template_shift+1)

    count = 0
    for lb, time in zip(labels_i, times_i):
        padding      = N_total * time
        if alignment:
            local_chunk = datablock[padding - 2*template_shift*N_total:padding + (2*template_shift+1)*N_total]
            local_chunk = local_chunk.reshape(2*N_t - 1, N_total)
        else:
            local_chunk = datablock[padding - template_shift*N_total:padding + (template_shift+1)*N_total]
            local_chunk = local_chunk.reshape(N_t, N_total)

        local_chunk  = local_chunk.astype(numpy.float32)
        local_chunk -= dtype_offset
        
        if nodes is not None:
            if not numpy.all(nodes == numpy.arange(N_total)):
                local_chunk = local_chunk[:, nodes]
        if do_spatial_whitening:
            local_chunk = numpy.dot(local_chunk, spatial_whitening)
        if do_temporal_whitening:
            local_chunk = scipy.ndimage.filters.convolve1d(local_chunk, temporal_whitening, axis=0, mode='constant')

        local_chunk = local_chunk[:, neighs]

        if alignment:
            idx   = numpy.where(neighs == src)[0]
            ydata = numpy.arange(len(neighs))
            f     = scipy.interpolate.RectBivariateSpline(xdata, ydata, local_chunk, s=0)
            smoothed    = smooth(f(cdata, idx)[:, 0], template_shift)
            rmin        = (numpy.argmin(smoothed) - len(cdata)/2.)/5.
            ddata       = numpy.linspace(rmin-template_shift, rmin+template_shift, N_t)
            local_chunk = f(ddata, ydata).astype(numpy.float32)

        if all_labels:
            lc        = numpy.where(nb_labels == lb)[0]
            stas[lc] += local_chunk.T
        else:
            if not mean_mode:
                stas[count, :, :] = local_chunk.T
                count            += 1
            else:
                stas += local_chunk.T

    return stas

def get_amplitudes(params, times_i, src, neighs, template, nodes=None):

    N_t          = params.getint('data', 'N_t')
    amplitudes   = numpy.zeros(len(times_i), dtype=numpy.float32)
    data_file    = params.get('data', 'data_file')
    data_offset  = params.getint('data', 'data_offset')
    dtype_offset = params.getint('data', 'dtype_offset')
    data_dtype   = params.get('data', 'data_dtype')
    N_total      = params.getint('data', 'N_total')
    alignment    = params.getboolean('data', 'alignment')
    datablock    = numpy.memmap(data_file, offset=data_offset, dtype=data_dtype, mode='r')
    template     = template.flatten()
    covariance   = numpy.zeros((len(template), len(template)), dtype=numpy.float32)
    norm_temp    = numpy.sum(template**2)

    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = params.getboolean('whitening', 'spatial')
    template_shift        = params.getint('data', 'template_shift')

    if do_spatial_whitening:
        spatial_whitening  = load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = load_data(params, 'temporal_whitening')

    if alignment:
        cdata = numpy.linspace(-template_shift, template_shift, 5*N_t)
        xdata = numpy.arange(-2*template_shift, 2*template_shift+1)

    for count, time in enumerate(times_i):
        padding      = N_total * time
        if alignment:
            local_chunk = datablock[padding - 2*template_shift*N_total:padding + (2*template_shift+1)*N_total]
            local_chunk = local_chunk.reshape(2*N_t - 1, N_total)
        else:
            local_chunk = datablock[padding - template_shift*N_total:padding + (template_shift+1)*N_total]
            local_chunk = local_chunk.reshape(N_t, N_total)

        local_chunk  = local_chunk.astype(numpy.float32)
        local_chunk -= dtype_offset

        if nodes is not None:
            if not numpy.all(nodes == numpy.arange(N_total)):
                local_chunk = local_chunk[:, nodes]
        if do_spatial_whitening:
            local_chunk = numpy.dot(local_chunk, spatial_whitening)
        if do_temporal_whitening:
            local_chunk = scipy.ndimage.filters.convolve1d(local_chunk, temporal_whitening, axis=0, mode='constant')
        
        local_chunk = local_chunk[:, neighs]

        if alignment:
            idx   = numpy.where(neighs == src)[0]
            ydata = numpy.arange(len(neighs))
            f     = scipy.interpolate.RectBivariateSpline(xdata, ydata, local_chunk, s=0)
            smoothed    = smooth(f(cdata, idx)[:, 0], template_shift)
            rmin        = (numpy.argmin(smoothed) - len(cdata)/2.)/5.
            ddata = numpy.linspace(rmin-template_shift, rmin+template_shift, N_t)
            local_chunk = f(ddata, ydata).astype(numpy.float32)

        local_chunk       = local_chunk.T.flatten()
        amplitudes[count] = numpy.dot(local_chunk, template)/norm_temp
        snippet     = (template - amplitudes[count]*local_chunk).reshape(len(template), 1)
        covariance += numpy.dot(snippet, snippet.T)

    covariance  /= len(times_i)
    evals, evecs = scipy.sparse.linalg.eigs(covariance, k=1, which='LM')
    evecs        = numpy.real(evecs).astype(numpy.float32)
    return amplitudes, evecs.reshape(len(neighs), N_t)


def load_chunk(params, idx, chunk_len, chunk_size=None, padding=(0, 0), nodes=None):
    
    if chunk_size is None:
        chunk_size = params.getint('data', 'chunk_size')
    data_file    = params.get('data', 'data_file')
    data_offset  = params.getint('data', 'data_offset')
    if params.getboolean('data', 'MCS'):
        data_offset, nb_channels = detect_header(data_file, 'MCS')
    dtype_offset = params.getint('data', 'dtype_offset')
    data_dtype   = params.get('data', 'data_dtype')
    N_total      = params.getint('data', 'N_total')
    datablock    = numpy.memmap(data_file, offset=data_offset, dtype=data_dtype, mode='r')
    local_chunk  = datablock[idx*chunk_len+padding[0]:(idx+1)*chunk_len+padding[1]]
    del datablock
    local_shape  = chunk_size + (padding[1]-padding[0])/N_total
    local_chunk  = local_chunk.reshape(local_shape, N_total)
    local_chunk  = local_chunk.astype(numpy.float32)
    local_chunk -= dtype_offset
    if nodes is not None:
        if not numpy.all(nodes == numpy.arange(N_total)):
            local_chunk = local_chunk[:, nodes]
    return numpy.ascontiguousarray(local_chunk), local_shape


def prepare_preview(params, preview_filename):
    chunk_size   = 2*params.getint('data', 'sampling_rate')
    data_file    = params.get('data', 'data_file')
    data_offset  = params.getint('data', 'data_offset')
    dtype_offset = params.getint('data', 'dtype_offset')
    data_dtype   = params.get('data', 'data_dtype')
    N_total      = params.getint('data', 'N_total')
    datablock    = numpy.memmap(data_file, offset=data_offset, dtype=data_dtype, mode='r')
    chunk_len    = N_total * chunk_size
    local_chunk  = datablock[0:chunk_len]

    output = open(preview_filename, 'wb')
    fid    = open(data_file, 'r')
    # We copy the header 
    for i in xrange(data_offset):
        output.write(fid.read(1))
    
    fid.close()

    #Then the datafile
    local_chunk.tofile(output)
    output.close()

def analyze_data(params, chunk_size=None):

    if chunk_size is None:
        chunk_size = params.getint('data', 'chunk_size')
    data_file      = params.get('data', 'data_file')
    data_offset    = params.getint('data', 'data_offset')
    if params.getboolean('data', 'MCS'):
        data_offset, nb_channels = detect_header(data_file, 'MCS')
    data_dtype     = params.get('data', 'data_dtype')
    N_total        = params.getint('data', 'N_total')
    template_shift = params.getint('data', 'template_shift')
    chunk_len      = N_total * chunk_size
    borders        = N_total * template_shift
    datablock      = numpy.memmap(data_file, offset=data_offset, dtype=data_dtype, mode='r')
    N              = len(datablock)
    nb_chunks      = N / chunk_len
    last_chunk_len = N - nb_chunks * chunk_len
    last_chunk_len = N_total * int(last_chunk_len/N_total)
    
    return borders, nb_chunks, chunk_len, last_chunk_len

def get_nodes_and_edges(parameters):
    
    edges  = {}
    nodes  = []
    probe  = read_probe(parameters)
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

    file_out        = params.get('data', 'file_out')
    file_out_suff   = params.get('data', 'file_out_suff')
    data_file_noext = params.get('data', 'data_file_noext')

    if data == 'thresholds':
        spike_thresh = params.getfloat('data', 'spike_thresh')
        if os.path.exists(file_out + '.basis.hdf5'):
            myfile     = h5py.File(file_out + '.basis.hdf5', 'r', libver='latest')
            thresholds = myfile.get('thresholds')[:]
            myfile.close()
            return spike_thresh * thresholds 
    elif data == 'spatial_whitening':
        if os.path.exists(file_out + '.basis.hdf5'):
            myfile  = h5py.File(file_out + '.basis.hdf5', 'r', libver='latest')
            spatial = numpy.ascontiguousarray(myfile.get('spatial')[:])
            myfile.close()
            return spatial
        else:
            raise Exception('Whitening matrix has to be computed first!')
    elif data == 'temporal_whitening':
        if os.path.exists(file_out + '.basis.hdf5'):
            myfile   = h5py.File(file_out + '.basis.hdf5', 'r', libver='latest')
            temporal = myfile.get('temporal')[:]
            myfile.close() 
            return temporal
        else:
            raise Exception('Whitening matrix has to be computed first!')
    elif data == 'basis':
        myfile     = h5py.File(file_out + '.basis.hdf5', 'r', libver='latest')
        basis_proj = numpy.ascontiguousarray(myfile.get('proj')[:])
        basis_rec  = numpy.ascontiguousarray(myfile.get('rec')[:])
        myfile.close()
        return basis_proj, basis_rec
    elif data == 'waveforms':
        myfile     = h5py.File(file_out + '.basis.hdf5', 'r', libver='latest')
        waveforms  = myfile.get('waveforms')[:]
        myfile.close()
        return waveforms
    elif data == 'templates':
        N_e = params.getint('data', 'N_e')
        N_t = params.getint('data', 'N_t')
        if os.path.exists(file_out_suff + '.templates%s.hdf5' %extension):
            temp_x = h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r', libver='latest').get('temp_x')[:]
            temp_y = h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r', libver='latest').get('temp_y')[:]
            temp_data = h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r', libver='latest').get('temp_data')[:]
            nb_templates = h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r', libver='latest').get('norms').shape[0]
            return scipy.sparse.csc_matrix((temp_data, (temp_x, temp_y)), shape=(N_e*N_t, nb_templates))
        else:
            raise Exception('No templates found! Check suffix?')
    elif data == 'norm-templates':
        if os.path.exists(file_out_suff + '.templates%s.hdf5' %extension):
            return h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r', libver='latest').get('norms')[:]
        else:
            raise Exception('No templates found! Check suffix?')
    elif data == 'spike-cluster':
        file_name = params.get('data', 'data_file_noext') + '.spike-cluster.hdf5'
        if os.path.exists(file_name):
            data       = h5py.File(file_name, 'r')
            clusters   = data.get('clusters').flatten()
            N_clusters = len(numpy.unique(clusters))
            spiketimes = data.get('spikes').flatten()
            return clusters, spiketimes, N_clusters
        else:
            raise Exception('Need to provide a spike-cluster file!')
    elif data == 'spikedetekt':
        file_name = params.get('data', 'data_file_noext') + ".kwik"
        if os.path.exists(file_name):
            return h5py.File(file_name).get('channel_groups/1/spikes/time_samples', 'r', libver='latest')[:].astype(numpy.int32)
        else:
            raise Exception('No clusters found! Check suffix or run clustering?')
    elif data == 'clusters':
        if os.path.exists(file_out_suff + '.clusters%s.hdf5' %extension):
            myfile = h5py.File(file_out_suff + '.clusters%s.hdf5' %extension, 'r', libver='latest')
            result = {}
            for key in myfile.keys():
                result[str(key)] = myfile.get(key)[:]
            myfile.close()
            return result
        else:
            raise Exception('No clusters found! Check suffix or run clustering?')
    elif data == 'electrodes':
        if os.path.exists(file_out_suff + '.clusters%s.hdf5' %extension):
            myfile     = h5py.File(file_out_suff + '.clusters%s.hdf5' %extension, 'r', libver='latest')
            electrodes = myfile.get('electrodes')[:]
            myfile.close()
            return electrodes
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
        if os.path.exists(file_out_suff + '.templates%s.hdf5' %extension):
            myfile = h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r', libver='latest')
            limits = myfile.get('limits')[:]
            myfile.close()
            return limits
        else:
            raise Exception('No templates found! Check suffix?')
    elif data == 'injected_spikes':
        try:
            spikes = h5py.File(data_file_noext + '/injected/result.hdf5').get('spiketimes')
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

def write_datasets(h5file, to_write, result, electrode=None):
    for key in to_write:
        if electrode is not None:
            mykey = key + str(electrode)
        else:
            mykey = key
        h5file.create_dataset(mykey, shape=result[mykey].shape, dtype=result[mykey].dtype, chunks=True)
        h5file.get(mykey)[:] = result[mykey]

def collect_data(nb_threads, params, erase=False, with_real_amps=False, with_voltages=False):

    file_out_suff  = params.get('data', 'file_out_suff')
    min_rate       = params.get('fitting', 'min_rate')
    N_e            = params.getint('data', 'N_e')
    N_t            = params.getint('data', 'N_t')
    duration       = data_stats(params, show=False)
    templates      = load_data(params, 'templates')
    sampling_rate  = params.getint('data', 'sampling_rate')
    refractory     = int(0.5*sampling_rate*1e-3)
    x, N_tm        = templates.shape

    print_and_log(["Gathering data from %d nodes..." %nb_threads], 'default', params)

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
        spiketimes_file = file_out_suff + '.spiketimes-%d.data' %node
        amplitudes_file = file_out_suff + '.amplitudes-%d.data' %node
        templates_file  = file_out_suff + '.templates-%d.data' %node
        if with_real_amps:
            real_amps_file = file_out_suff + '.real_amps-%d.data' %node
            real_amps      = numpy.fromfile(real_amps_file, dtype=numpy.float32)
        if with_voltages:
            voltages_file  = file_out_suff + '.voltages-%d.data' %node
            voltages       = numpy.fromfile(voltages_file, dtype=numpy.float32)

        if os.path.exists(amplitudes_file):

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

        if refractory > 0:
            violations = numpy.where(numpy.diff(result['spiketimes'][key]) <= refractory)[0] + 1
            result['spiketimes'][key] = numpy.delete(result['spiketimes'][key], violations)
            result['amplitudes'][key] = numpy.delete(result['amplitudes'][key], violations, axis=0)
            if with_real_amps:
                result['real_amps'][key] = numpy.delete(result['real_amps'][key], violations)
            if with_voltages:
                result['voltages'][key] = numpy.delete(result['voltages'][key], violations)

    keys = ['spiketimes', 'amplitudes']
    if with_real_amps:
        keys += ['real_amps']
    if with_voltages:
        keys += ['voltages']

    mydata = h5py.File(file_out_suff + '.result.hdf5', 'w', libver='latest')
    for key in keys:
        mydata.create_group(key)
        for temp in result[key].keys():
            tmp_path = '%s/%s' %(key, temp)
            mydata.create_dataset(tmp_path, data=result[key][temp])
    mydata.close()        
    
    count = 0
    for item in result['spiketimes'].keys():
        count += len(result['spiketimes'][item])

    print_and_log(["Number of spikes fitted : %d" %count], 'info', params)

    if erase:
        purge(file_out_suff, '.data')

def get_results(params, extension=''):
    file_out_suff        = params.get('data', 'file_out_suff')
    result               = {}
    myfile               = h5py.File(file_out_suff + '.result%s.hdf5' %extension, 'r', libver='latest')
    for key in myfile.keys():
        result[str(key)] = {}
        for temp in myfile.get(key).keys():
            result[str(key)][str(temp)] = myfile.get(key).get(temp)[:]
    myfile.close()
    return result

def get_overlaps(comm, params, extension='', erase=False, parallel_hdf5=False, normalize=True, maxoverlap=True, verbose=True, half=False):

    file_out_suff  = params.get('data', 'file_out_suff')   
    tmp_path       = os.path.join(os.path.abspath(params.get('data', 'data_file_noext')), 'tmp')
    if maxoverlap:
        templates  = load_data(params, 'templates', extension=extension)
    else:
        templates  = load_data(params, 'templates')
    filename       = file_out_suff + '.overlap%s.hdf5' %extension
    if extension == '-merged':
        best_elec  = load_data(params, 'electrodes', extension)
    else:
        best_elec  = load_data(params, 'electrodes')
    N_total        = params.getint('data', 'N_total')
    nodes, edges   = get_nodes_and_edges(params)
    N_e            = params.getint('data', 'N_e')
    N_t            = params.getint('data', 'N_t')
    x,        N_tm = templates.shape

    if half:
        N_tm /= 2

    if os.path.exists(filename) and not erase:
        return h5py.File(filename, 'r')
    else:
        if os.path.exists(filename) and erase and (comm.rank == 0):
            os.remove(filename)
    
    comm.Barrier()
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)

    cuda_string = 'using %d CPU...' %comm.size
    
    try:
        import cudamat as cmt
        HAVE_CUDA = True
        if parallel_hdf5:
            if nb_gpu > nb_cpu:
                gpu_id = int(comm.rank/nb_cpu)
            else:
                gpu_id = 0
        else:
            gpu_id = 0
        cmt.cuda_set_device(gpu_id)
        cmt.init()
        cmt.cuda_sync_threads()
    except Exception:
        HAVE_CUDA = False

    if HAVE_CUDA:
        cuda_string = 'using %d GPU...' %comm.size

    #print "Normalizing the templates..."
    if normalize:
        norm_templates = load_data(params, 'norm-templates')[:N_tm]

    all_delays      = numpy.arange(1, N_t+1)

    local_templates = numpy.zeros(0, dtype=numpy.int32)
    for ielec in range(comm.rank, N_e, comm.size):
        local_templates = numpy.concatenate((local_templates, numpy.where(best_elec == ielec)[0]))

    if half:
        nb_total     = len(local_templates)
        upper_bounds = N_tm
    else:
        nb_total     = 2*len(local_templates)
        upper_bounds = N_tm/2

    if comm.rank == 0:
        if verbose:
            print_and_log(["Computing the overlaps %s" %cuda_string], 'default', params)
        N_0  = len(range(comm.rank, N_e, comm.size))
        pbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()], maxval=N_0).start()

    over_x    = numpy.zeros(0, dtype=numpy.int32)
    over_y    = numpy.zeros(0, dtype=numpy.int32)
    over_data = numpy.zeros(0, dtype=numpy.float32)
    
    for count, ielec in enumerate(range(comm.rank, N_e, comm.size)):
        
        local_idx = numpy.where(best_elec == ielec)[0]
        len_local = len(local_idx)

        if not half:
            local_idx = numpy.concatenate((local_idx, local_idx + upper_bounds))

        if len_local > 0:

            loc_templates = templates[:, local_idx].toarray().reshape(N_e, N_t, len(local_idx))
            to_consider   = numpy.arange(upper_bounds)
            if not half:
                to_consider = numpy.concatenate((to_consider, to_consider + upper_bounds))
            
            nb_elements = loc_templates.shape[2]
            
            if normalize:
                loc_templates /= norm_templates[local_idx]
            
            for idelay in all_delays:

                size  = N_e*idelay    
                tmp_1 = loc_templates[:, :idelay]
                
                if HAVE_CUDA:
                    tmp_1 = cmt.CUDAMatrix(tmp_1.reshape(size, nb_elements))
                else:
                    tmp_1 = tmp_1.reshape(size, nb_elements)
                
                loc_templates2 = templates[:, to_consider].toarray().reshape(N_e, N_t, len(to_consider))
                tmp_2          = loc_templates2[:, -idelay:, :]
                if normalize:
                    tmp_2 /= norm_templates[to_consider]

                lb_2  = tmp_2.shape[2]
                
                if HAVE_CUDA:
                    tmp_2 = cmt.CUDAMatrix(tmp_2.reshape(size, lb_2))
                    data  = cmt.dot(tmp_1.T, tmp_2).asarray()
                else:
                    tmp_2 = tmp_2.reshape(size, lb_2)
                    data  = numpy.dot(tmp_1.T, tmp_2).reshape(nb_elements, lb_2)

                dx, dy     = data.nonzero()
                ddx        = local_idx[dx].astype(numpy.int32)
                ddy        = to_consider[dy].astype(numpy.int32)
                data       = data.flatten()
                dd         = data.nonzero()[0].astype(numpy.int32)
                over_x     = numpy.concatenate((over_x, ddx*N_tm + ddy))
                over_y     = numpy.concatenate((over_y, (idelay-1)*numpy.ones(len(dx), dtype=numpy.int32)))
                over_data  = numpy.concatenate((over_data, data[dd]))
                if idelay < N_t:
                    over_x     = numpy.concatenate((over_x, ddy*N_tm + ddx))
                    over_y     = numpy.concatenate((over_y, (2*N_t-idelay-1)*numpy.ones(len(dx), dtype=numpy.int32)))
                    over_data  = numpy.concatenate((over_data, data[dd]))

        if comm.rank == 0:
            pbar.update(count)

    if comm.rank == 0:
        pbar.finish()
        print_and_log(["Overlaps computed, now gathering data by MPI"], 'debug', params)

    comm.Barrier()

    #We need to gather the sparse arrays
    over_x    = gather_array(over_x, comm, dtype='int32')            
    over_y    = gather_array(over_y, comm, dtype='int32')
    over_data = gather_array(over_data, comm)

    #We need to add the transpose matrices

    if comm.rank == 0:
        hfile      = h5py.File(filename, 'w', libver='latest')
        hfile.create_dataset('over_x', data=over_x)
        hfile.create_dataset('over_y', data=over_y)
        hfile.create_dataset('over_data', data=over_data)
        hfile.create_dataset('over_shape', data=numpy.array([N_tm**2, 2*N_t - 1], dtype=numpy.int32))
        hfile.close()

        if maxoverlap:

            overlap    = scipy.sparse.csr_matrix((over_data, (over_x, over_y)), shape=(N_tm**2, 2*N_t - 1))
            myfile     = h5py.File(filename, 'r+', libver='latest')
            myfile2    = h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r+', libver='latest')
            if 'maxoverlap' in myfile2.keys():
                maxoverlap = myfile2.get('maxoverlap')
            else:
                maxoverlap = myfile2.create_dataset('maxoverlap', shape=(N_tm, N_tm), dtype=numpy.float32)
            for i in xrange(N_tm-1):
                rows                = numpy.arange(i*N_tm+i+1, (i+1)*N_tm)
                maxoverlap[i, i+1:] = numpy.max(overlap[rows, :].toarray(), 1)
                maxoverlap[i+1:, i] = maxoverlap[i, i+1:]
            myfile.close()  
            myfile2.close()

    comm.Barrier()
    return h5py.File(filename, 'r')
