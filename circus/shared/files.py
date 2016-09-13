from __future__ import division

import warnings

from circus.shared.utils import get_progressbar

warnings.simplefilter(action = "ignore", category = FutureWarning)
import numpy, h5py, os, platform, re, sys, scipy
import ConfigParser as configparser
import sys
from colorama import Fore
from mpi import all_gather_array
from mpi4py import MPI
from .mpi import gather_array
import logging

from circus.files.datafile import *


def get_data_file(params):
    
    data_type = params.get('data', 'data_type')
    data_file = params.get('data', 'data_file')

    if data_type == 'raw_binary':
        return RawBinaryFile(data_file, params)
    elif data_type == 'mcs_raw_binary':
        return RawMCSFile(data_file, params) 
    elif data_type == 'hdf5':
        return H5File(data_file, params)
    else:
        print_error(['The type %s is not recognized as a valid file format' %data_type])


def get_header():

    import circus
    version = circus.__version__

    if len(version) == 3:
        title = '#####            Welcome to the SpyKING CIRCUS (%s)         #####' %version
    elif len(version) == 5:
        title = '#####           Welcome to the SpyKING CIRCUS (%s)        #####' %version

    header = '''
##################################################################
%s
#####                                                        #####
#####              Written by P.Yger and O.Marre             #####
##################################################################

''' %title

    return header

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
    """Set a new value to a flag of a given parameter file."""
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
    filename = os.path.abspath(os.path.expanduser(parser.get('data', 'mapping')))
    if not os.path.exists(filename):
        print_error(["The probe file can not be found"])
        sys.exit(0)
    try:
        with open(filename, 'r') as f:
            probetext = f.read()
            exec(probetext, probe)
    except Exception as ex:
        print_error(["Something wrong with the syntax of the probe file:\n" + str(ex)])
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

    sections = ['data', 'whitening', 'extracting', 'clustering', 'fitting', 'filtering', 'merging', 'noedits', 'triggers', 'detection', 'validating', 'converting']

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
            parser.set(key, 'safety_time', '%g' %(N_t//3.))

    sampling_rate   = parser.getint('data', 'sampling_rate')
    N_t             = int(sampling_rate*N_t*1e-3)
    if numpy.mod(N_t, 2) == 0:
        N_t += 1
    parser.set('data', 'N_t', str(N_t))
    parser.set('data', 'template_shift', str(int((N_t-1)//2)))

    data_offset = parser.get('data', 'data_offset')
    if data_offset == 'MCS':
        parser.set('data', 'MCS', 'True')
    else:
        parser.set('data', 'MCS', 'False')
   
    probe = read_probe(parser)

    parser.set('data', 'N_total', str(probe['total_nb_channels']))   
    N_e = 0
    for key in probe['channel_groups'].keys():
        N_e += len(probe['channel_groups'][key]['channels'])

    parser.set('data', 'N_e', str(N_e))   
    parser.set('fitting', 'space_explo', '0.5')
    parser.set('fitting', 'nb_chances', '3')
    parser.set('clustering', 'm_ratio', '0.01')
    parser.set('clustering', 'sub_dim', '5')

    try: 
        parser.get('data', 'radius')
    except Exception:
        parser.set('data', 'radius', 'auto')
    try:
        parser.getint('data', 'radius')
    except Exception:
        parser.set('data', 'radius', str(int(probe['radius'])))

    new_values = [['fitting', 'amp_auto', 'bool', 'True'], 
                  ['fitting', 'refractory', 'float', '0.5'],
                  ['data', 'global_tmp', 'bool', 'True'],
                  ['data', 'chunk_size', 'int', '10'],
                  ['data', 'multi-files', 'bool', 'False'],
                  ['data', 'data_type', 'string', 'raw_binary'],
                  ['detection', 'alignment', 'bool', 'True'],
                  ['detection', 'matched-filter', 'bool', 'False'],
                  ['detection', 'matched_thresh', 'float', '5'],
                  ['detection', 'peaks', 'string', 'negative'],
                  ['detection', 'spike_thresh', 'float', '6'],
                  ['triggers', 'clean_artefact', 'bool', 'False'],
                  ['triggers', 'make_plots', 'string', 'png'],
                  ['triggers', 'trig_file', 'string', ''],
                  ['triggers', 'trig_windows', 'string', ''],
                  ['whitening', 'chunk_size', 'int', '60'],
                  ['filtering', 'remove_median', 'bool', 'False'],
                  ['clustering', 'max_clusters', 'int', '10'],
                  ['clustering', 'nb_repeats', 'int', '3'],
                  ['clustering', 'make_plots', 'string', 'png'],
                  ['clustering', 'test_clusters', 'bool', 'False'],
                  ['clustering', 'sim_same_elec', 'float', '2'],
                  ['clustering', 'smart_search', 'bool', 'False'],
                  ['clustering', 'safety_space', 'bool', 'True'],
                  ['clustering', 'compress', 'bool', 'True'],
                  ['clustering', 'noise_thr', 'float', '0.8'],
                  ['clustering', 'cc_merge', 'float', '0.975'],
                  ['clustering', 'extraction', 'string', 'median-raw'],
                  ['clustering', 'remove_mixture', 'bool', 'True'],
                  ['clustering', 'dispersion', 'string', '(5, 5)'],
                  ['extracting', 'cc_merge', 'float', '0.95'],
                  ['extracting', 'noise_thr', 'float', '1.'],
                  ['merging', 'cc_overlap', 'float', '0.5'],
                  ['merging', 'cc_bin', 'float', '2'],
                  ['merging', 'correct_lag', 'bool', 'False'],
                  ['converting', 'export_pcs', 'string', 'prompt'],
                  ['converting', 'erase_all', 'bool', 'True'],
                  ['validating', 'nearest_elec', 'string', 'auto'],
                  ['validating', 'max_iter', 'int', '200'],
                  ['validating', 'learning_rate', 'float', '1.0e-3'],
                  ['validating', 'roc_sampling', 'int', '10'],
                  ['validating', 'make_plots', 'string', 'png'],
                  ['validating', 'test_size', 'float', '0.3'],
                  ['validating', 'radius_factor', 'float', '0.5'],
                  ['validating', 'juxta_dtype', 'string', 'uint16'],
                  ['validating', 'juxta_thresh', 'float', '6.0'],
                  ['validating', 'juxta_valley', 'bool', 'False'],
                  ['validating', 'matching_jitter', 'float', '2.0']]

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
  
    if parser.getboolean('data', 'multi-files'):
        parser.set('data', 'data_multi_file', file_name)
        pattern     = os.path.basename(file_name).replace('0', 'all')
        multi_file  = os.path.join(file_path, pattern)
        parser.set('data', 'data_file', multi_file)
        f_next, extension = os.path.splitext(multi_file)
    else:
        parser.set('data', 'data_file', file_name)

    if parser.getboolean('triggers', 'clean_artefact'):
        if (parser.get('triggers', 'trig_file') == '') or (parser.get('triggers', 'trig_windows') == ''):
            print_and_log(["trig_file and trig_windows must be specified"], 'error', parser)
            sys.exit(0)
    
    parser.set('triggers', 'trig_file', os.path.abspath(os.path.expanduser(parser.get('triggers', 'trig_file'))))
    parser.set('triggers', 'trig_windows', os.path.abspath(os.path.expanduser(parser.get('triggers', 'trig_windows'))))

    chunk_size = parser.getint('data', 'chunk_size')
    parser.set('data', 'chunk_size', str(chunk_size*sampling_rate))
    chunk_size = parser.getint('whitening', 'chunk_size')
    parser.set('whitening', 'chunk_size', str(chunk_size*sampling_rate))

    test = (parser.get('clustering', 'extraction') in ['median-raw', 'median-pca', 'mean-raw', 'mean-pca'])
    if not test:
        print_and_log(["Only 5 extraction modes: median-raw, median-pca, mean-raw or mean-pca!"], 'error', parser)
        sys.exit(0)

    test = (parser.get('detection', 'peaks') in ['negative', 'positive', 'both'])
    if not test:
        print_and_log(["Only 3 detection modes for peaks: negative, positive, both"], 'error', parser)
        sys.exit(0)

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

    test = (parser.getfloat('clustering', 'nclus_min') >= 0) and (parser.getfloat('clustering', 'nclus_min') < 1)
    if not test:
        print_and_log(["nclus_min in clustering should be in [0,1["], 'error', parser)
        sys.exit(0)

    test = (parser.getfloat('clustering', 'noise_thr') >= 0) and (parser.getfloat('clustering', 'noise_thr') <= 1)
    if not test:
        print_and_log(["noise_thr in clustering should be in [0,1]"], 'error', parser)
        sys.exit(0)

    test = (parser.getfloat('validating', 'test_size') > 0) and (parser.getfloat('validating', 'test_size') < 1)
    if not test:
        print_and_log(["test_size in validating should be in ]0,1["], 'error', parser)
        sys.exit(0)

    fileformats = ['png', 'pdf', 'eps', 'jpg', '', 'None']
    test = parser.get('clustering', 'make_plots') in fileformats
    if not test:
        print_and_log(["make_plots in clustering should be in %s" %str(fileformats)], 'error', parser)
        sys.exit(0)
    test = parser.get('validating', 'make_plots') in fileformats
    if not test:
        print_and_log(["make_plots in clustering should be in %s" %str(fileformats)], 'error', parser)
        sys.exit(0)
    
    dispersion     = parser.get('clustering', 'dispersion').replace('(', '').replace(')', '').split(',')
    dispersion     = map(float, dispersion)
    test =  (0 < dispersion[0]) and (0 < dispersion[1])
    if not test:
        print_and_log(["min and max dispersions should be positive"], 'error', parser)
        sys.exit(0)
        

    pcs_export = ['prompt', 'none', 'all', 'some']
    test = parser.get('converting', 'export_pcs') in pcs_export
    if not test:
        print_and_log(["export_pcs in converting should be in %s" %str(pcs_export)], 'error', parser)
        sys.exit(0)
    else:
        if parser.get('converting', 'export_pcs') == 'none':
            parser.set('converting', 'export_pcs', 'n')
        elif parser.get('converting', 'export_pcs') == 'some':
            parser.set('converting', 'export_pcs', 's')
        elif parser.get('converting', 'export_pcs') == 'all':
            parser.set('converting', 'export_pcs', 'a')
    

    return parser


def data_stats(data_file, show=True, export_times=False):
    multi_files    = data_file.params.getboolean('data', 'multi-files')
    chunk_size     = 60 * data_file.rate    

    if not multi_files:
        _, nb_chunks, chunk_len, last_chunk_len = data_file.analyze(chunk_size)
    else:
        all_files      = get_multi_files(data_file.params)
        N              = 0
        nb_chunks      = 0
        last_chunk_len = 0
        t_start        = 0
        times          = []
        for f in all_files:
            data_file.params.set('data', 'data_file', f)
            data   = get_data_file(data_file.params)
            _, loc_nb_chunks, chunk_len, last_chunk_len = data_file.analyze(chunk_size)

            nb_chunks      += loc_nb_chunks
            last_chunk_len += data.max_offset - (loc_nb_chunks*data_file.rate)

            times   += [[t_start, t_start + data.max_offset]]
            t_start += data.max_offset

    N_t = data_file.params.getint('data', 'N_t')
    N_t = numpy.round(1000.*N_t/data_file.rate, 1)

    nb_extra        = last_chunk_len//data_file.rate
    nb_chunks      += nb_extra
    last_chunk_len -= (nb_extra*data_file.rate)
    last_chunk_len  = int(last_chunk_len/data_file.rate)

    lines = ["Number of recorded channels : %d" %data_file.N_tot,
             "Number of analyzed channels : %d" %data_file.N_e,
             "Data format                 : %s" %data_file.params.get('data', 'data_type'),
             "Data type                   : %s" %str(data_file.data_dtype),
             "Sampling rate               : %d kHz" %(data_file.rate//1000.),
             "Header offset for the data  : %d" %data_file.data_offset,
             "Duration of the recording   : %d min %s s" %(nb_chunks, last_chunk_len),
             "Width of the templates      : %d ms" %N_t,
             "Spatial radius considered   : %d um" %data_file.params.getint('data', 'radius'),
             "Threshold crossing          : %s" %data_file.params.get('detection', 'peaks'),
             "Waveform alignment          : %s" %data_file.params.getboolean('detection', 'alignment'),
             "Matched filters             : %s" %data_file.params.getboolean('detection', 'matched-filter'),
             "Template Extraction         : %s" %data_file.params.get('clustering', 'extraction'),
             "Smart Search                : %s" %data_file.params.getboolean('clustering', 'smart_search')]
    
    if multi_files:
        lines += ["Multi-files activated       : %s files" %len(all_files)]    

    print_and_log(lines, 'info', data_file.params, show)

    if not export_times:
        return nb_chunks*60 + last_chunk_len
    else:
        return times

def print_and_log(to_print, level='info', logger=None, display=True):
    if display:
        if level == 'default':
            for line in to_print:
                print Fore.WHITE + line + '\r'
        if level == 'info':
            print_info(to_print)
        elif level == 'error':
            print_error(to_print)

    if logger is not None:
        write_to_logger(logger, to_print, level)

    sys.stdout.flush()


def print_info(lines):
    """Prints informations messages, enhanced graphical aspects."""
    print Fore.YELLOW + "-------------------------  Informations  -------------------------\r"
    for line in lines:
        print Fore.YELLOW + "| " + line + '\r'
    print Fore.YELLOW + "------------------------------------------------------------------\r"

def print_error(lines):
    """Prints errors messages, enhanced graphical aspects."""
    print Fore.RED + "----------------------------  Error  -----------------------------\r"
    for line in lines:
        print Fore.RED + "| " + line + '\r'
    print Fore.RED + "------------------------------------------------------------------\r"



def get_stas(data_file, times_i, labels_i, src, neighs, nodes=None, mean_mode=False, all_labels=False, pos='neg', auto_align=True):

    N_t          = data_file.params.getint('data', 'N_t')
    if not all_labels:
        if not mean_mode:
            stas = numpy.zeros((len(times_i), len(neighs), N_t), dtype=numpy.float32)
        else:
            stas = numpy.zeros((len(neighs), N_t), dtype=numpy.float32)
    else:
        nb_labels = numpy.unique(labels_i)
        stas      = numpy.zeros((len(nb_labels), len(neighs), N_t), dtype=numpy.float32)

    alignment     = data_file.params.getboolean('detection', 'alignment') and auto_align

    do_temporal_whitening = data_file.params.getboolean('whitening', 'temporal')
    do_spatial_whitening  = data_file.params.getboolean('whitening', 'spatial')
    template_shift        = data_file.params.getint('data', 'template_shift')

    if do_spatial_whitening:
        spatial_whitening  = load_data(data_file.params, 'spatial_whitening')
    if do_temporal_whitening:     
        temporal_whitening = load_data(data_file.params, 'temporal_whitening')

    if alignment:
        cdata = numpy.linspace(-template_shift, template_shift, 5*N_t)
        xdata = numpy.arange(-2*template_shift, 2*template_shift+1)

    count = 0
    for lb, time in zip(labels_i, times_i):
        if alignment:
            local_chunk = data_file.get_snippet(time - 2*template_shift, 2*N_t - 1, nodes=nodes)
        else:
            local_chunk = data_file.get_snippet(time - template_shift, N_t, nodes=nodes)
        
        if do_spatial_whitening:
            local_chunk = numpy.dot(local_chunk, spatial_whitening)
        if do_temporal_whitening:
            local_chunk = scipy.ndimage.filters.convolve1d(local_chunk, temporal_whitening, axis=0, mode='constant')

        local_chunk = numpy.take(local_chunk, neighs, axis=1)

        if alignment:
            idx   = numpy.where(neighs == src)[0]
            ydata = numpy.arange(len(neighs))
            if len(ydata) == 1:
                f           = scipy.interpolate.UnivariateSpline(xdata, local_chunk, s=0)
                if pos == 'neg':
                    rmin    = (numpy.argmin(f(cdata)) - len(cdata)/2.)/5.
                elif pos =='pos':
                    rmin    = (numpy.argmax(f(cdata)) - len(cdata)/2.)/5.
                ddata       = numpy.linspace(rmin-template_shift, rmin+template_shift, N_t)
                local_chunk = f(ddata).astype(numpy.float32).reshape(N_t, 1)
            else:
                f           = scipy.interpolate.RectBivariateSpline(xdata, ydata, local_chunk, s=0, ky=min(len(ydata)-1, 3))
                if pos == 'neg':
                    rmin    = (numpy.argmin(f(cdata, idx)[:, 0]) - len(cdata)/2.)/5.
                elif pos == 'pos':
                    rmin    = (numpy.argmax(f(cdata, idx)[:, 0]) - len(cdata)/2.)/5.
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


def get_stas_memshared(data_file, comm, times_i, labels_i, src, neighs, nodes=None,
                       mean_mode=False, all_labels=False, auto_align=True):
    
    # First we need to identify machines in the MPI ring.
    from uuid import getnode as get_mac
    myip = numpy.int64(get_mac()) % 100000
    ##### TODO: remove quarantine zone
    # intsize = MPI.INT.Get_size()
    ##### end quarantine zone
    float_size = MPI.FLOAT.Get_size() 
    sub_comm = comm.Split(myip, 0)
    params = data_file.params

    # Load parameters.
    N_t = params.getint('data', 'N_t')
    data_file = params.get('data', 'data_file')
    data_offset = params.getint('data', 'data_offset')
    dtype_offset = params.getint('data', 'dtype_offset')
    data_dtype = params.get('data', 'data_dtype')
    N_total = params.getint('data', 'N_total')
    alignment = params.getboolean('detection', 'alignment') and auto_align
    datablock = numpy.memmap(data_file, offset=data_offset, dtype=data_dtype, mode='r')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening = params.getboolean('whitening', 'spatial')
    template_shift = params.getint('data', 'template_shift')
    
    # Calculate the sizes of the data structures to share.
    nb_triggers = 0
    nb_neighs = 0
    nb_ts = 0
    if sub_comm.Get_rank() == 0:
        if not all_labels:
            if not mean_mode:
                ##### TODO: clean quarantine zone
                # nb_times = len(times_i)
                ##### end quarantine zone
                nb_triggers = len(times_i)
            else:
                nb_triggers = 1
        else:
            ##### TODO: remove quarantine zone
            # nb_labels = len(numpy.unique(labels_i))
            ##### end quarantine zone
            nb_triggers = len(numpy.unique(labels_i))
        nb_neighs = len(neighs)
        nb_ts = N_t
    
    sub_comm.Barrier()
    
    # Broadcast the sizes of the data structures to share.
    triggers_size = numpy.int64(sub_comm.bcast(numpy.array([nb_triggers], dtype=numpy.int32), root=0)[0])
    neighs_size = numpy.int64(sub_comm.bcast(numpy.array([nb_neighs], dtype=numpy.int32), root=0)[0])
    ts_size = numpy.int64(sub_comm.bcast(numpy.array([nb_ts], dtype=numpy.int32), root=0)[0])
    
    # Declare the data structures to share.
    if sub_comm.Get_rank() == 0:
        stas_bytes = triggers_size * neighs_size * ts_size * float_size
    else:
        stas_bytes = 0
    if triggers_size == 1:
        stas_shape = (neighs_size, ts_size)
    else:
        stas_shape = (triggers_size, neighs_size, ts_size)
    
    win_stas = MPI.Win.Allocate_shared(stas_bytes, float_size, comm=sub_comm)
    buf_stas, _ = win_stas.Shared_query(0)
    buf_stas = numpy.array(buf_stas, dtype='B', copy=False)
    stas = numpy.ndarray(buffer=buf_stas, dtype=numpy.float32, shape=stas_shape)
    
    sub_comm.Barrier()
    
    # Let master node initialize the data structures to share.
    if sub_comm.Get_rank() == 0:
        if do_spatial_whitening:
            spatial_whitening = load_data(params, 'spatial_whitening')
        if do_temporal_whitening:
            temporal_whitening = load_data(params, 'temporal_whitening')
        if alignment:
            cdata = numpy.linspace(- template_shift, template_shift, 5 * N_t)
            xdata = numpy.arange(- 2 * template_shift, 2 * template_shift + 1)
        count = 0
        for lb, time in zip(labels_i, times_i):
            
            if alignment:
                local_chunk = data_file.get_snippet(time - 2*template_shift, 2*N_t - 1, nodes=nodes)
            else:
                local_chunk = data_file.get_snippet(time - template_shift, N_t, nodes=nodes)
            
            if do_spatial_whitening:
                local_chunk = numpy.dot(local_chunk, spatial_whitening)
            if do_temporal_whitening:
                local_chunk = scipy.ndimage.filters.convolve1d(local_chunk, temporal_whitening, axis=0, mode='constant')
            
            local_chunk = numpy.take(local_chunk, neighs, axis=1)
            
            if alignment:
                idx = numpy.where(neighs == src)[0]
                ydata = numpy.arange(len(neighs))
                if len(ydata) == 1:
                    f = scipy.interpolate.UnivariateSpline(xdata, local_chunk, s=0)
                    rmin = (numpy.argmin(f(cdata)) - len(cdata) / 2.0) / 5.0
                    ddata = numpy.linspace(rmin - template_shift, rmin + template_shift, N_t)
                    local_chunk = f(ddata).astype(numpy.float32).reshape(N_t, 1)
                else:
                    f = scipy.interpolate.RectBivariateSpline(xdata, ydata, local_chunk, s=0, ky=min(len(ydata) - 1, 3))
                    rmin = (numpy.argmin(f(cdata, idx)[:, 0]) - len(cdata) / 2.0) / 5.0
                    ddata = numpy.linspace(rmin - template_shift, rmin + template_shift, N_t)
                    local_chunk = f(ddata, ydata).astype(numpy.float32)
            if not all_labels:
                if not mean_mode:
                    # #####
                    # print(stas.shape)
                    # print(count)
                    # #####
                    stas[count, :, :] = local_chunk.T
                    count += 1
                else:
                    stas += local_chunk.T
            else:
                lc = numpy.where(nb_triggers == lb)[0]
                stas[lc] += local_chunk.T
    
    sub_comm.Barrier()
    
    # # Let each node wrap the data structures to share.
    # if not all_labels and mean_mode:
    #     stas_shape = (nb_neighs, nb_ts)
    # else:
    #     stas_shape = (nb_triggers, nb_neighs, nb_ts)
    # stas = numpy.reshape(stas, stas_shape)
    
    sub_comm.Free()
    
    return stas

##### end working zone


def get_artefact(data_file, times_i, tau, nodes, normalize=True):
    

    artefact     = numpy.zeros((len(nodes), tau), dtype=numpy.float32)
    for time in times_i:
        artefact += data_file.get_snippet(time, tau, nodes).T

    if normalize:
        artefact /= len(times_i)

    return artefact



def load_chunk(data_file, idx, chunk_len, chunk_size=None, padding=(0, 0), nodes=None):
    
    return data_file.get_data(idx, chunk_len, chunk_size, padding, nodes)


def prepare_preview(data_file, preview_filename):
    
    data_file.prepare_preview(preview_filename)


def analyze_data(data_file, chunk_size=None):

    return data_file.analyze(chunk_size)


def get_nodes_and_edges(parameters, validating=False):
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
    probe  = read_probe(parameters)
    radius = parameters.getint('data', 'radius')

    if validating:
        radius_factor = parameters.getfloat('validating', 'radius_factor')
        radius = int(radius_factor * float(radius))

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

def get_averaged_n_edges(parameters):
    nodes, edges = get_nodes_and_edges(parameters)
    n = 0
    for key, value in edges.items():
        n += len(value)
    return n/float(len(edges.values()))


def load_data_memshared(params, comm, data, extension='', normalize=False, transpose=False, nb_cpu=1, nb_gpu=0, use_gpu=False):

    file_out        = params.get('data', 'file_out')
    file_out_suff   = params.get('data', 'file_out_suff')
    data_file_noext = params.get('data', 'data_file_noext')

    ## First we need to identify machines in the MPI ring
    from uuid import getnode as get_mac
    myip = numpy.int64(get_mac()) % 100000

    intsize   = MPI.INT.Get_size()
    floatsize = MPI.FLOAT.Get_size() 
    sub_comm  = comm.Split(myip, 0)
    
    if data == 'templates':
        N_e = params.getint('data', 'N_e')
        N_t = params.getint('data', 'N_t')
        if os.path.exists(file_out_suff + '.templates%s.hdf5' %extension):
            nb_data = 0
            nb_ptr  = 0
            nb_templates = h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r', libver='latest').get('norms').shape[0]

            if sub_comm.rank == 0:
                temp_x       = h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r', libver='latest').get('temp_x')[:]
                temp_y       = h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r', libver='latest').get('temp_y')[:]
                temp_data    = h5py.File(file_out_suff + '.templates%s.hdf5' %extension, 'r', libver='latest').get('temp_data')[:]    
                sparse_mat = scipy.sparse.csc_matrix((temp_data, (temp_x, temp_y)), shape=(N_e*N_t, nb_templates))
                if normalize:
                    norm_templates = load_data(params, 'norm-templates')
                    for idx in xrange(sparse_mat.shape[1]):
                        myslice = numpy.arange(sparse_mat.indptr[idx], sparse_mat.indptr[idx+1])
                        sparse_mat.data[myslice] /= norm_templates[idx]
                if transpose:
                    sparse_mat = sparse_mat.T

                nb_data = len(sparse_mat.data)
                nb_ptr  = len(sparse_mat.indptr)

            sub_comm.Barrier()   
            long_size  = numpy.int64(sub_comm.bcast(numpy.array([nb_data], dtype=numpy.int32), root=0)[0])
            short_size = numpy.int64(sub_comm.bcast(numpy.array([nb_ptr], dtype=numpy.int32), root=0)[0])

            if sub_comm.rank == 0:
                indptr_bytes  = short_size * intsize
                indices_bytes = long_size * intsize
                data_bytes    = long_size * floatsize
            else:
                indptr_bytes  = 0
                indices_bytes = 0
                data_bytes    = 0

            win_data    = MPI.Win.Allocate_shared(data_bytes, floatsize, comm=sub_comm)
            win_indices = MPI.Win.Allocate_shared(indices_bytes, intsize, comm=sub_comm)
            win_indptr  = MPI.Win.Allocate_shared(indptr_bytes, intsize, comm=sub_comm)

            buf_data, _    = win_data.Shared_query(0)
            buf_indices, _ = win_indices.Shared_query(0)
            buf_indptr, _  = win_indptr.Shared_query(0)
                
            buf_data    = numpy.array(buf_data, dtype='B', copy=False)
            buf_indices = numpy.array(buf_indices, dtype='B', copy=False)
            buf_indptr  = numpy.array(buf_indptr, dtype='B', copy=False)
                                
            data    = numpy.ndarray(buffer=buf_data, dtype=numpy.float32, shape=(long_size,))
            indices = numpy.ndarray(buffer=buf_indices, dtype=numpy.int32, shape=(long_size,))
            indptr  = numpy.ndarray(buffer=buf_indptr, dtype=numpy.int32, shape=(short_size,))

            sub_comm.Barrier()

            if sub_comm.rank == 0:
                data[:nb_data]    = sparse_mat.data
                indices[:nb_data] = sparse_mat.indices
                indptr[:nb_data]  = sparse_mat.indptr
                del sparse_mat

            sub_comm.Barrier()
            if not transpose:
                templates = scipy.sparse.csc_matrix((N_e*N_t, nb_templates), dtype=numpy.float32)
            else:
                templates = scipy.sparse.csr_matrix((nb_templates, N_e*N_t), dtype=numpy.float32)
            templates.data    = data
            templates.indices = indices
            templates.indptr  = indptr

            sub_comm.Free()

            return templates
        else:
            raise Exception('No templates found! Check suffix?')
    elif data == "overlaps":
        
        c_overlap  = get_overlaps(comm, params, nb_cpu=nb_cpu, nb_gpu=nb_gpu, use_gpu=use_gpu)
        over_shape = c_overlap.get('over_shape')[:]
        N_over     = numpy.int64(numpy.sqrt(over_shape[0]))
        S_over     = over_shape[1]
        c_overs    = {}
            
        if sub_comm.rank == 0:
            over_x     = c_overlap.get('over_x')[:]
            over_y     = c_overlap.get('over_y')[:]
            over_data  = c_overlap.get('over_data')[:]
            c_overlap.close()

            # To be faster, we rearrange the overlaps into a dictionnary
            overlaps  = scipy.sparse.csr_matrix((over_data, (over_x, over_y)), shape=(over_shape[0], over_shape[1]))
            del over_x, over_y, over_data

        sub_comm.Barrier()                
        
        nb_data = 0
        nb_ptr  = 0

        for i in xrange(N_over):
            
            if sub_comm.rank == 0:
                sparse_mat = overlaps[i*N_over:(i+1)*N_over]
                nb_data    = len(sparse_mat.data)
                nb_ptr     = len(sparse_mat.indptr)

            long_size  = numpy.int64(sub_comm.bcast(numpy.array([nb_data], dtype=numpy.int32), root=0)[0])
            short_size = numpy.int64(sub_comm.bcast(numpy.array([nb_ptr], dtype=numpy.int32), root=0)[0])

            if sub_comm.rank == 0:
                indptr_bytes  = short_size * intsize
                indices_bytes = long_size * intsize
                data_bytes    = long_size * floatsize
            else:
                indptr_bytes  = 0
                indices_bytes = 0
                data_bytes    = 0

            win_data    = MPI.Win.Allocate_shared(data_bytes, floatsize, comm=sub_comm)
            win_indices = MPI.Win.Allocate_shared(indices_bytes, intsize, comm=sub_comm)
            win_indptr  = MPI.Win.Allocate_shared(indptr_bytes, intsize, comm=sub_comm)

            buf_data, _    = win_data.Shared_query(0)
            buf_indices, _ = win_indices.Shared_query(0)
            buf_indptr, _  = win_indptr.Shared_query(0)
                
            buf_data    = numpy.array(buf_data, dtype='B', copy=False)
            buf_indices = numpy.array(buf_indices, dtype='B', copy=False)
            buf_indptr  = numpy.array(buf_indptr, dtype='B', copy=False)
                                
            data    = numpy.ndarray(buffer=buf_data, dtype=numpy.float32, shape=(long_size,))
            indices = numpy.ndarray(buffer=buf_indices, dtype=numpy.int32, shape=(long_size,))
            indptr  = numpy.ndarray(buffer=buf_indptr, dtype=numpy.int32, shape=(short_size,))

            sub_comm.Barrier()

            if sub_comm.rank == 0:
                data[:]    = sparse_mat.data
                indices[:] = sparse_mat.indices
                indptr[:]  = sparse_mat.indptr
                del sparse_mat

            c_overs[i]         = scipy.sparse.csr_matrix((N_over, over_shape[1]), dtype=numpy.float32)
            c_overs[i].data    = data
            c_overs[i].indices = indices
            c_overs[i].indptr  = indptr

            sub_comm.Barrier()
                    
        if sub_comm.rank == 0:
            del overlaps

        sub_comm.Free()

        return c_overs

    elif data == 'clusters-light':

        if os.path.exists(file_out_suff + '.clusters%s.hdf5' %extension):
            myfile = h5py.File(file_out_suff + '.clusters%s.hdf5' %extension, 'r', libver='latest')
            result = {}

            nb_data = 0

            for key in myfile.keys():
            
                if ('clusters_' in key) or (key == 'electrodes'):
                    if sub_comm.rank == 0:
                        locdata = myfile.get(key)[:]
                        nb_data = len(locdata)

                    data_size  = numpy.int64(sub_comm.bcast(numpy.array([nb_data], dtype=numpy.int32), root=0)[0])

                    if sub_comm.rank == 0:
                        if locdata.dtype == 'int32':
                            type_size = 0
                        elif locdata.dtype == 'float32':
                            type_size = 1
                        data_bytes = data_size * 4
                    else:
                        type_size  = 0
                        data_bytes = 0

                    type_size  = numpy.int64(sub_comm.bcast(numpy.array([type_size], dtype=numpy.int32), root=0)[0])

                    win_data    = MPI.Win.Allocate_shared(data_bytes, 4, comm=sub_comm)
                    buf_data, _ = win_data.Shared_query(0)
                        
                    buf_data    = numpy.array(buf_data, dtype='B', copy=False)

                    if type_size == 0:
                        data = numpy.ndarray(buffer=buf_data, dtype=numpy.int32, shape=(data_size,))
                    elif type_size == 1:
                        data = numpy.ndarray(buffer=buf_data, dtype=numpy.float32, shape=(data_size,))

                    if sub_comm.rank == 0:
                        data[:]    = locdata

                    sub_comm.Barrier()

                    result[str(key)] = data

            sub_comm.Free()

            myfile.close()
            return result

    

def load_data(params, data, extension=''):
    """
    Load data from a dataset.
    
    Parameters
    ----------
    data : {'thresholds', 'spatial_whitening', 'temporal_whitening', 'basis',
            'templates', 'norm-templates', 'spike-cluster', 'spikedetekt',
            'clusters', 'electrodes', 'results', 'overlaps', 'limits',
            'injected_spikes', 'triggers'}
    
    """

    file_out_suff   = params.get('data', 'file_out_suff')
    data_file_noext = params.get('data', 'data_file_noext')

    if data == 'thresholds':
        spike_thresh = params.getfloat('detection', 'spike_thresh')
        if os.path.exists(file_out_suff + '.basis.hdf5'):
            myfile     = h5py.File(file_out_suff + '.basis.hdf5', 'r', libver='latest')
            thresholds = myfile.get('thresholds')[:]
            myfile.close()
            return spike_thresh * thresholds 
    elif data == 'matched-thresholds':
        matched_thresh = params.getfloat('detection', 'matched_thresh')
        if os.path.exists(file_out_suff + '.basis.hdf5'):
            myfile     = h5py.File(file_out_suff + '.basis.hdf5', 'r', libver='latest')
            thresholds = myfile.get('matched_thresholds')[:]
            myfile.close()
            return matched_thresh * thresholds 
    elif data == 'matched-thresholds-pos':
        matched_thresh = params.getfloat('detection', 'matched_thresh')
        if os.path.exists(file_out_suff + '.basis.hdf5'):
            myfile     = h5py.File(file_out_suff + '.basis.hdf5', 'r', libver='latest')
            thresholds = myfile.get('matched_thresholds_pos')[:]
            myfile.close()
            return matched_thresh * thresholds 
    elif data == 'spatial_whitening':
        if os.path.exists(file_out_suff + '.basis.hdf5'):
            myfile  = h5py.File(file_out_suff + '.basis.hdf5', 'r', libver='latest')
            spatial = numpy.ascontiguousarray(myfile.get('spatial')[:])
            myfile.close()
            return spatial
        else:
            raise Exception('Whitening matrix has to be computed first!')
    elif data == 'temporal_whitening':
        if os.path.exists(file_out_suff + '.basis.hdf5'):
            myfile   = h5py.File(file_out_suff + '.basis.hdf5', 'r', libver='latest')
            temporal = myfile.get('temporal')[:]
            myfile.close() 
            return temporal
        else:
            raise Exception('Whitening matrix has to be computed first!')
    elif data == 'basis':
        myfile     = h5py.File(file_out_suff + '.basis.hdf5', 'r', libver='latest')
        basis_proj = numpy.ascontiguousarray(myfile.get('proj')[:])
        basis_rec  = numpy.ascontiguousarray(myfile.get('rec')[:])
        myfile.close()
        return basis_proj, basis_rec
    elif data == 'basis-pos':
        myfile     = h5py.File(file_out_suff + '.basis.hdf5', 'r', libver='latest')
        basis_proj = numpy.ascontiguousarray(myfile.get('proj_pos')[:])
        basis_rec  = numpy.ascontiguousarray(myfile.get('rec_pos')[:])
        myfile.close()
        return basis_proj, basis_rec
    elif data == 'waveform':
        myfile     = h5py.File(file_out_suff + '.basis.hdf5', 'r', libver='latest')
        waveforms  = myfile.get('waveform')[:]
        myfile.close()
        return waveforms
    elif data == 'waveforms':
        myfile     = h5py.File(file_out_suff + '.basis.hdf5', 'r', libver='latest')
        waveforms  = myfile.get('waveforms')[:]
        myfile.close()
        return waveforms
    elif data == 'waveform-pos':
        myfile     = h5py.File(file_out_suff + '.basis.hdf5', 'r', libver='latest')
        waveforms  = myfile.get('waveform_pos')[:]
        myfile.close()
        return waveforms
    elif data == 'waveforms-pos':
        myfile     = h5py.File(file_out_suff + '.basis.hdf5', 'r', libver='latest')
        waveforms  = myfile.get('waveforms_pos')[:]
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
            clusters   = data.get('clusters').ravel()
            N_clusters = len(numpy.unique(clusters))
            spiketimes = data.get('spikes').ravel()
            return clusters, spiketimes, N_clusters
        else:
            raise Exception('Need to provide a spike-cluster file!')
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
    elif data == 'clusters-light':
        if os.path.exists(file_out_suff + '.clusters%s.hdf5' %extension):
            myfile = h5py.File(file_out_suff + '.clusters%s.hdf5' %extension, 'r', libver='latest')
            result = {}
            for key in myfile.keys():
                if ('clusters_' in key) or (key == 'electrodes'):
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
    elif data == 'triggers':
        filename = file_out_suff + '.triggers%s.npy' %extension
        if os.path.exists(filename):
            triggers = numpy.load(filename)
            N_tr = triggers.shape[0]

            data_file = params.get('data', 'data_file')
            data_offset = params.getint('data', 'data_offset')
            data_dtype = params.get('data', 'data_dtype')
            chunk_size = params.getint('data', 'chunk_size')
            N_total = params.getint('data', 'N_total')
            N_t = params.getint('data', 'N_t')
            dtype_offset = params.getint('data', 'dtype_offset')
            
            datablock = numpy.memmap(data_file, offset=data_offset, dtype=data_dtype, mode='r')
            template_shift = numpy.int64((N_t - 1) / 2)

            spikes = numpy.zeros((N_t, N_total, N_tr))
            for (count, idx) in enumerate(triggers):
                chunk_len = numpy.int64(chunk_size) * N_total
                chunk_start = (idx - template_shift) * N_total
                chunk_end = (idx + template_shift + 1)  * N_total
                local_chunk = datablock[chunk_start:chunk_end]

                local_chunk = local_chunk.reshape(N_t, N_total)
                local_chunk = local_chunk.astype(numpy.float32)
                local_chunk -= dtype_offset

                spikes[:, :, count] = local_chunk
            return triggers, spikes
        else:
            raise Exception('No triggers found! Check suffix or check if file `%s` exists?' %filename)
    elif data == 'juxta-mad':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='latest')
            try:
                juxta_mad = beer_file.get('juxta_mad').value
            finally:
                beer_file.close()
            return juxta_mad
        else:
            raise Exception('No median absolute deviation found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data == 'juxta-triggers':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='latest')
            try:
                juxta_spike_times = beer_file.get('juxta_spiketimes/elec_0')[:]
            finally:
                beer_file.close()
            return juxta_spike_times
        else:
            raise Exception('No triggers found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data == 'juxta-values':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='latest')
            try:
                juxta_spike_values = beer_file.get('juxta_spike_values/elec_0')[:]
            finally:
                beer_file.close()
            return juxta_spike_values
        else:
            raise Exception('No values found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data == 'extra-mads':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='latest')
            try:
                extra_mads = beer_file.get('extra_mads')[:]
            finally:
                beer_file.close()
            return extra_mads
        else:
            raise Exception('No median absolute deviation found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data == 'extra-triggers':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='latest')
            N_e = params.getint('data', 'N_e')
            extra_spike_times = N_e * [None]
            try:
                for e in xrange(0, N_e):
                    key = "extra_spiketimes/elec_{}".format(e)
                    extra_spike_times[e] = beer_file.get(key)[:]
            finally:
                beer_file.close()
            return extra_spike_times
        else:
            raise Exception('No triggers found! Check if file `{}` exists?'.format(filename))
    elif data == 'extra-values':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='latest')
            N_e = params.getint('data', 'N_e')
            extra_spike_values = N_e * [None]
            try:
                for e in xrange(0, N_e):
                    key = "extra_spike_values/elec_{}".format(e)
                    extra_spike_values[e] = beer_file.get(key)[:]
            finally:
                beer_file.close()
            return extra_spike_values
        else:
            raise Exception('No values found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data == 'class-weights':
        filename = file_out_suff + '.beer.hdf5'
        if os.path.exists(filename):
            bfile = h5py.File(filename, 'r', libver='latest')
            class_weights = bfile.get('class-weights')[:]
            bfile.close()
            return class_weights
        else:
            raise Exception('No class weights found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data == 'confusion-matrices':
        filename = file_out_suff + '.beer.hdf5'
        if os.path.exists(filename):
            bfile = h5py.File(filename, 'r', libver='latest')
            confusion_matrices = bfile.get('confusion_matrices')[:]
            bfile.close()
            return confusion_matrices
        else:
            raise Exception('No confusion matrices found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data == 'proportion':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='latest')
            try:
                proportion = beer_file.get('proportion').value
            finally:
                beer_file.close()
            return proportion
        else:
            raise Exception('No proportion found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data == 'threshold-false-negatives':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='latest')
            try:
                threshold_false_negatives = beer_file.get('thresh_fn').value
            finally:
                beer_file.close()
            return threshold_false_negatives
        else:
            raise Exception('No threshold false negatives found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data in ['false-positive-rates', 'true-positive-rates',
                  'false-positive-error-rates', 'false-negative-error-rates']:
        # Retrieve saved data.
        confusion_matrices = load_data(params, 'confusion-matrices')
        threshold_false_negatives = load_data(params, 'threshold-false-negatives')
        # Correct counts of false negatives.
        for confusion_matrix in confusion_matrices:
            confusion_matrix[0, 1] += threshold_false_negatives
        # Compute the wanted statistics.
        if data == 'false-positive-rates':
            # Compute false positive rates (i.e. FP / (FP + TN)).
            results = [M[1, 0] / (M[1, 0] + M[1, 1]) for M in confusion_matrices]
            # Add false positive rate endpoints.
            results = [1.0] + results + [0.0]
        if data == 'true-positive-rates':
            # Compute true positive rates (i.e. TP / (TP + FN)).
            results = [M[0, 0] / (M[0, 0] + M[0, 1]) for M in confusion_matrices]
            # Add true positive rate endpoints.
            results = [1.0] + results + [0.0]
        if data == 'false-positive-error-rates':
            # Compute false positive error rates (i.e. FP / (TP + FP)).
            results = [M[1, 0] / (M[0, 0] + M[1, 0]) for M in confusion_matrices]
            # Add false positive error rate endpoints.
            results = [1.0] + results + [0.0]
        if data == 'false-negative-error-rates':
            # Compute false negative error rates (i.e. FN / (TP + FN)).
            results = [M[0, 1] / (M[0, 0] + M[0, 1]) for M in confusion_matrices]
            # Add false negative error rate endpoints.
            results = [0.0] + results + [1.0]
        results = numpy.array(results, dtype=numpy.float)
        return results
    elif data == 'sc-contingency-matrices':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='latest')
            try:
                sc_contingency_matrices = beer_file.get('sc_contingency_matrices')[:]
            finally:
                beer_file.close()
            return sc_contingency_matrices
        else:
            raise Exception('No contingency matrices found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data in ['sc-false-positive-error-rates', 'sc-false-negative-error-rates']:
        # Retrieve saved data.
        sc_contingency_matrices = load_data(params, 'sc-contingency-matrices')
        threshold_false_negatives = load_data(params, 'threshold-false-negatives')
        # Correct counts of false negatives.
        for sc_contingency_matrix in sc_contingency_matrices:
            sc_contingency_matrix[0, 1] += threshold_false_negatives
        # Compute the wanted statistics.
        if data == 'sc-false-positive-error-rates':
            # Compute false positive error rates.
            results = [float(M[1, 1]) / float(M[1, 0] + M[1, 1]) if 0 < M[1, 0] + M[1, 1] else 0.0
                       for M in sc_contingency_matrices]
        if data == 'sc-false-negative-error-rates':
            # Compute false negative error rates.
            results = [float(M[0, 1]) / float(M[0, 0] + M[0, 1]) if 0 < M[0, 0] + M[0, 1] else 0.0
                       for M in sc_contingency_matrices]
        results = numpy.array(results, dtype=numpy.float)
        return results
    elif data == 'sc-contingency-matrix':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='latest')
            try:
                sc_contingency_matrix = beer_file.get('sc_contingency_matrix')[:]
            finally:
                beer_file.close()
            return sc_contingency_matrix
        else:
            raise Exception('No contingency matrix found! Check suffix or check if file `{}` exists?'.format(filename))
    elif data in ['sc-best-false-positive-error-rate', 'sc-best-false-negative-error-rate']:
        sc_contingency_matrix = load_data(params, 'sc-contingency-matrix')
        threshold_false_negatives = load_data(params, 'threshold-false-negatives')
        # Correct count of false negatives.
        sc_contingency_matrix[0, 1] += threshold_false_negatives
        # Compute the wanted statistics.
        if data == 'sc-best-false-positive-error-rate':
            # Compute best false positive error rate.
            M = sc_contingency_matrix
            result = float(M[1, 1]) / float(M[1, 0] + M[1, 1]) if 0 < M[1, 0] + M[1, 1] else 0.0
        if data == 'sc-best-false-negative-error-rate':
            # Compute best false negative error rate.
            M = sc_contingency_matrix
            result = float(M[0, 1]) / float(M[0, 0] + M[0, 1]) if 0 < M[0, 0] + M[0, 1] else 0.0
        return result
    elif data == 'selection':
        filename = "{}.beer{}.hdf5".format(file_out_suff, extension)
        if os.path.exists(filename):
            beer_file = h5py.File(filename, 'r', libver='latest')
            try:
                selection = beer_file.get('selection')[:]
            finally:
                beer_file.close()
            return selection
        else:
            raise Exception('No selection found! Check suffix or check if file `{}` exists?'.format(filename))


def write_datasets(h5file, to_write, result, electrode=None):
    for key in to_write:
        if electrode is not None:
            mykey = key + str(electrode)
        else:
            mykey = key
        h5file.create_dataset(mykey, shape=result[mykey].shape, dtype=result[mykey].dtype, chunks=True)
        h5file.get(mykey)[:] = result[mykey]

def collect_data(nb_threads, data_file, erase=False, with_real_amps=False, with_voltages=False, benchmark=False):

    # Retrieve the key parameters.
    params         = data_file.params
    file_out_suff  = params.get('data', 'file_out_suff')
    N_e            = params.getint('data', 'N_e')
    N_t            = params.getint('data', 'N_t')
    max_chunk      = params.getfloat('fitting', 'max_chunk')
    chunks         = params.getfloat('fitting', 'chunk')
    data_length    = data_stats(data_file, show=False)
    duration       = int(min(chunks*max_chunk, data_length))
    templates      = load_data(params, 'norm-templates')
    sampling_rate  = params.getint('data', 'sampling_rate')
    refractory     = numpy.int64(params.getfloat('fitting', 'refractory')*sampling_rate*1e-3)
    N_tm           = len(templates)

    print_and_log(["Gathering data from %d nodes..." %nb_threads], 'default', params)

    # Initialize data collection.
    result = {'spiketimes' : {}, 'amplitudes' : {}, 'info' : {'duration' : numpy.array([duration], dtype=numpy.uint64)}}
    if with_real_amps:
        result['real_amps'] = {}
    if with_voltages:
        result['voltages'] = {}    
    for i in xrange(N_tm//2):
        result['spiketimes']['temp_' + str(i)]  = numpy.empty(shape=0)
        result['amplitudes']['temp_' + str(i)]  = numpy.empty(shape=(0, 2))
        if with_real_amps:
            result['real_amps']['temp_' + str(i)] = numpy.empty(shape=0)
        if with_voltages:
            result['voltages']['temp_' + str(i)] = numpy.empty(shape=0)

    pbar = get_progressbar(size=nb_threads)

    # For each thread/process collect data.
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
            spiketimes = numpy.fromfile(spiketimes_file, dtype=numpy.uint32)
            templates  = numpy.fromfile(templates_file, dtype=numpy.int32)
            N          = len(amplitudes)
            amplitudes = amplitudes.reshape(N//2, 2)
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

    # TODO: find a programmer comment.
    for key in result['spiketimes']:
        result['spiketimes'][key] = numpy.array(result['spiketimes'][key], dtype=numpy.uint32)
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

    keys = ['spiketimes', 'amplitudes', 'info']
    if with_real_amps:
        keys += ['real_amps']
    if with_voltages:
        keys += ['voltages']

    # Save results into `<dataset>/<dataset>.result.hdf5`.
    mydata = h5py.File(file_out_suff + '.result.hdf5', 'w', libver='latest')
    for key in keys:
        mydata.create_group(key)
        for temp in result[key].keys():
            tmp_path = '%s/%s' %(key, temp)
            mydata.create_dataset(tmp_path, data=result[key][temp])
    mydata.close()        

    # Count and print the number of spikes.
    count = 0
    for item in result['spiketimes'].keys():
        count += len(result['spiketimes'][item])

    if benchmark:
        to_print = "injected"
    else:
        to_print = "fitted"

    print_and_log(["Number of spikes %s : %d" %(to_print, count)], 'info', params)

    # TODO: find a programmer comment
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

def get_overlaps(comm, params, extension='', erase=False, normalize=True, maxoverlap=True, verbose=True, half=False, use_gpu=False, nb_cpu=1, nb_gpu=0):

    import h5py
    parallel_hdf5  = h5py.get_config().mpi

    try:
        SHARED_MEMORY = True
        MPI.Win.Allocate_shared(1, 1, MPI.INFO_NULL, MPI.COMM_SELF).Free()
    except NotImplementedError:
        SHARED_MEMORY = False

    file_out_suff  = params.get('data', 'file_out_suff')   
    tmp_path       = os.path.join(os.path.abspath(params.get('data', 'data_file_noext')), 'tmp')
    filename       = file_out_suff + '.overlap%s.hdf5' %extension

    if os.path.exists(filename) and not erase:
        return h5py.File(filename, 'r')
    else:
        if os.path.exists(filename) and erase and (comm.rank == 0):
            os.remove(filename)

    if maxoverlap:
        if SHARED_MEMORY:
            templates  = load_data_memshared(params, comm, 'templates', extension=extension, normalize=normalize)
        else:
            templates  = load_data(params, 'templates', extension=extension)
    else:
        if SHARED_MEMORY:
            templates  = load_data_memshared(params, comm, 'templates', normalize=normalize)
        else:
            templates  = load_data(params, 'templates')

    if extension == '-merged':
        best_elec  = load_data(params, 'electrodes', extension)
    else:
        best_elec  = load_data(params, 'electrodes')
    N_total        = params.getint('data', 'N_total')
    nodes, edges   = get_nodes_and_edges(params)
    N_e            = params.getint('data', 'N_e')
    N_t            = params.getint('data', 'N_t')
    x,        N_tm = templates.shape

    if not SHARED_MEMORY and normalize:
        norm_templates = load_data(params, 'norm-templates')[:N_tm]
        for idx in xrange(N_tm):
            myslice = numpy.arange(templates.indptr[idx], templates.indptr[idx+1])
            templates.data[myslice] /= norm_templates[idx]

    if half:
        N_tm //= 2
    
    comm.Barrier()
    inv_nodes        = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)

    cuda_string = 'using %d CPU...' %comm.size
    
    if use_gpu:
        import cudamat as cmt
        if parallel_hdf5:
            if nb_gpu > nb_cpu:
                gpu_id = int(comm.rank//nb_cpu)
            else:
                gpu_id = 0
        else:
            gpu_id = 0
        cmt.cuda_set_device(gpu_id)
        cmt.init()
        cmt.cuda_sync_threads()

    if use_gpu:
        cuda_string = 'using %d GPU...' %comm.size    

    all_delays      = numpy.arange(1, N_t+1)

    local_templates = numpy.zeros(0, dtype=numpy.int32)
    for ielec in range(comm.rank, N_e, comm.size):
        local_templates = numpy.concatenate((local_templates, numpy.where(best_elec == ielec)[0]))

    if half:
        nb_total     = len(local_templates)
        upper_bounds = N_tm
    else:
        nb_total     = 2*len(local_templates)
        upper_bounds = N_tm//2

    if comm.rank == 0:
        if verbose:
            print_and_log(["Pre-computing the overlaps of templates %s" %cuda_string], 'default', params)
        N_0  = len(range(comm.rank, N_e, comm.size))
        pbar = get_progressbar(size=N_0)

    over_x    = numpy.zeros(0, dtype=numpy.int32)
    over_y    = numpy.zeros(0, dtype=numpy.int32)
    over_data = numpy.zeros(0, dtype=numpy.float32)
    rows      = numpy.arange(N_e*N_t)
                
    for count, ielec in enumerate(range(comm.rank, N_e, comm.size)):
        
        local_idx = numpy.where(best_elec == ielec)[0]
        len_local = len(local_idx)

        if not half:
            local_idx = numpy.concatenate((local_idx, local_idx + upper_bounds))

        if len_local > 0:

            to_consider   = numpy.arange(upper_bounds)
            if not half:
                to_consider = numpy.concatenate((to_consider, to_consider + upper_bounds))
            
            loc_templates  = templates[:, local_idx].tocsr()
            loc_templates2 = templates[:, to_consider].tocsr()
            
            for idelay in all_delays:

                srows = numpy.where(rows % N_t < idelay)[0]
                tmp_1 = loc_templates[srows]

                srows = numpy.where(rows % N_t >= (N_t - idelay))[0]
                tmp_2 = loc_templates2[srows]
                
                if use_gpu:
                    tmp_1 = cmt.SparseCUDAMatrix(tmp_1.T.tocsr())
                    tmp_2 = cmt.CUDAMatrix(tmp_2.toarray(), copy_on_host=False)
                    data  = cmt.sparse_dot(tmp_1, tmp_2).asarray()
                else:
                    data  = tmp_1.T.dot(tmp_2)
                    data  = data.toarray()

                dx, dy     = data.nonzero()
                ddx        = numpy.take(local_idx, dx).astype(numpy.int32)
                ddy        = numpy.take(to_consider, dy).astype(numpy.int32)
                data       = data.ravel()
                dd         = data.nonzero()[0].astype(numpy.int32)
                over_x     = numpy.concatenate((over_x, ddx*N_tm + ddy))
                over_y     = numpy.concatenate((over_y, (idelay-1)*numpy.ones(len(dx), dtype=numpy.int32)))
                over_data  = numpy.concatenate((over_data, numpy.take(data, dd)))
                if idelay < N_t:
                    over_x     = numpy.concatenate((over_x, ddy*N_tm + ddx))
                    over_y     = numpy.concatenate((over_y, (2*N_t-idelay-1)*numpy.ones(len(dx), dtype=numpy.int32)))
                    over_data  = numpy.concatenate((over_data, numpy.take(data, dd)))

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
            if 'maxlag' in myfile2.keys():
                maxlag = myfile2.get('maxlag')
            else:
                maxlag = myfile2.create_dataset('maxlag', shape=(N_tm, N_tm), dtype=numpy.int32)

            for i in xrange(N_tm-1):
                data                = overlap[i*N_tm+i+1:(i+1)*N_tm].toarray()
                maxlag[i, i+1:]     = N_t - numpy.argmax(data, 1)
                maxlag[i+1:, i]     = maxlag[i, i+1:]
                maxoverlap[i, i+1:] = numpy.max(data, 1)
                maxoverlap[i+1:, i] = maxoverlap[i, i+1:]
            myfile.close()  
            myfile2.close()

    comm.Barrier()
    return h5py.File(filename, 'r')
