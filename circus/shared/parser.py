"""
parser.py

author: Pierre Yger
mail: pierre.yger <at> inserm.fr

Contains the class to read *param files
"""
import ConfigParser as configparser

from circus.shared.messages import print_and_log
from circus.shared.probes import read_probe, parse_dead_channels
from circus.shared.mpi import comm, check_if_cluster, check_valid_path
from circus.files import __supported_data_files__

import os
import sys
import copy
import numpy
import logging


logger = logging.getLogger(__name__)


class CircusParser(object):
    """
    Circus class to read *param files.

    Attributes
    ----------
    do_folders : bool
        if a folder must be created upon reading

    file_format : str
        file format of the recording (see with spyking-circus help -i)

    file_name : str
        path of recording file (e.g., '/analysis/continuous.dat')

    file_params : str
        path of the parameters file (e.g., '/analysis/continuous.params')
 
    logfile : str
        path of the spyking-circus log file created after execution

    nb_channels : int 
        number of recording channels described in the probe file.

    parser : configparser

    probe : dict
        the content of the probe file.
    
    Methods
    -------

    get(section, data)
        gets the data variable in the section of the params file.

    getboolean(section, data)
        gets the data variable in the section of the params file as bool.

    getfloat(section, data)
        gets the data variable in the section of the params file as float.

    getint(section, data)
        gets the data variable in the section of the params file as integer.

    set(section, data)
        sets a data variable in the section of the params file.

    get_data_file(is_empty=False, params=None, source=False, has_been_created=True)
        creates a dictionary form the datafile section in the param file.

    write(section, flag, value, preview_)
        writes the value in the variable data of a section of a param file.
    
    
    """

    __all_sections__ = ['data', 'whitening', 'extracting', 'clustering',
                       'fitting', 'filtering', 'merging', 'noedits', 'triggers',
                       'detection', 'validating', 'converting']

    __default_values__ = [['fitting', 'amp_auto', 'bool', 'True'],
                          ['fitting', 'refractory', 'float', '0.5'],
                          ['fitting', 'collect_all', 'bool', 'False'],
                          ['fitting', 'gpu_only', 'bool', 'False'],
                          ['fitting', 'ratio_thresh', 'float', '0.9'],
                          ['fitting', 'two_components', 'bool', 'True'],
                          ['fitting', 'auto_nb_chances', 'bool', 'True'],
                          ['data', 'global_tmp', 'bool', 'True'],
                          ['data', 'chunk_size', 'int', '30'],
                          ['data', 'stream_mode', 'string', 'None'],
                          ['data', 'overwrite', 'bool', 'True'],
                          ['data', 'parallel_hdf5', 'bool', 'True'],
                          ['data', 'output_dir', 'string', ''],
                          ['data', 'hdf5_compress', 'bool', 'True'],
                          ['data', 'blosc_compress', 'bool', 'False'],
                          ['data', 'is_cluster', 'bool', 'False'],
                          ['data', 'shared_memory', 'bool', 'True'],
                          ['data', 'status_bars', 'bool', 'True'],
                          ['detection', 'alignment', 'bool', 'True'],
                          ['detection', 'hanning', 'bool', 'True'],
                          ['detection', 'oversampling_factor', 'int', '10'],
                          ['detection', 'matched-filter', 'bool', 'False'],
                          ['detection', 'matched_thresh', 'float', '5'],
                          ['detection', 'peaks', 'string', 'negative'],
                          ['detection', 'spike_thresh', 'float', '6'],
                          ['detection', 'N_t', 'string', '3'],
                          ['detection', 'isolation', 'bool', 'True'],
                          ['detection', 'dead_channels', 'string', ''],
                          ['detection', 'spike_width', 'float', '0'],
                          ['triggers', 'clean_artefact', 'bool', 'False'],
                          ['triggers', 'make_plots', 'string', ''],
                          ['triggers', 'trig_file', 'string', ''],
                          ['triggers', 'trig_windows', 'string', ''],
                          ['triggers', 'trig_unit', 'string', 'ms'],
                          ['triggers', 'dead_unit', 'string', 'ms'],
                          ['triggers', 'dead_file', 'string', ''],
                          ['triggers', 'ignore_times', 'bool', 'False'],
                          ['whitening', 'chunk_size', 'int', '30'],
                          ['whitening', 'fudge', 'float', '1e-15'],
                          ['whitening', 'safety_space', 'bool', 'True'],
                          ['whitening', 'temporal', 'bool', 'False'],
                          ['whitening', 'ignore_spikes', 'bool', 'False'],
                          ['filtering', 'remove_median', 'bool', 'False'],
                          ['filtering', 'common_ground', 'string', ''],
                          ['clustering', 'nb_repeats', 'int', '3'],
                          ['clustering', 'make_plots', 'string', ''],
                          ['clustering', 'debug_plots', 'string', ''],
                          ['clustering', 'test_clusters', 'bool', 'False'],
                          ['clustering', 'smart_search', 'bool', 'True'],
                          ['clustering', 'safety_space', 'bool', 'True'],
                          ['clustering', 'compress', 'bool', 'True'],
                          ['clustering', 'noise_thr', 'float', '0.5'],
                          ['clustering', 'cc_merge', 'float', '0.95'],
                          ['clustering', 'n_abs_min', 'int', '20'],
                          ['clustering', 'sensitivity', 'float', '3'],
                          ['clustering', 'extraction', 'string', 'median-raw'],
                          ['clustering', 'merging_method', 'string', 'distance'],
                          ['clustering', 'merging_param', 'string', 'default'],
                          ['clustering', 'remove_mixture', 'bool', 'True'],
                          ['clustering', 'dispersion', 'string', '(5, 5)'],
                          ['clustering', 'two_components', 'bool', 'True'],
                          ['clustering', 'templates_normalization', 'bool', 'True'],
                          ['clustering', 'halo_rejection', 'float', 'inf'],
                          ['clustering', 'adapted_cc', 'bool', 'False'],
                          ['clustering', 'adapted_thr', 'int', '100'],
                          ['clustering', 'ignored_mixtures', 'float', '20'],
                          ['extracting', 'cc_merge', 'float', '0.95'],
                          ['merging', 'erase_all', 'bool', 'True'],
                          ['merging', 'cc_overlap', 'float', '0.75'],
                          ['merging', 'cc_bin', 'float', '2'],
                          ['merging', 'correct_lag', 'bool', 'False'],
                          ['merging', 'auto_mode', 'float', '0'],
                          ['merging', 'default_lag', 'float', '5'],
                          ['merging', 'remove_noise', 'bool', 'False'],
                          ['merging', 'noise_limit', 'float', '0.75'],
                          ['merging', 'sparsity_limit', 'float', '0.75'],
                          ['merging', 'merge_drifts', 'bool', 'True'],
                          ['merging', 'drift_limit', 'float', '1'],
                          ['merging', 'time_rpv', 'float', '5'],
                          ['merging', 'rpv_threshold', 'float', '0.02'],
                          ['merging', 'min_spikes', 'int', '100'],
                          ['converting', 'export_pcs', 'string', 'prompt'],
                          ['converting', 'erase_all', 'bool', 'True'],
                          ['converting', 'export_all', 'bool', 'False'],
                          ['converting', 'sparse_export', 'bool', 'False'],
                          ['converting', 'prelabelling', 'bool', 'False'],
                          ['converting', 'rpv_threshold', 'float', '0.05'],
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
                          ['validating', 'matching_jitter', 'float', '2.0'],
                          ['validating', 'filter', 'bool', 'True'],
                          ['validating', 'juxta_spikes', 'string', ''],
                          ['validating', 'greedy_mode', 'bool', 'True'],
                          ['validating', 'extension', 'string', ''],
                          ['noedits', 'filter_done', 'sting', 'False'],
                          ['noedits', 'median_done', 'string', 'False'],
                          ['noedits', 'ground_done', 'string', 'False'],
                          ['noedits', 'artefacts_done', 'string', 'False']]

    __extra_values__ = [['fitting', 'nb_chances', 'float', '3'],
                        ['fitting', 'max_chunk', 'float', 'inf'],
                        ['fitting', 'chunk_size', 'int', '1'],
                        ['fitting', 'debug', 'bool', 'False'],
                        ['fitting', 'max_nb_chances', 'int', '10'],
                        ['fitting', 'percent_nb_chances', 'float', '99'],
                        ['fitting', 'min_second_component', 'float', '0.1'],
                        ['filtering', 'butter_order', 'int', '3'],
                        ['clustering', 'm_ratio', 'float', '0.01'],
                        ['clustering', 'debug', 'bool', 'False'],
                        ['clustering', 'sub_dim', 'int', '10'],
                        ['clustering', 'decimation', 'bool', 'True'],
                        ['clustering', 'sparsify', 'float', '0.25'],
                        ['clustering', 'nb_ss_bins', 'string', 'auto'],
                        ['clustering', 'nb_ss_rand', 'int', '10000'],
                        ['clustering', 'nb_snippets', 'int', '50'],
                        ['clustering', 'fine_amplitude', 'bool', 'True'],
                        ['detection', 'jitter_range', 'float', '0.2'],
                        ['detection', 'smoothing_factor', 'float', '1.48'],
                        ['detection', 'rejection_threshold', 'float', '1'],
                        ['data', 'memory_usage', 'float', '0.1'],
                        ['clustering', 'safety_time', 'string', 'auto'],
                        ['clustering', 'savgol', 'bool', 'True'],
                        ['clustering', 'savgol_time', 'float', '0.2'],
                        ['detection', 'noise_time', 'float', '0.1'],
                        ['whitening', 'safety_time', 'string', 'auto'],
                        ['extracting', 'safety_time', 'string', 'auto']]

    def __init__(self, file_name, create_folders=True, **kwargs):
        """
        Parameters:
        ----------
        file_name : string
            a path containing the *params file.

        create_folders : bool
            if a folder will be created. If true, output_dir in the data section
        of the param file will be created.

        
        Returns:
        --------
        a CircusParser object. 

        """ 
        self.file_name = os.path.abspath(file_name)
        f_next, extension = os.path.splitext(self.file_name)
        file_path = os.path.dirname(self.file_name)
        self.file_params = f_next + '.params'
        self.do_folders = create_folders
        self.parser = configparser.ConfigParser()

        valid_path = check_valid_path(self.file_params)
        if not valid_path:
            print_and_log(["Not all nodes can read/write the data file. Check path?"], 'error', logger)
            sys.exit(0)

        # # First, we remove all tabulations from the parameter file, in order to secure the parser.
        if comm.rank == 0:
            myfile = open(self.file_params, 'r')
            lines = myfile.readlines()
            myfile.close()
            myfile = open(self.file_params, 'w')
            for l in lines:
                myfile.write(l.replace('\t', ''))
            myfile.close()

        comm.Barrier()

        self._N_t = None

        if not os.path.exists(self.file_params):
            if comm.rank == 0:
                print_and_log(["%s does not exist" % self.file_params], 'error', logger)
            sys.exit(0)

        if comm.rank == 0:
            print_and_log(['Creating a Circus Parser for datafile %s' % self.file_name], 'debug', logger)
        self.parser.read(self.file_params)

        for section in self.__all_sections__:
            if self.parser.has_section(section):
                for (key, value) in self.parser.items(section):
                    self.parser.set(section, key, value.split('#')[0].rstrip())
            else:
                self.parser.add_section(section)

        for item in self.__default_values__ + self.__extra_values__:
            section, name, val_type, value = item
            try:
                if val_type is 'bool':
                    self.parser.getboolean(section, name)
                elif val_type is 'int':
                    self.parser.getint(section, name)
                elif val_type is 'float':
                    self.parser.getfloat(section, name)
                elif val_type is 'string':
                    self.parser.get(section, name)
            except Exception:
                self.parser.set(section, name, value)

        for key, value in kwargs.items():
            for section in self.__all_sections__:
                if self.parser._sections[section].has_key(key):
                    self.parser._sections[section][key] = value

        if self.do_folders and self.parser.get('data', 'output_dir') == '':
            try:
                os.makedirs(f_next)
            except Exception:
                pass

        self.parser.set('data', 'data_file', self.file_name)

        if self.parser.get('data', 'output_dir') != '':
            path = os.path.abspath(os.path.expanduser(self.parser.get('data', 'output_dir')))
            self.parser.set('data', 'output_dir', path)
            file_out = os.path.join(path, os.path.basename(f_next))
            if not os.path.exists(file_out) and self.do_folders:
                os.makedirs(file_out)
            self.logfile = file_out + '.log'
        else:
            file_out = os.path.join(f_next, os.path.basename(f_next))
            self.logfile = f_next + '.log'

        self.parser.set('data', 'data_file_no_overwrite', file_out + '_all_sc.dat')
        self.parser.set('data', 'file_out', file_out)  # Output file without suffix
        self.parser.set('data', 'file_out_suff', file_out + self.parser.get('data', 'suffix'))  # Output file with suffix
        self.parser.set('data', 'data_file_noext', f_next)  # Data file (assuming .filtered at the end)

        # read .prb file
        try:
            self.parser.get('detection', 'radius')
        except Exception:  # radius == auto by default
            self.parser.set('detection', 'radius', 'auto')

        try:
            self.parser.getint('detection', 'radius')
        except Exception:  # when radius = auto in params file
            self.probe = read_probe(self.parser, radius_in_probe=True)
            self.parser.set('detection', 'radius', str(int(self.probe['radius'])))
        else:
            self.probe = read_probe(self.parser, radius_in_probe=False)

        dead_channels = self.parser.get('detection', 'dead_channels')
        if dead_channels != '':
            dead_channels = parse_dead_channels(dead_channels)
            if comm.rank == 0:
                print_and_log(["Removing dead channels %s" % str(dead_channels)], 'debug', logger)
            for key in dead_channels.keys():
                if key in self.probe["channel_groups"].keys():
                    for channel in dead_channels[key]:
                        n_before = len(self.probe["channel_groups"][key]['channels'])
                        self.probe["channel_groups"][key]['channels'] = list(set(self.probe["channel_groups"][key]['channels']).difference(dead_channels[key]))
                        n_after = len(self.probe["channel_groups"][key]['channels'])
                else:
                    if comm.rank == 0:
                        print_and_log(["Probe has no group named %s for dead channels" % key], 'debug', logger)

        N_e = 0
        for key in self.probe['channel_groups'].keys():
            N_e += len(self.probe['channel_groups'][key]['channels'])

        self.set('data', 'N_e', str(N_e))
        self.set('data', 'N_total', str(self.probe['total_nb_channels']))
        self.set('data', 'nb_channels', str(self.probe['total_nb_channels']))
        self.nb_channels = self.probe['total_nb_channels']

        if N_e > self.nb_channels:
            if comm.rank == 0:
                print_and_log(['The number of analyzed channels is higher than the number of recorded channels'], 'error', logger)
            sys.exit(0)

        if N_e == 1 and self.parser.getboolean('filtering', 'remove_median'):
            if comm.rank == 0:
                print_and_log(["With 1 channel, remove_median in [filtering] is not possible"], 'error', logger)
            sys.exit(0)

        to_write = ["You must specify explicitly the file format in the config file",
                    "Please have a look to the documentation and add a file_format",
                    "parameter in the [data] section. Valid files formats can be:", '']
        try:
            self.file_format = self.parser.get('data', 'file_format')
        except Exception:
            if comm.rank == 0:
                for f in __supported_data_files__.keys():
                    to_write += ['-- %s -- %s' % (f, __supported_data_files__[f].extension)]

                to_write += [
                    "",
                    "To get more info on a given file format, see",
                    ">> spyking-circus file_format -i",
                ]

                print_and_log(to_write, 'error', logger)
            sys.exit(0)

        test = self.file_format.lower() in __supported_data_files__.keys()
        if not test:
            if comm.rank == 0:
                for f in __supported_data_files__.keys():
                    to_write += ['-- %s -- %s' % (f, __supported_data_files__[f].extension)]

                to_write += [
                    "",
                    "To get more info on a given file format, see",
                    ">> spyking-circus file_format -i",
                ]

                print_and_log(to_write, 'error', logger)
            sys.exit(0)

        try:
            self.parser.get('detection', 'radius')
        except Exception:
            self.parser.set('detection', 'radius', 'auto')
        try:
            self.parser.getint('detection', 'radius')
        except Exception:
            self.parser.set('detection', 'radius', str(int(self.probe['radius'])))

        if self.parser.getboolean('triggers', 'clean_artefact'):
            if (self.parser.get('triggers', 'trig_file') == '') or (self.parser.get('triggers', 'trig_windows') == ''):
                if comm.rank == 0:
                    print_and_log(["trig_file and trig_windows must be specified in [triggers]"], 'error', logger)
                sys.exit(0)

        units = ['ms', 'timestep']
        test = self.parser.get('triggers', 'trig_unit').lower() in units
        if not test:
            if comm.rank == 0:
                print_and_log(["trig_unit in [triggers] should be in %s" % str(units)], 'error', logger)
            sys.exit(0)
        else:
            self.parser.set('triggers', 'trig_in_ms', str(self.parser.get('triggers', 'trig_unit').lower() == 'ms'))

        if self.parser.getboolean('triggers', 'clean_artefact'):
            for key in ['trig_file', 'trig_windows']:
                myfile = os.path.abspath(os.path.expanduser(self.parser.get('triggers', key)))
                if not os.path.exists(myfile):
                    if comm.rank == 0:
                        print_and_log(["File %s can not be found" % str(myfile)], 'error', logger)
                    sys.exit(0)
                self.parser.set('triggers', key, myfile)

        units = ['ms', 'timestep']
        test = self.parser.get('triggers', 'dead_unit').lower() in units
        if not test:
            if comm.rank == 0:
                print_and_log(["dead_unit in [triggers] should be in %s" % str(units)], 'error', logger)
            sys.exit(0)
        else:
            self.parser.set('triggers', 'dead_in_ms', str(self.parser.get('triggers', 'dead_unit').lower() == 'ms'))

        if self.parser.getboolean('triggers', 'ignore_times'):
            myfile = os.path.abspath(os.path.expanduser(self.parser.get('triggers', 'dead_file')))
            if not os.path.exists(myfile):
                if comm.rank == 0:
                    print_and_log(["File %s can not be found" % str(myfile)], 'error', logger)
                sys.exit(0)
            self.parser.set('triggers', 'dead_file', myfile)

        test = (self.parser.get('clustering', 'extraction').lower() in ['median-raw', 'mean-raw'])
        if not test:
            if comm.rank == 0:
                print_and_log(["Only 4 extraction modes in [clustering]: median-raw, mean-raw!"], 'error', logger)
            sys.exit(0)

        test = (self.parser.get('detection', 'peaks').lower() in ['negative', 'positive', 'both'])
        if not test:
            if comm.rank == 0:
                print_and_log(["Only 3 detection modes for peaks in [detection]: negative, positive, both"], 'error', logger)
            sys.exit(0)

        common_ground = self.parser.get('filtering', 'common_ground')
        if common_ground != '':
            try:
                self.parser.set('filtering', 'common_ground', str(int(common_ground)))
            except Exception:
                self.parser.set('filtering', 'common_ground', '-1')
        else:
            self.parser.set('filtering', 'common_ground', '-1')

        common_ground = self.parser.getint('filtering', 'common_ground')

        all_electrodes = []
        for key in self.probe['channel_groups'].keys():
            all_electrodes += self.probe['channel_groups'][key]['channels']

        test = (common_ground == -1) or common_ground in all_electrodes
        if not test:
            if comm.rank == 0:
                print_and_log(["Common ground in filtering section should be a valid electrode"], 'error', logger)
            sys.exit(0)

        is_cluster = check_if_cluster()

        self.parser.set('data', 'is_cluster', str(is_cluster))

        if is_cluster:
            print_and_log(["Cluster detected, so using local /tmp folders and blosc compression"], 'debug', logger)
            self.parser.set('data', 'global_tmp', 'False')
            self.parser.set('data', 'blosc_compress', 'True')
        else:
            print_and_log(["Cluster not detected, so using global /tmp folder"], 'debug', logger)

        for section in ['whitening', 'clustering']:
            test = (self.parser.getfloat(section, 'nb_elts') > 0) and (self.parser.getfloat(section, 'nb_elts') <= 1)
            if not test:
                if comm.rank == 0:
                    print_and_log(["nb_elts in [%s] should be in [0,1]" %section], 'error', logger)
                sys.exit(0)

        test = (self.parser.getfloat('validating', 'test_size') > 0) and (self.parser.getfloat('validating', 'test_size') < 1)
        if not test:
            if comm.rank == 0:
                print_and_log(["test_size in [validating] should be in ]0,1["], 'error', logger)
            sys.exit(0)

        test = (self.parser.getfloat('fitting', 'ratio_thresh') > 0) and (self.parser.getfloat('fitting', 'ratio_thresh') <= 1)
        if not test:
            if comm.rank == 0:
                print_and_log(["ratio_thresh in [fitting] should be in ]0,1]"], 'error', logger)
            sys.exit(0)

        test = (self.parser.getfloat('clustering', 'cc_merge') >= 0) and (self.parser.getfloat('clustering', 'cc_merge') <= 1)
        if not test:
            if comm.rank == 0:
                print_and_log(["cc_merge in [validating] should be in [0,1]"], 'error', logger)
            sys.exit(0)

        # test = (self.parser.getfloat('clustering', 'ignored_mixtures') >= 0) and (self.parser.getfloat('clustering', 'ignored_mixtures') <= 1)
        # if not test:
        #     if comm.rank == 0:
        #         print_and_log(["ignored_mixtures in [validating] should be in [0,1]"], 'error', logger)
        #     sys.exit(0)

        test = (self.parser.getfloat('data', 'memory_usage') > 0) and (self.parser.getfloat('data', 'memory_usage') <= 1)
        if not test:
            if comm.rank == 0:
                print_and_log(["memory_usage in [data] should be in ]0,1]"], 'error', logger)
            sys.exit(0)

        test = (self.parser.getfloat('merging', 'auto_mode') >= 0) and (self.parser.getfloat('merging', 'auto_mode') <= 1)
        if not test:
            if comm.rank == 0:
                print_and_log(["auto_mode in [merging] should be in [0, 1]"], 'error', logger)
            sys.exit(0)

        test = (self.parser.getfloat('merging', 'noise_limit') >= 0)
        if not test:
            if comm.rank == 0:
                print_and_log(["noise_limit in [merging] should be > 0"], 'error', logger)
            sys.exit(0)

        test = (self.parser.getfloat('merging', 'sparsity_limit') <= 1)
        if not test:
            if comm.rank == 0:
                print_and_log(["sparsity_limit in [merging] should be < 1"], 'error', logger)
            sys.exit(0)

        test = (self.parser.getint('detection', 'oversampling_factor') >= 0)
        if not test:
            if comm.rank == 0:
                print_and_log(["oversampling_factor in [detection] should be positive["], 'error', logger)
            sys.exit(0)

        test = (self.parser.getfloat('detection', 'smoothing_factor') >= 0)
        if not test:
            if comm.rank == 0:
                print_and_log(["smoothing_factor in [detection] should be positive["], 'error', logger)
            sys.exit(0)

        test = (not self.parser.getboolean('data', 'overwrite') and not self.parser.getboolean('filtering', 'filter'))
        if test:
            if comm.rank == 0:
                print_and_log(["If no filtering, then overwrite should be True"], 'error', logger)
            sys.exit(0)

        fileformats = ['png', 'pdf', 'eps', 'jpg', '', 'None']
        for section in ['clustering', 'validating', 'triggers']:
            test = self.parser.get('clustering', 'make_plots').lower() in fileformats
            if not test:
                if comm.rank == 0:
                    print_and_log(["make_plots in [%s] should be in %s" % (section, str(fileformats))], 'error', logger)
                sys.exit(0)

        fileformats = ['png', 'pdf', 'eps', 'jpg', '', 'None']
        for section in ['clustering']:
            test = self.parser.get('clustering', 'debug_plots').lower() in fileformats
            if not test:
                if comm.rank == 0:
                    print_and_log(["debug_plots in [%s] should be in %s" % (section, str(fileformats))], 'error', logger)
                sys.exit(0)

        methods = ['distance', 'dip', 'folding', 'nd-folding', 'bhatta', 'nd-bhatta']
        test = self.parser.get('clustering', 'merging_method').lower() in methods
        if not test:
            if comm.rank == 0:
                print_and_log(["merging_method in [%s] should be in %s" % (section, str(methods))], 'error', logger)
            sys.exit(0)

        if self.parser.get('clustering', 'merging_param').lower() == 'default':
            method = self.parser.get('clustering', 'merging_method').lower()
            if method == 'dip':
                self.parser.set('clustering', 'merging_param', '0.5')
            elif method == 'distance':
                self.parser.set('clustering', 'merging_param', '3')
            elif method in ['folding', 'nd-folding']:
                self.parser.set('clustering', 'merging_param', '1e-9')
            elif method in ['bhatta', 'nd-bhatta']:
                self.parser.set('clustering', 'merging_param', '2')

        has_same_elec = self.parser.has_option('clustering', 'sim_same_elec')
        has_dip_thresh = self.parser.has_option('clustering', 'dip_threshold')

        if has_dip_thresh:
            dip_threshold = self.parser.getfloat('clustering', 'dip_threshold')

        if has_dip_thresh and dip_threshold > 0:
          if comm.rank == 0:
            print_and_log(["dip_threshold in [clustering] is deprecated since 0.8.4",
                           "and you should now use merging_method and merging_param",
                           "Please upgrade your parameter file to a more recent version",
                           "By default a nd-bhatta merging method with param 3 is assumed"], 'info', logger)
          self.parser.set('clustering', 'merging_param', str(3))
          self.parser.set('clustering', 'merging_method', 'distance')
        elif has_same_elec:
          sim_same_elec = self.parser.get('clustering', 'sim_same_elec')
          if comm.rank == 0:
            print_and_log(["sim_same_elec in [clustering] is deprecated since 0.8.4",
                           "and you should now use merging_method and merging_param",
                           "Please upgrade your parameter file to a more recent version",
                           "Meanwhile a distance merging method with param %s is assumed" %sim_same_elec], 'info', logger)
          self.parser.set('clustering', 'merging_param', sim_same_elec)
          self.parser.set('clustering', 'merging_method', 'distance')

        dispersion = self.parser.get('clustering', 'dispersion').replace('(', '').replace(')', '').split(',')
        dispersion = map(float, dispersion)
        test = (0 < dispersion[0]) and (0 < dispersion[1])
        if not test:
            if comm.rank == 0:
                print_and_log(["min and max dispersions in [clustering] should be positive"], 'error', logger)
            sys.exit(0)

        pcs_export = ['prompt', 'none', 'all', 'some']
        test = self.parser.get('converting', 'export_pcs').lower() in pcs_export
        if not test:
            if comm.rank == 0:
                print_and_log(["export_pcs in [converting] should be in %s" % str(pcs_export)], 'error', logger)
            sys.exit(0)
        else:
            if self.parser.get('converting', 'export_pcs').lower() == 'none':
                self.parser.set('converting', 'export_pcs', 'n')
            elif self.parser.get('converting', 'export_pcs').lower() == 'some':
                self.parser.set('converting', 'export_pcs', 's')
            elif self.parser.get('converting', 'export_pcs').lower() == 'all':
                self.parser.set('converting', 'export_pcs', 'a')

        if self.parser.getboolean('detection', 'hanning'):
            if comm.rank == 0:
                print_and_log(["Hanning filtering is activated"], 'debug', logger)

    def get(self, section, data, check=True):
        """
        Gets the value of the  variable data in a section as a string.

        Parameters
        ----------
        section :  str
            the section in *params to be read (e.g., 'detection')
        
        data : str
            the variable data to be read (e.g., 'oversampling_factor')

        Returns
        -------
        str 
            The value of the variable data.    
        
        """
        if check is False:
            return self.parser.get(section, data)

        try:
            return self.parser.get(section, data)
        except Exception:
            if comm.rank == 0:
                print_and_log(["Parameter %s is missing in section [%s]" % (data, section)], 'error', logger)
            sys.exit(0)

    def getboolean(self, section, data, check=True):
        """
        Gets the boolean variable from the variable data in a section

        Parameters
        ----------
        section :  str
            the section in *params to be read (e.g., 'filtering')
        
        data : str
            the variable data to be read (e.g., 'remove_median')

        Returns
        -------
        bool
            if the variable data is applied or not.    
        
        """
        if check is False:
            return self.parser.getboolean(section, data)

        try:
            return self.parser.getboolean(section, data)
        except Exception:
            if comm.rank == 0:
                print_and_log(["Parameter %s is missing in section [%s]" % (data, section)], 'error', logger)
            sys.exit(0)

    def getfloat(self, section, data, check=True):
        """
        Gets the value of the  variable data in a section

        Parameters
        ----------
        section :  str
            the section in *params to be read (e.g., 'clustering')
        
        data : str
            the variable data to be read (e.g., 'nb_elts')

        Returns
        -------
        float 
            The value of the variable data.    
        
        """
        if check is False:
            return self.parser.getfloat(section, data)

        try:
            return self.parser.getfloat(section, data)
        except Exception:
            if comm.rank == 0:
                print_and_log(["Parameter %s is missing in section [%s]" % (data, section)], 'error', logger)
            sys.exit(0)

    def getint(self, section, data, check=True):
        """
        Gets the value of the  variable data in a section

        Parameters
        ----------
        section :  str
            the section in *params to be read (e.g., 'detection')
        
        data : str
            the variable data to be read (e.g., 'oversampling_factor')

        Returns
        -------
        int 
            The value of the variable data.    
        
        """
        if check is False:
            return self.parser.getint(section, data)

        try:
            return self.parser.getint(section, data)
        except Exception:
            if comm.rank == 0:
                print_and_log(["Parameter %s is missing in section [%s]" % (data, section)], 'error', logger)
            sys.exit(0)

    def set(self, section, data, value):
        """
        Assigns a value to a variable data in a section

        Parameters
        ----------
        section :  str
            the section in *params to be read (e.g., 'detection')
        
        data : {int, float, str}
            the variable data to be read (e.g., 'oversampling_factor')

        """

        try:
            myval = str(value)
        except Exception as ex:
            print('"%s" cannot be converted to str: %s' % (value, ex))

        self.parser.set(section, data, myval)

    def _update_rate_values(self):
        """
        Updates the values in sampling points of the following values:

        - template width (N_t) in [detection] 
        - minimal distance between peaks (dist_peaks) in [detection]
        - the template shift (template_shift) in [detection]
        - the jitter range (jitter_range) in [detection]
        - the (chunk_size) in [data, whitening, fitting]
        - the (safety_time) in [clustering, whitening, extracting]
        - the (refractory) in [fitting]
        """

        if self._N_t is None:

            if comm.rank == 0:
                print_and_log(['Changing all values in the param depending on the rate'], 'debug', logger)

            spike_width = self.getfloat('detection', 'spike_width')
            self._N_t = self.getfloat('detection', 'N_t')

            # template width from millisecond to sampling points
            self._N_t = int(self.rate * self._N_t * 1e-3) 
            if numpy.mod(self._N_t, 2) == 0:
                self._N_t += 1
            self.set('detection', 'N_t', self._N_t)
            self.set('detection', 'dist_peaks', self._N_t)
            self.set('detection', 'template_shift', (self._N_t - 1) // 2)
            self.set('detection', 'spike_width', int(self.rate * spike_width * 1e-3))

            # jitter_range form millisecond sampling points
            jitter = self.getfloat('detection', 'jitter_range')
            jitter_range = int(self.rate * jitter * 1e-3)
            self.set('detection', 'jitter_range', jitter_range)

            if 'chunk' in self.parser._sections['fitting']:
                self.parser.set('fitting', 'chunk_size', 
                    self.parser._sections['fitting']['chunk'])

            # savgol from millisecond to sampling points
            savgol_time = self.getfloat('clustering', 'savgol_time')
            self._savgol = int(self.rate * savgol_time * 1e-3)
            if numpy.mod(self._savgol, 2) == 0:
                self._savgol += 1

            self.set('clustering', 'savgol_window', self._savgol)

            # noise from millisecond to sampling points
            noise_time = self.getfloat('detection', 'noise_time')
            self._noise = int(self.rate * noise_time * 1e-3)

            self.set('detection', 'noise_time', self._noise)

            over_factor = self.getfloat('detection', 'oversampling_factor')
            nb_jitter = int(over_factor * 2 * jitter_range)
            if numpy.mod(nb_jitter, 2) == 0:
              nb_jitter += 1
            self.set('detection', 'nb_jitter', nb_jitter)

            # chunk_size from second to sampling points
            for section in ['data', 'whitening', 'fitting']:
                chunk = self.parser.getfloat(section, 'chunk_size')
                chunk_size = int(chunk * self.rate)
                self.set(section, 'chunk_size', chunk_size)

            # safety_time from millisecond to sampling points
            for section in ['clustering', 'whitening', 'extracting']:
                safety_time = self.get(section, 'safety_time')
                if safety_time == 'auto':
                    self.set(section, 'safety_time', self._N_t // 3)
                else:
                    safety_time = int(float(safety_time) * self.rate * 1e-3)
                    self.set(section, 'safety_time', safety_time)
                    
            # refractory from millisecond to sampling points
            refractory = self.getfloat('fitting', 'refractory')
            self.set('fitting', 'refractory', int(refractory*self.rate * 1e-3))

    def _create_data_file(self, data_file, is_empty, params, stream_mode):
        """
        Creates a data file with parameters from the param file.

        Parameters
        ----------

        data_file : str 
            the path for the datafile
        
        is_empty : bool

        params :  dict

        stream_mode : str
            multi-files, multi-folders or single-file.

        Returns
        -------
        dict
            a supported data file with the parameters from the param file.
        """
        file_format = params.pop('file_format').lower()
        if comm.rank == 0:
            print_and_log(['Trying to read file %s as %s' %(data_file, file_format)], 'debug', logger)

        data = __supported_data_files__[file_format](data_file, params, is_empty, stream_mode)
        self.rate = data.sampling_rate
        self.nb_channels = data.nb_channels
        self.gain = data.gain
        self.data_file = data
        self._update_rate_values()

        N_e = self.getint('data', 'N_e')
        if N_e > self.nb_channels:
            if comm.rank == 0:
                lines = ['Analyzed %d channels but only %d are recorded' % (N_e, self.nb_channels)]
                print_and_log(lines, 'error', logger)
            sys.exit(0)

        return data

    def get_data_file(self, is_empty=False, params=None, source=False, has_been_created=True):
        """
        Gets the datafile as described in the param files.

        Parameters
        ----------
        is_empty : bool

        params : dict

        source : bool

        has_been_created : bool
            if the data file was 

        Returns
        -------
        dict   
            A dictionary with the parameters of created data file.

        """

        if params is None:
            params = {}

        for key, value in self.parser._sections['data'].items():
            if key not in params:
                params[key] = value

        data_file = params.pop('data_file')
        stream_mode = self.get('data', 'stream_mode').lower()

        if stream_mode in ['none']:
            stream_mode = None

        if not self.getboolean('data', 'overwrite'):
            # If we do not want to overwrite, we first read the original data file
            # Then, if we do not want to obtain it as a source file, we switch the
            # format to raw_binary and the output file name

            if not source:

                # First we read the original data file, that should not be empty
                print_and_log(['Reading first the real data file to get the parameters'], 'debug', logger)
                tmp = self._create_data_file(data_file, False, params, stream_mode)

                # Then we change the dataa_file name
                data_file = self.get('data', 'data_file_no_overwrite')

                if comm.rank == 0:
                    print_and_log(['Forcing the exported data file to be of type raw_binary'], 'debug', logger)

                # And we force the results to be of type float32, without streams
                params['file_format'] = 'raw_binary'
                params['data_dtype'] = 'float32'
                params['dtype_offset'] = 0
                params['data_offset'] = 0
                params['sampling_rate'] = self.rate
                params['nb_channels'] = self.nb_channels
                params['gain'] = self.gain
                stream_mode = None
                data_file, extension = os.path.splitext(data_file)
                data_file += ".dat"
            else:
                if has_been_created:
                    data_file = self.get('data', 'data_file_no_overwrite')
                    if not os.path.exists(data_file):
                        if comm.rank == 0:
                            lines = ['The overwrite option is only valid if the filtering step is launched before!']
                            print_and_log(lines, 'error', logger)
                        sys.exit(0)
                else:
                    if comm.rank == 0:
                        print_and_log(['The copy file has not yet been created! Returns normal file'], 'debug', logger)


        return self._create_data_file(data_file, is_empty, params, stream_mode)

    def write(self, section, flag, value, preview_path=False):
        """
        Writes the section of the param file with 

        Parameters
        ----------
        section : str
            a section in the param file

        flag : str

            a variable of a section in the param file

        value : str

            the value of the variable data in a section of param file.
        """
        if comm.rank == 0:
            print_and_log(['Writing value %s for %s:%s' % (value, section, flag)], 'debug', logger)
        self.parser.set(section, flag, value)
        if preview_path:
            f = open(self.get('data', 'preview_path'), 'r')
        else:
            f = open(self.file_params, 'r')

        lines = f.readlines()
        f.close()
        spaces = ''.join([' ']*(max(0, 15 - len(flag))))

        to_write = '%s%s= %s              #!! AUTOMATICALLY EDITED: DO NOT MODIFY !!\n' % (flag, spaces, value)

        section_area = [0, len(lines)]
        idx = 0
        for count, line in enumerate(lines):

            if (idx == 1) and line.strip().replace('[', '').replace(']', '') in self.__all_sections__:
                section_area[idx] = count
                idx += 1

            if line.find('[%s]' % section) > -1:
                section_area[idx] = count
                idx += 1

        has_been_changed = False

        for count in range(section_area[0]+1, section_area[1]):
            if '=' in lines[count]:
                key = lines[count].split('=')[0].replace(' ', '')
                if key == flag:
                    lines[count] = to_write
                    has_been_changed = True

        if not has_been_changed:
            lines.insert(section_area[1]-1, to_write)

        if preview_path:
            f = open(self.get('data', 'preview_path'), 'w')
        else:
            f = open(self.file_params, 'w')
        for line in lines:
            f.write(line)
        f.close()
