import ConfigParser as configparser
from messages import print_and_log
from circus.shared.probes import read_probe
from circus.shared.mpi import comm
from circus.files import __supported_data_files__

import os, sys, copy, numpy, logging

logger = logging.getLogger(__name__)

class CircusParser(object):

    __all_sections__ = ['data', 'whitening', 'extracting', 'clustering', 
                       'fitting', 'filtering', 'merging', 'noedits', 'triggers', 
                       'detection', 'validating', 'converting']

    __default_values__ = [['fitting', 'amp_auto', 'bool', 'True'], 
                          ['fitting', 'refractory', 'float', '0.5'],
                          ['fitting', 'collect_all', 'bool', 'False'],
                          ['data', 'global_tmp', 'bool', 'True'],
                          ['data', 'chunk_size', 'int', '30'],
                          ['data', 'stream_mode', 'string', 'None'],
                          ['data', 'overwrite', 'bool', 'True'],
                          ['detection', 'alignment', 'bool', 'True'],
                          ['detection', 'matched-filter', 'bool', 'False'],
                          ['detection', 'matched_thresh', 'float', '5'],
                          ['detection', 'peaks', 'string', 'negative'],
                          ['detection', 'spike_thresh', 'float', '6'],
                          ['triggers', 'clean_artefact', 'bool', 'False'],
                          ['triggers', 'make_plots', 'string', 'png'],
                          ['triggers', 'trig_file', 'string', ''],
                          ['triggers', 'trig_windows', 'string', ''],
                          ['triggers', 'trig_unit', 'string', 'ms'],
                          ['whitening', 'chunk_size', 'int', '30'],
                          ['filtering', 'remove_median', 'bool', 'False'],
                          ['clustering', 'max_clusters', 'int', '10'],
                          ['clustering', 'nb_repeats', 'int', '3'],
                          ['clustering', 'make_plots', 'string', 'png'],
                          ['clustering', 'test_clusters', 'bool', 'False'],
                          ['clustering', 'sim_same_elec', 'float', '2'],
                          ['clustering', 'smart_search', 'bool', 'True'],
                          ['clustering', 'smart_select', 'bool', 'False'],
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
                          ['converting', 'export_all', 'bool', 'False'],
                          ['converting', 'sparse_export', 'bool', 'False'],
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
                          ['noedits', 'median_done', 'bool', 'False'],
                          ['noedits', 'artefacts_done', 'bool', 'False']]

    __extra_values__ = [['fitting', 'space_explo', 'float', '0.5'],
                        ['fitting', 'nb_chances', 'int', '3'],
                        ['clustering', 'm_ratio', 'float', '0.01'],
                        ['clustering', 'sub_dim', 'int', '5']]

    def __init__(self, file_name, **kwargs):

        self.file_name    = os.path.abspath(file_name)
        f_next, extension = os.path.splitext(self.file_name)
        file_path         = os.path.dirname(self.file_name)
        self.file_params  = f_next + '.params'
        self.logfile      = f_next + '.log'
        self.parser       = configparser.ConfigParser()
        self._N_t         = None

        if not os.path.exists(self.file_params):
            if comm.rank == 0:
                print_and_log(["%s does not exist" %self.file_params], 'error', logger)
            sys.exit(1)

        if comm.rank == 0:
            print_and_log(['Creating a Circus Parser for datafile %s' %self.file_name], 'debug', logger)
        self.parser.read(self.file_params)
    
        for section in self.__all_sections__:
            if self.parser.has_section(section):
                for (key, value) in self.parser.items(section):
                    self.parser.set(section, key, value.split('#')[0].replace(' ', '').replace('\t', '')) 
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


        self.probe = read_probe(self.parser)

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
            sys.exit(1)

        to_write = ["You must specify explicitly the file format in the config file", 
                    "Please have a look to the documentation and add a file_format", 
                    "parameter in the [data] section. Valid files formats can be:", '']
        try:
            self.file_format = self.parser.get('data', 'file_format')
        except Exception:
            if comm.rank == 0:
                for f in __supported_data_files__.keys():
                    to_write += ['-- %s -- %s' %(f, __supported_data_files__[f].extension)]

                to_write += ['', "To get more info on a given file format, see", 
                    ">> spyking-circus file_format -i"]

                print_and_log(to_write, 'error', logger)
            sys.exit(1)	

        test = self.file_format.lower() in __supported_data_files__.keys()
        if not test:
            if comm.rank == 0:
                for f in __supported_data_files__.keys():
                    to_write += ['-- %s -- %s' %(f, __supported_data_files__[f].extension)]

                to_write += ['', "To get more info on a given file format, see", 
                    ">> spyking-circus file_format -i"]

                print_and_log(to_write, 'error', logger)
            sys.exit(1) 


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
                sys.exit(1)

        units = ['ms', 'timestep']
        test = self.parser.get('triggers', 'trig_unit').lower() in units
        if not test:
          if comm.rank == 0:
              print_and_log(["trig_unit in [triggers] should be in %s" %str(units)], 'error', logger)
          sys.exit(1)
        else:
          self.parser.set('triggers', 'trig_in_ms', str(self.parser.get('triggers', 'trig_unit').lower() == 'ms'))
    
        self.parser.set('triggers', 'trig_file', os.path.abspath(os.path.expanduser(self.parser.get('triggers', 'trig_file'))))
        self.parser.set('triggers', 'trig_windows', os.path.abspath(os.path.expanduser(self.parser.get('triggers', 'trig_windows'))))

        test = (self.parser.get('clustering', 'extraction').lower() in ['median-raw', 'median-pca', 'mean-raw', 'mean-pca'])
        if not test:
            if comm.rank == 0:
                print_and_log(["Only 4 extraction modes in [clustering]: median-raw, median-pca, mean-raw or mean-pca!"], 'error', logger)
            sys.exit(1)

        test = (self.parser.get('detection', 'peaks').lower() in ['negative', 'positive', 'both'])
        if not test:
            if comm.rank == 0:
                print_and_log(["Only 3 detection modes for peaks in [detection]: negative, positive, both"], 'error', logger)

        try:
            os.makedirs(f_next)
        except Exception:
            pass

        self.parser.set('data', 'data_file', file_name)        
        self.parser.set('data', 'data_file_no_overwrite', os.path.join(file_path, os.path.basename(f_next) + '_all_sc.dat'))
        file_out = os.path.join(f_next, os.path.basename(f_next))
        self.parser.set('data', 'file_out', file_out) # Output file without suffix
        self.parser.set('data', 'file_out_suff', file_out  + self.parser.get('data', 'suffix')) # Output file with suffix
        self.parser.set('data', 'data_file_noext', f_next)   # Data file (assuming .filtered at the end)   

        for section in ['whitening', 'clustering']:
            test = (self.parser.getfloat(section, 'nb_elts') > 0) and (self.parser.getfloat(section, 'nb_elts') <= 1)
            if not test: 
                if comm.rank == 0:
                    print_and_log(["nb_elts in [%s] should be in [0,1]" %section], 'error', logger)
                sys.exit(1)

        test = (self.parser.getfloat('clustering', 'nclus_min') >= 0) and (self.parser.getfloat('clustering', 'nclus_min') < 1)
        if not test:
            if comm.rank == 0:
                print_and_log(["nclus_min in [clustering] should be in [0,1["], 'error', logger)
            sys.exit(1)

        test = (self.parser.getfloat('clustering', 'noise_thr') >= 0) and (self.parser.getfloat('clustering', 'noise_thr') <= 1)
        if not test:
            if comm.rank == 0:
                print_and_log(["noise_thr in [clustering] should be in [0,1]"], 'error', logger)
            sys.exit(1)

        test = (self.parser.getfloat('validating', 'test_size') > 0) and (self.parser.getfloat('validating', 'test_size') < 1)
        if not test:
            if comm.rank == 0:
                print_and_log(["test_size in [validating] should be in ]0,1["], 'error', logger)
            sys.exit(1)

        test = (self.parser.getfloat('clustering', 'cc_merge') >= 0) and (self.parser.getfloat('clustering', 'cc_merge') <= 1)
        if not test:
            if comm.rank == 0:
                print_and_log(["cc_merge in [validating] should be in [0,1]"], 'error', logger)
            sys.exit(1)

        test = (self.parser.getfloat('clustering', 'nclus_min') >= 0) and (self.parser.getfloat('clustering', 'nclus_min') < 1)
        if not test:
            if comm.rank == 0:
                print_and_log(["nclus_min in [validating] should be in [0,1["], 'error', logger)
            sys.exit(1)

        fileformats = ['png', 'pdf', 'eps', 'jpg', '', 'None']
        for section in ['clustering', 'validating', 'triggers']:
          test = self.parser.get('clustering', 'make_plots').lower() in fileformats
          if not test:
              if comm.rank == 0:
                  print_and_log(["make_plots in [%s] should be in %s" %(section, str(fileformats))], 'error', logger)
              sys.exit(1)

        dispersion     = self.parser.get('clustering', 'dispersion').replace('(', '').replace(')', '').split(',')
        dispersion     = map(float, dispersion)
        test =  (0 < dispersion[0]) and (0 < dispersion[1])
        if not test:
            if comm.rank == 0:
                print_and_log(["min and max dispersions in [clustering] should be positive"], 'error', logger)
            sys.exit(1)
        

        pcs_export = ['prompt', 'none', 'all', 'some']
        test = self.parser.get('converting', 'export_pcs').lower() in pcs_export
        if not test:
            if comm.rank == 0:
                print_and_log(["export_pcs in [converting] should be in %s" %str(pcs_export)], 'error', logger)
            sys.exit(1)
        else:
            if self.parser.get('converting', 'export_pcs').lower() == 'none':
                self.parser.set('converting', 'export_pcs', 'n')
            elif self.parser.get('converting', 'export_pcs').lower() == 'some':
                self.parser.set('converting', 'export_pcs', 's')
            elif self.parser.get('converting', 'export_pcs').lower() == 'all':
                self.parser.set('converting', 'export_pcs', 'a')
	    

    def get(self, section, data):
      	return self.parser.get(section, data)

    def getboolean(self, section, data):
      	return self.parser.getboolean(section, data)

    def getfloat(self, section, data):
      	return self.parser.getfloat(section, data)

    def getint(self, section, data):
      	return self.parser.getint(section, data)

    def set(self, section, data, value):
        self.parser.set(section, data, value)

    def _update_rate_values(self):

        if self._N_t is None:

            if comm.rank == 0:
                print_and_log(['Changing all values in the param depending on the rate'], 'debug', logger)
    

            try:
                self._N_t = self.getfloat('detection', 'N_t')
            except Exception:
                if comm.rank == 0:
                    print_and_log(['N_t must now be defined in the [detection] section'], 'error', logger)
                sys.exit(1)

            self._N_t = int(self.rate*self._N_t*1e-3)
            if numpy.mod(self._N_t, 2) == 0:
                self._N_t += 1

            self.set('detection', 'N_t', str(self._N_t))
            self.set('detection', 'dist_peaks', str(self._N_t))
            self.set('detection', 'template_shift', str((self._N_t-1)//2))

            if self.parser._sections['fitting'].has_key('chunk'):
                self.parser.set('fitting', 'chunk_size', self.parser._sections['fitting']['chunk'])

            for section in ['data', 'whitening', 'fitting']:
                chunk_size = int(self.parser.getfloat(section, 'chunk_size') * self.rate)
                self.set(section, 'chunk_size', str(chunk_size))
            
            for section in ['clustering', 'whitening', 'extracting']:
                safety_time = self.get(section, 'safety_time')
                if safety_time == 'auto':
                    self.set(section, 'safety_time', str(self._N_t//3))
                else:
                    safety_time = float(safety_time)
                    self.set(section, 'safety_time', str(int(safety_time*self.rate*1e-3)))

            refactory = self.getfloat('fitting', 'refractory')
            self.set('fitting', 'refactory', str(int(refactory*self.rate*1e-3)))


    def _create_data_file(self, data_file, is_empty, params, stream_mode):
        file_format       = params.pop('file_format').lower()
        if comm.rank == 0:
            print_and_log(['Trying to read file %s as %s' %(data_file, file_format)], 'debug', logger)

        data              = __supported_data_files__[file_format](data_file, params, is_empty, stream_mode)
        self.rate         = data.sampling_rate
        self.nb_channels  = data.nb_channels
        self.gain         = data.gain
        self.data_file    = data
        self._update_rate_values()

        N_e = self.getint('data', 'N_e')
        if N_e > self.nb_channels:
            if comm.rank == 0:
                print_and_log(['Analyzed %d channels but only %d are recorded' %(N_e, self.nb_channels)], 'error', logger)
            sys.exit(1)

        return data


    def get_data_file(self, is_empty=False, params=None, source=False, has_been_created=True):

        if params is None:
            params = {}

        for key, value in self.parser._sections['data'].items():
            if key not in params:
                params[key] = value

        data_file     = params.pop('data_file')
        stream_mode   = self.get('data', 'stream_mode').lower()

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
                params['file_format']   = 'raw_binary'
                params['data_dtype']    = 'float32'
                params['dtype_offset']  = 0
                params['data_offset']   = 0
                params['sampling_rate'] = self.rate
                params['nb_channels']   = self.nb_channels
                params['gain']          = self.gain
                stream_mode             = None            
                data_file, extension    = os.path.splitext(data_file)
                data_file              += ".dat" 
            
            else:
                if has_been_created:
                  data_file = self.get('data', 'data_file_no_overwrite')
                  if not os.path.exists(data_file):
                      if comm.rank== 0:
                          print_and_log(['The overwrite option is only valid if the filtering step is launched before!'], 'error', logger)
                      sys.exit(1)
                else:
                    if comm.rank== 0:
                        print_and_log(['The copy file has not yet been created! Returns normal file'], 'debug', logger)

        return self._create_data_file(data_file, is_empty, params, stream_mode)


    def write(self, section, flag, value):
        if comm.rank == 0:
            print_and_log(['Writing value %s for %s:%s' %(value, section, flag)], 'debug', logger)
        self.parser.set(section, flag, value)
        f     = open(self.file_params, 'r')
        lines = f.readlines()
        f.close()
        spaces = ''.join([' ']*(max(0, 15 - len(flag))))

        to_write = '%s%s= %s              #!! AUTOMATICALLY EDITED: DO NOT MODIFY !!\n' %(flag, spaces, value)
        
        section_area = [0, len(lines)]
        idx          = 0
        for count, line in enumerate(lines):

            if (idx == 1) and line.strip().replace('[', '').replace(']', '') in self.__all_sections__ :
                section_area[idx] = count
                idx += 1

            if (line.find('[%s]' %section) > -1):
                section_area[idx] = count
                idx += 1

        has_been_changed = False

        for count in xrange(section_area[0]+1, section_area[1]):
            if '=' in line:
                key  = lines[count].split('=')[0].replace(' ', '')
                if key == flag:
                    lines[count] = to_write
                    has_been_changed = True

        if not has_been_changed:
            lines.insert(section_area[1]-1, to_write)

        f     = open(self.file_params, 'w')
        for line in lines:
            f.write(line)
        f.close()