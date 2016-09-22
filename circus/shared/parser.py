import ConfigParser as configparser
from messages import print_error, print_and_log
from circus.shared.probes import read_probe
from circus.files import __supported_data_files__
	
import os, sys, copy, StringIO

class CircusParser(object):

    __all_section__ = ['data', 'whitening', 'extracting', 'clustering', 
                       'fitting', 'filtering', 'merging', 'noedits', 'triggers', 
                       'detection', 'validating', 'converting']

    __default_values__ = [['fitting', 'amp_auto', 'bool', 'True'], 
		                  ['fitting', 'refractory', 'float', '0.5'],
		                  ['fitting', 'collect_all', 'bool', 'False'],
		                  ['data', 'global_tmp', 'bool', 'True'],
		                  ['data', 'chunk_size', 'int', '30'],
		                  ['data', 'multi-files', 'bool', 'False'],
		                  ['detection', 'alignment', 'bool', 'True'],
		                  ['detection', 'matched-filter', 'bool', 'False'],
		                  ['detection', 'matched_thresh', 'float', '5'],
		                  ['detection', 'peaks', 'string', 'negative'],
		                  ['detection', 'spike_thresh', 'float', '6'],
		                  ['triggers', 'clean_artefact', 'bool', 'False'],
		                  ['triggers', 'make_plots', 'string', 'png'],
		                  ['triggers', 'trig_file', 'string', ''],
		                  ['triggers', 'trig_windows', 'string', ''],
		                  ['whitening', 'chunk_size', 'int', '30'],
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
		                  ['converting', 'export_all', 'bool', 'False'],
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

    __extra_values__ = [['fitting', 'space_explo', 'float', '0.5'],
                        ['fitting', 'nb_chances', 'int', '3'],
                        ['clustering', 'm_ratio', 'float', '0.01'],
                        ['clustering', 'sub_dim', 'int', '5']]


    def __init__(self, file_name):

        self.file_name    = os.path.abspath(file_name)
        f_next, extension = os.path.splitext(self.file_name)
        file_path         = os.path.dirname(self.file_name)
        self.file_params  = f_next + '.params'
        self.parser       = configparser.ConfigParser()

        if not os.path.exists(self.file_params):
            print_error(["%s does not exist" %self.file_params])
            sys.exit(0)

        self.parser.read(self.file_params)
	    
        for section in self.__all_section__:
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

        self.probe = read_probe(self.parser)


        N_e = 0
        for key in self.probe['channel_groups'].keys():
            N_e += len(self.probe['channel_groups'][key]['channels'])

        self.set('data', 'N_e', str(N_e))      
        self.set('data', 'N_total', str(self.probe['total_nb_channels']))      

        '''
        try:
            self.file_format = parser.get('data', 'file_format')
        except Exception:
            print_error(["Now you must specify explicitly the file format in the config file", 
                       "Please have a look to the documentation and add a file_format", 
                       "parameter in the [data] section. Valid files formats can be:",
                       "", 
                       ", ".join(__supported_data_files__.keys())])
            sys.exit(0)
        '''
	

        try: 
            self.parser.get('detection', 'radius')
        except Exception:
            self.parser.set('detection', 'radius', 'auto')
        try:
            self.parser.getint('detection', 'radius')
        except Exception:
            self.parser.set('detection', 'radius', str(int(self.probe['radius'])))	    

        if self.parser.getboolean('data', 'multi-files'):
            self.parser.set('data', 'data_multi_file', file_name)
            pattern     = os.path.basename(file_name).replace('0', 'all')
            multi_file  = os.path.join(file_path, pattern)
            self.parser.set('data', 'data_file', multi_file)
            f_next, extension = os.path.splitext(multi_file)
        else:
            self.parser.set('data', 'data_file', file_name)

        if self.parser.getboolean('triggers', 'clean_artefact'):
            if (self.parser.get('triggers', 'trig_file') == '') or (self.parser.get('triggers', 'trig_windows') == ''):
                print_and_log(["trig_file and trig_windows must be specified"], 'error', self.parser)
                sys.exit(0)
	    
        self.parser.set('triggers', 'trig_file', os.path.abspath(os.path.expanduser(self.parser.get('triggers', 'trig_file'))))
        self.parser.set('triggers', 'trig_windows', os.path.abspath(os.path.expanduser(self.parser.get('triggers', 'trig_windows'))))

        test = (self.parser.get('clustering', 'extraction') in ['median-raw', 'median-pca', 'mean-raw', 'mean-pca'])
        if not test:
            print_and_log(["Only 5 extraction modes: median-raw, median-pca, mean-raw or mean-pca!"], 'error', self.parser)
            sys.exit(0)

        test = (self.parser.get('detection', 'peaks') in ['negative', 'positive', 'both'])
        if not test:
            print_and_log(["Only 3 detection modes for peaks: negative, positive, both"], 'error', self.parser)
            sys.exit(0)

        try:
            os.makedirs(f_next)
        except Exception:
            pass

        file_out = os.path.join(f_next, os.path.basename(f_next))
        self.parser.set('data', 'file_out', file_out) # Output file without suffix
        self.parser.set('data', 'file_out_suff', file_out  + self.parser.get('data', 'suffix')) # Output file with suffix
        self.parser.set('data', 'data_file_noext', f_next)   # Data file (assuming .filtered at the end)   

        for section in ['whitening', 'clustering']:
            test = (self.parser.getfloat(section, 'nb_elts') > 0) and (self.parser.getfloat(section, 'nb_elts') <= 1)
            if not test: 
                print_and_log(["nb_elts in %s should be in [0,1]" %section], 'error', self.parser)
                sys.exit(0)

        test = (self.parser.getfloat('clustering', 'nclus_min') >= 0) and (self.parser.getfloat('clustering', 'nclus_min') < 1)
        if not test:
            print_and_log(["nclus_min in clustering should be in [0,1["], 'error', self.parser)
            sys.exit(0)

        test = (self.parser.getfloat('clustering', 'noise_thr') >= 0) and (self.parser.getfloat('clustering', 'noise_thr') <= 1)
        if not test:
            print_and_log(["noise_thr in clustering should be in [0,1]"], 'error', self.parser)
            sys.exit(0)

        test = (self.parser.getfloat('validating', 'test_size') > 0) and (self.parser.getfloat('validating', 'test_size') < 1)
        if not test:
            print_and_log(["test_size in validating should be in ]0,1["], 'error', self.parser)
            sys.exit(0)

        fileformats = ['png', 'pdf', 'eps', 'jpg', '', 'None']
        test = self.parser.get('clustering', 'make_plots') in fileformats
        if not test:
            print_and_log(["make_plots in clustering should be in %s" %str(fileformats)], 'error', self.parser)
            sys.exit(0)
        test = self.parser.get('validating', 'make_plots') in fileformats
        if not test:
            print_and_log(["make_plots in clustering should be in %s" %str(fileformats)], 'error', self.parser)
            sys.exit(0)
	    
        dispersion     = self.parser.get('clustering', 'dispersion').replace('(', '').replace(')', '').split(',')
        dispersion     = map(float, dispersion)
        test =  (0 < dispersion[0]) and (0 < dispersion[1])
        if not test:
            print_and_log(["min and max dispersions should be positive"], 'error', self.parser)
            sys.exit(0)
	        

        pcs_export = ['prompt', 'none', 'all', 'some']
        test = self.parser.get('converting', 'export_pcs') in pcs_export
        if not test:
            print_and_log(["export_pcs in converting should be in %s" %str(pcs_export)], 'error', self.parser)
            sys.exit(0)
        else:
            if self.parser.get('converting', 'export_pcs') == 'none':
                self.parser.set('converting', 'export_pcs', 'n')
            elif self.parser.get('converting', 'export_pcs') == 'some':
                self.parser.set('converting', 'export_pcs', 's')
            elif self.parser.get('converting', 'export_pcs') == 'all':
                self.parser.set('converting', 'export_pcs', 'a')
	    

    def get(self, data, section):
      	return self.parser.get(data, section)

    def getboolean(self, data, section):
      	return self.parser.getboolean(data, section)

    def getfloat(self, data, section):
      	return self.parser.getfloat(data, section)

    def getint(self, data, section):
      	return self.parser.getint(data, section)

    def set(self, data, section, value):
        self.parser.set(data, section, value)


    def _create_data_file(self, params, is_empty=False):
        keys        = __supported_data_files__.keys()
        
        file_format = params['file_format']
        data_file   = params['data_file']

        if file_format not in keys:
            print_error(["The type %s is not recognized as a valid file format" %file_format, 
                         "Valid files formats can be:", 
                         ", ".join(keys)])
            sys.exit(0)
        else:
            data = __supported_data_files__[file_format](data_file, params, is_empty)

        return data

    def get_data_file(self, multi=False, is_empty=False, force_raw=False):

        ## A bit tricky because we want to deal with multifiles export
        # If multi is False, we use the default REAL data files
        # If multi is True, we use the combined file of all data files

        several_files = self.parser.getboolean('data', 'multi-files')
        
        params        = self.parser._sections['data']

        '''
        if force_raw == True:  
            data = self._create_data_file(params, is_empty)

            params['file_format'] = 'raw_binary'
            params['data_dtype']  = data.data_dtype
            params['data_offset'] = 0
            params['']
        elif force_raw == True:
            self.parser.set('data', 'file_format', 'raw_binary')
        '''

        if not multi:
            data_file = self.parser.get('data', 'data_file')    
        else:
            data_file = self.parser.get('data', 'data_multi_file')

        file_format = self.parser.get('data', 'file_format').lower()
        print file_format, data_file, params

        
        return self._create_data_file(params, is_empty)


    def get_multi_files(self):
        file_name   = self.parser.get('data', 'data_multi_file')
        dirname     = os.path.abspath(os.path.dirname(file_name))
        all_files   = os.listdir(dirname)
        pattern     = os.path.basename(file_name)
        to_process  = []
        count       = 0

        while pattern in all_files:
            to_process += [os.path.join(os.path.abspath(dirname), pattern)]
            pattern     = pattern.replace(str(count), str(count+1))
            count      += 1

        print_and_log(['Multi-files:'] + to_process, 'debug', self.parser)
        return to_process

    def write(self, section, flag, value):
        f     = open(self.file_params, 'r')
        lines = f.readlines()
        f.close()
        to_write = '%s      = %s              #!! AUTOMATICALLY EDITED: DO NOT MODIFY !!\n' %(flag, value)
        
        section_area = [0, len(lines)]
        idx          = 0
        for count, line in enumerate(lines):

            if (idx == 1) and line.strip().replace('[', '').replace(']', '') in self.__all_section__ :
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