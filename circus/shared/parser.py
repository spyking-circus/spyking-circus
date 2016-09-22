import ConfigParser as configparser
from messages import print_error, print_and_log
from circus.shared.probes import read_probe
#from circus.files import __supported_data_files__
	
import os, sys, copy


class CircusParser(object):

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

	__all_section__ = ['data', 'whitening', 'extracting', 'clustering', 
	                   'fitting', 'filtering', 'merging', 'noedits', 'triggers', 
	                   'detection', 'validating', 'converting']


	def __init__(self, file_name):

	    file_name         = os.path.abspath(file_name)
	    f_next, extension = os.path.splitext(file_name)
	    file_path         = os.path.dirname(file_name)
	    self.file_params  = f_next + '.params'
	    parser            = configparser.ConfigParser()
	    self.original     = copy.copy(parser)
	    if not os.path.exists(self.file_params):
	        print_error(["%s does not exist" %self.file_params])
	        sys.exit(0)

	    parser.read(self.file_params)
	    
	    for section in self.__all_section__:
	        if parser.has_section(section):
	            for (key, value) in parser.items(section):
	                parser.set(section, key, value.split('#')[0].replace(' ', '').replace('\t', '')) 
	        else:
	            parser.add_section(section)

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
	        parser.get('detection', 'radius')
	    except Exception:
	        parser.set('detection', 'radius', 'auto')
	    try:
	        parser.getint('detection', 'radius')
	    except Exception:
	        parser.set('detection', 'radius', str(int(probe['radius'])))

	    

	    for item in self.__default_values__:
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

	    try:
	      parser.get('data', 'file_format')
	    except Exception:
	      print_error(["Now you must specify explicitly the file format in the config file", 
	                   "Please have a look to the documentation and add a file_format", 
	                   "parameter in the [data] section. Valid files formats can be:",
	                   "", 
	                   ", ".join(__supported_data_files__.keys())])
	      sys.exit(0)

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
	    

	    self.parser = parser


	def get(self, data, section):
	  	return self.parser.get(data, section)

	def getboolean(self, data, section):
	  	return self.parser.getboolean(data, section)

	def getfloat(self, data, section):
	  	return self.parser.getfloat(data, section)

	def getint(self, data, section):
	  	return self.parser.getint(data, section)

	def set(self, data, section, value):
	   	return self.original.set(data, section, value)

	def save(self):
		f = open(self.file_params, 'wb')
		self.original.write(f)
		f.close()
    
