#!/usr/bin/env python
import os, sys, socket, getopt, shutil
from termcolor import colored
hostname    = socket.gethostname()

if hostname == 'spikesorter':
    nb_cpu = 14
    nb_gpu = 7
else:
    nb_cpu = 1
    nb_gpu = 1

try:
    import cudamat as cmt   
    cmt.init() 
    HAVE_CUDA = True
except Exception:
    HAVE_CUDA = False
    nb_gpu    = 0

all_steps = ['whitening', 'clustering', 'fitting', 'gathering', 'extracting', 'filtering', 'converting', 'benchmarking']
steps     = ['filtering', 'whitening', 'clustering', 'fitting']
hostfile  = '%s.hosts' %hostname

header = '''
##############################################################
#####          Welcome to the SpyKING CIRCUS             #####
#####                                                    #####
#####          Written by P.Yger and O.Marre             #####
##############################################################
'''

message = '''
Syntax is spyking-circus file [options]

Options are:
-h or --help     : display the help
-m or --method   : by default, first 4 steps of the algorithm are 
                  performed, but if specified, only one can be 
                  done. This has to be among
                    - filtering
                    - whitening
                    - clustering
                    - fitting
                    - (extra) gathering
                    - (extra) extracting
                    - (extra) converting
                    - (extra) benchmarking
                  Three extra steps are also available, but only
                  for custom needs (see documentation)
                  Note that you can give a sequence of steps, 
                  separated by a comma (i.e. -m clustering,fitting)
-c or --cpu      : number of CPU (default 1)
-g or --gpu      : number of GPU (default 1 if CUDA available)
-H or --hostfile : hostfile for MPI (default is hostname.hosts)'''

noparams='''
The parameter file is not present!
You must have a file named %s, properly configured, 
in the same folder, with the data file.'''

if len(sys.argv) < 2:
    print colored(header, 'green'), message
    sys.exit()
else:
    filename   = sys.argv[1]
    if not os.path.exists(filename):
        print colored(header, 'green'), "The data file %s can not be found!" %filename
        sys.exit()
    else:
        extension       = '.' + filename.split('.')[-1]
        file_params     = filename.replace(extension, '.params')
        if not os.path.exists(file_params):
            print colored(header, 'green'), noparams %file_params
            key = ''
            while key not in ['y', 'n']:
                key = raw_input("Do you want SpyKING CIRCUS to create a parameter file? [y/n]")
            if key == 'y':
                print "Generating template file", file_params
                print "Please fill it properly before launching the code! (see documentation)"
                shutil.copy2('circus/config.params', file_params)
            sys.exit()
        opts, args  = getopt.getopt(sys.argv[2:], "hvm:H:c:g:", ["help", "method=", "hostfile=", "cpu=", "gpu="])

for opt, arg in opts:
    if opt in ('-h', '--help'):        
        print colored(header, 'green'), message
        sys.exit()
    elif opt == '-d':
        verbose = True
    elif opt in ('-m', '--method'):
        args = arg.split(',')
        for arg in args:
            if arg not in all_steps:
                print "The method has to be in", all_steps
                sys.exit(0)
        else:
            steps = args
    elif opt in ('-g', '--gpu'):
        nb_gpu = int(arg)
    elif opt in ('-c', '--cpu'):
        nb_cpu = int(arg)
    elif opt in ('-H', '--hostfile'):
        hostfile = arg

print colored(header, 'green')
print "Steps         :", colored(", ".join(steps), 'cyan')
print "GPU detected  :", colored(HAVE_CUDA, 'cyan')
print "Number of CPU :", colored(nb_cpu, 'cyan')
if HAVE_CUDA:
    print "Number of GPU :", colored(nb_gpu, 'cyan')
print "Hostfile      :", colored(hostfile, 'cyan')
print ""
print "##############################################################"
print ""

if not os.path.exists(hostfile):
    host_string = ''
else:
    host_string = '-hostfile %s' %hostfile

os.system('python circus/infos.py %s' %filename)

if 'filtering' in steps:
    os.system('mpirun %s -np %d python circus/filtering.py %s' %(host_string, nb_cpu, filename))
if 'whitening' in steps:
    os.system('mpirun %s -np %d python circus/whitening.py %s' %(host_string, nb_cpu, filename))
    os.system('mpirun %s -np %d python circus/basis.py %s'     %(host_string, nb_cpu, filename))
if 'clustering' in steps:
    os.system('mpirun %s -np %d python circus/clustering.py %s' %(host_string, nb_cpu, filename))
if 'fitting' in steps:
    if HAVE_CUDA and nb_gpu > 0:
        os.system('mpirun -x LD_LIBRARY_PATH %s -np %d python circus/fitting.py %s True' %(host_string, nb_gpu, filename))
    else:
        os.system('mpirun %s -np %d python circus/fitting.py %s False' %(host_string, nb_cpu, filename))
if 'extracting' in steps:
    os.system('mpirun %s -np %d python circus/extracting.py %s' %(host_string, nb_cpu, filename))
if 'gathering' in steps:
    os.system('python circus/gathering.py %s %s' %(nb_cpu, filename))
if 'converting' in steps:
    os.system('python circus/export_phy.py %s' %(filename))
if 'benchmarking' in steps:
    os.system('mpirun %s -np %d python circus/synthetic.py %s' %(host_string, nb_cpu, filename))

