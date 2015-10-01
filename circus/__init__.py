import os

def launch(path, task, filename, host_string, nb_cpu, nb_gpu, HAVE_CUDA):

    whitening  = path + '/circus/whitening.py'
    basis      = path + '/circus/basis.py'
    clustering = path + '/circus/clustering.py'
    fitting    = path + '/circus/fitting.py'
    gathering  = path + '/circus/gathering.py'
    extracting = path + '/circus/extracting.py'
    merging    = path + '/circus/merging.py'
    filtering  = path + '/circus/filtering.py'

    if 'filtering' in task:
        os.system('mpirun %s -np %d python %s %s' %(host_string, 1, filtering, filename))
    if 'whitening' in task:
        os.system('mpirun %s -np %d python %s %s' %(host_string, nb_cpu, whitening, filename))
        os.system('mpirun %s -np %d python %s %s'     %(host_string, nb_cpu, basis, filename))
    if 'clustering' in task:
        os.system('mpirun %s -np %d python %s %s' %(host_string, nb_cpu, clustering, filename))
    if 'fitting' in task:
        if HAVE_CUDA:
            os.system('mpirun -x LD_LIBRARY_PATH %s -np %d python %s %s' %(host_string, nb_gpu, fitting, filename))
        else:
            os.system('mpirun %s -np %d python %s %s' %(host_string, nb_cpu, fitting, filename))
    if 'extracting' in task:
        os.system('mpirun %s -np %d python %s %s' %(host_string, nb_cpu, extracting, filename))
    if 'gathering' in task:
        os.system('python %s %s %s' %(nb_cpu, gathering, filename))
    if 'merging' in task:
        os.system('python %s %s' %(merging, filename))