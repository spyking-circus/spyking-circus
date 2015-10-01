# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import os, sys, time
os.environ['MDP_DISABLE_SKLEARN']='yes'
import numpy, pylab, os, mpi4py, hdf5storage, mdp, progressbar, tempfile
import scipy.linalg, scipy.optimize, cPickle, socket, ConfigParser, tempfile, shutil
from mpi4py import MPI
comm = MPI.COMM_WORLD
import algorithms as algo
import files as io

def gather_array(data, mpi_comm, root=0, shape=0, dtype='float32'):
    # gather 1D or 2D numpy arrays
    assert isinstance(data, numpy.ndarray)
    assert len(data.shape) < 3
    # first we pass the data size
    size  = data.size
    sizes = mpi_comm.gather(size, root=root) or []
    # now we pass the data
    displacements = [int(sum(sizes[:i])) for i in range(len(sizes))]
    if dtype is 'float32':
        gdata         = numpy.empty(int(sum(sizes)), dtype=numpy.float32)
        mpi_comm.Gatherv([data.flatten(), size, MPI.FLOAT], [gdata, (sizes, displacements), MPI.FLOAT], root=root)
    elif dtype is 'int32':
        gdata         = numpy.empty(int(sum(sizes)), dtype=numpy.int32)
        mpi_comm.Gatherv([data.flatten(), size, MPI.INT], [gdata, (sizes, displacements), MPI.INT], root=root)        
    if len(data.shape) == 1:
        return gdata
    else:
        if shape == 0:
            num_lines = data.shape[0]
            if num_lines > 0:
                return gdata.reshape((num_lines, gdata.size/num_lines))
            else: 
                return gdata.reshape((0, gdata.shape[1]))
        if shape == 1:
            num_columns = data.shape[1]
            if num_columns > 0:
                return gdata.reshape((gdata.size/num_columns, num_columns))
            else:
                return gdata.reshape((gdata.shape[0], 0))

def all_gather_array(data, mpi_comm, shape=0, dtype='float32'):
    # gather 1D or 2D numpy arrays
    assert isinstance(data, numpy.ndarray)
    assert len(data.shape) < 3
    # first we pass the data size
    size  = data.size
    sizes = mpi_comm.allgather(size) or []
    # now we pass the data
    displacements = [int(sum(sizes[:i])) for i in range(len(sizes))]
    if dtype is 'float32':
        gdata         = numpy.empty(int(sum(sizes)), dtype=numpy.float32)
        mpi_comm.Allgatherv([data.flatten(), size, MPI.FLOAT], [gdata, (sizes, displacements), MPI.FLOAT])
    elif dtype is 'int32':
        gdata         = numpy.empty(int(sum(sizes)), dtype=numpy.int32)
        mpi_comm.Allgatherv([data.flatten(), size, MPI.INT], [gdata, (sizes, displacements), MPI.INT])        
    if len(data.shape) == 1:
        return gdata
    else:
        if shape == 0:
            num_lines = data.shape[0]
            if num_lines > 0:
                return gdata.reshape((num_lines, gdata.size/num_lines))
            else: 
                return gdata.reshape((0, gdata.shape[1]))
        if shape == 1:
            num_columns = data.shape[1]
            if num_columns > 0:
                return gdata.reshape((gdata.size/num_columns, num_columns))
            else:
                return gdata.reshape((gdata.shape[0], 0))

def get_mpi_type(data_type):
    if data_type == 'int16':
        return MPI.SHORT
    elif data_type == 'uint16':
        return MPI.UNSIGNED_SHORT
    elif data_type == 'float32':
        return MPI.FLOAT

def get_whitening_matrix(X, fudge=1E-18):
   from numpy.linalg import eigh
   Xcov = numpy.dot(X.T, X)/X.shape[0]
   d,V  = eigh(Xcov)
   D    = numpy.diag(1./numpy.sqrt(d+fudge))
   W    = numpy.dot(numpy.dot(V,D), V.T)
   return W

def frechet(x, m, s, alpha):
    x0 = (x - m)/s
    with numpy.errstate(over='ignore', divide='ignore', invalid='ignore'):
        return (alpha/s)*(x0**(-(1+alpha)))*numpy.exp(-x0**(-alpha))

def normal(x, m, s):
    return (1/(numpy.sqrt(2*numpy.pi)*s))*numpy.exp(-(x - m)**2/(2*s**2))


def fit_noise(xdata, ydata, min_mean=0.5, init=None, display=False):

    def g(x, a, b, c, d, e, f, g): 
        with numpy.errstate(over='ignore', divide='ignore'):
            return abs(a)*normal(x, min_mean + abs(b), abs(c)) + abs(d)*frechet(x, abs(e), abs(f), abs(g))

    try:
        if init is None:
            amp_1    = ydata[numpy.searchsorted(xdata, 1)]
            amp_2    = ydata[numpy.argsort(xdata)[0]]
            init     = [amp_1, 1-min_mean, 0.1, amp_2, 0, 0.1, 2]
        else:
            init[1] -= min_mean

        result, pcov = scipy.optimize.curve_fit(g, xdata, ydata, init)
        data_fit     = g(xdata, result[0], result[1], result[2], result[3], result[4], result[5], result[6])
        result       = numpy.abs(result)
        result[1]   += min_mean       
    except Exception:
        return None
    if display:
        idx      = numpy.argsort(xdata)
        xdata    = xdata[idx]
        ydata    = ydata[idx]    
        pylab.plot(xdata, ydata, '.')
        pylab.plot(xdata, data_fit)
        pylab.show()
    return result


def fake_data_from_templates(templates, recording=1, file_name='fake', rate=10, n_cells=None, noise=0.2, sampling=20000):

    file_name = 'data/synthetic/' + file_name

    if not os.path.exists(file_name):
        os.makedirs(file_name)
    N_e, N_t, N_tm = templates.shape
    
    print "Generating fake data with", n_cells, "cells at", rate, "Hz during", recording, "min"

    file_out       = file_name + '/' + file_name
    chunksize      = 60 
    dt             = 1./sampling
    uint16_offset  = 2**15 - 1          # Padding for uint16
    file           = open(file_name + '.raw', 'w')
    if n_cells is None:
        n_cells    = N_tm/2
        cells      = numpy.arange(n_cells)
    else:
        cells      = numpy.random.permutation(numpy.arange(N_tm/2))[:n_cells]
    ngbins         = int(recording*60/chunksize)
    nbins          = int(chunksize/dt)
    result         = {'spiketimes' : {}, 'amplitudes' : {}}
    template_shift = int((N_t-1)/2)

    for i in xrange(N_tm/2):
        result['spiketimes']['temp_' + str(i)] = []
        result['amplitudes']['temp_' + str(i)] = []

    pbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()], maxval=ngbins).start()

    for k in xrange(ngbins):
        curve          = numpy.zeros((N_e, nbins), dtype=numpy.float32)
        count          = 0
        spikes         = numpy.random.rand(nbins) < rate*dt*n_cells
        mask           = numpy.where(spikes == True)[0]
        offset         = k*recording*sampling
        t_start        = N_t
        t_stop         = nbins - N_t

        for spike in mask:
            if (spike > t_start) and (spike < t_stop):
                n_template = cells[numpy.random.random_integers(0, n_cells-1)]
                amplitude  = 0.8 + 0.4*numpy.random.rand()
                curve[:, spike-template_shift:spike+template_shift+1] += amplitude*templates[:, :, n_template]
                result['spiketimes']['temp_' + str(n_template)]       += [spike + offset]
                result['amplitudes']['temp_' + str(n_template)]       += [(amplitude, 0)]
                count += 1

        #print count, 'spikes inserted...'
        curve += noise*numpy.random.randn(curve.shape[0], curve.shape[1])
        curve  = curve.T.flatten()
        curve /= 0.01
        curve += uint16_offset
        curve  = curve.astype(numpy.uint16)
        file.write(curve)
    
        pbar.update(k)

    pbar.finish()
    file.close()

    for i in xrange(N_tm/2):
        result['spiketimes']['temp_' + str(i)] = numpy.array(result['spiketimes']['temp_' + str(i)], dtype=numpy.int32)
        result['amplitudes']['temp_' + str(i)] = numpy.array(result['amplitudes']['temp_' + str(i)], dtype=numpy.float32)

    if os.path.exists(file_out + '.overlap.mat'):
        os.remove(file_out + '.overlap.mat')
    hdf5storage.savemat(file_out + '.spiketimes-real', result['spiketimes'])
    hdf5storage.savemat(file_out + '.amplitudes-real', result['amplitudes'])
    hdf5storage.savemat(file_out + '.templates', {'templates' : templates})