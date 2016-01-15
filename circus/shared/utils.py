    # -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import os, sys, time
os.environ['MDP_DISABLE_SKLEARN']='yes'
import numpy, pylab, os, mpi4py, mdp, progressbar, tempfile
import scipy.linalg, scipy.optimize, cPickle, socket, tempfile, shutil, scipy.ndimage.filters
from circus.shared import files as io
from mpi import *

def get_progressbar(size):

    return progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()], maxval=size).start()


def get_whitening_matrix(X, fudge=1E-18):
   from numpy.linalg import eigh
   Xcov = numpy.dot(X.T, X)/X.shape[0]
   d,V  = eigh(Xcov)
   D    = numpy.diag(1./numpy.sqrt(d+fudge))
   W    = numpy.dot(numpy.dot(V,D), V.T)
   return W
