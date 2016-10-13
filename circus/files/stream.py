import h5py, numpy, re, sys, os, logging
from circus.shared.messages import print_and_log
from circus.shared.mpi import comm

logger = logging.getLogger(__name__)

class Stream(object):

    def __init__(self, data_sources, **kwargs):

        self.sources = data_sources
        self._times  = [0]
        for source in self.sources:
            self._times += [source.duration]
        
    def get_data(self, idx, chunk_size, padding=(0, 0), nodes=None):
        
        if not hasattr(self, '_chunks_in_sources'):
            print_and_log(['The Stream must be initialized with the analyze() function'], 'error', logger)

        cidx = numpy.searchsorted(idx, self._chunks_in_sources)
        return self.sources[cidx].get_data(idx - self._chunks_in_sources[cidx], chunk_size, padding, nodes)

        
    def get_snippet(self, time, length, nodes=None):
        cidx  = numpy.searchsorted(time, numpy.cumsum(self._times))
        time -= numpy.cumsum(self._times)[cidx]
        return self.sources[cidx].get_data(0, chunk_size=length, padding=(time, time), nodes=nodes)


    def set_data(self, time, data):
        cidx = numpy.searchsorted(time, numpy.cumsum(self._times))
        return self.sources[cidx].set_data(time - numpy.cumsum(self._times)[cidx], data)


    def analyze(self, chunk_size):
        nb_chunks = 0
        self._chunks_in_sources = [0]
        for source in self.sources:
            a, b = source.analyze(chunk_size)
            nb_chunks += a
            if b > 0:
                nb_chunks += 1
            self._chunks_in_sources += [nb_chunks]
        return nb_chunks


    @property
    def duration(self):
        return numpy.sum(self._times)


    @property
    def shape(self):
        return (self.duration, self.nb_channels)

         
    @property
    def nb_channels(self):
        return self.sources[0].nb_channels