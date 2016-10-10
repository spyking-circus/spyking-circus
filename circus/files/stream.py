import h5py, numpy, re, sys, os, logging
from circus.shared.messages import print_and_log
from circus.shared.mpi import comm

logger = logging.getLogger(__name__)

class Stream(object):

    def __init__(self, data_sources, **kwargs):

        self.sources = data_sources


    def get_data(self, idx, chunk_size, padding=(0, 0), nodes=None):
        
        if not hasattr(self, '_chunks_in_sources'):
            print_and_log(['The Stream must be initialized with the analyze() function'], 'error', logger)

        cidx = numpy.searchsorted(idx, self._chunks_in_sources)

        return self.sources[cidx].get_data(idx - 0, chunk_size, padding, nodes)

        
    def get_snippet(self, time, length, nodes=None):
        return self.get_data(0, chunk_size=length, padding=(time, time), nodes=nodes)


    def set_data(self, time, data):
        
        if not hasattr(self, '_chunks_in_sources'):
            print_and_log(['The Stream must be initialized with the analyze() function'], 'error', logger)

        cidx = numpy.searchsorted(idx, self._chunks_in_sources)
        
        return self.sources[idx].set_data()

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
        for source in self.sources:
            duration += b.duration
        return duration

    @property
    def shape(self):
        return (self.duration, self.nb_channels)
         
    @property
    def nb_channels(self):
        return self.sources[0].nb_channels