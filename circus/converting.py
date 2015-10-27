"""Conversion script from template matching files to Kwik.

There are several improvements:

* Don't load all features and masks in memory (need for KwikCreator to accept
  features and masks as generators, not lists).

"""

from .shared.utils import *
import os
import os.path as op
import shutil

import numpy as np
from circus.shared.files import detect_header

from phy.detect.spikedetekt import SpikeDetekt
from phy.electrode import load_probe
from phy.io.h5 import open_h5
from phy.io.kwik import create_kwik, KwikCreator, KwikModel
from phy.utils.event import ProgressReporter
from phy.traces.waveform import WaveformLoader, SpikeLoader
from phy.traces.filter import bandpass_filter, apply_filter
from phy.utils.logging import info
from phy.utils.array import _spikes_per_cluster

extract_features = True #False is working now with max's branch
filtered_datfile = True

def main(filename, params, nb_cpu, nb_gpu, use_gpu):

    def _read_spikes(basename):
      with open_h5(basename + '.spiketimes.mat', 'r') as f:
          spike_samples = {}
          for name in f.children():
              cluster = int(name.split('_')[1])
              samples = f.read(name)[:].ravel().astype(np.uint64)
              spike_samples[cluster] = samples
          clusters = np.sort(list(spike_samples.keys()))
          # n_clusters = len(clusters)
          counts = {cluster: len(spikes)
                    for cluster, spikes in spike_samples.items()}
          spikes = np.hstack([spike_samples[cluster]
                              for cluster in clusters])
          idx = np.argsort(spikes)
          spike_clusters = np.repeat(clusters, [counts[cluster]
                                                for cluster in clusters])
          return spikes[idx], spike_clusters[idx]


    def _read_templates(basename, probe, n_total_channels, n_channels):
        with open_h5(basename + '.templates.mat', 'r') as f:
            templates = f.read('/templates')
            n_templates, n_samples, n_channels = templates.shape
            n_templates    //= 2
            templates        = templates[:n_templates, :, :]
            masks            = np.zeros((n_templates, n_channels))
            electrodes       = np.argmax(np.abs(templates).max(1), 1)

            inv_nodes        = np.zeros(n_total_channels, dtype=np.int32)
            nodes            = []
            for key in probe['channel_groups'].keys():
              nodes += probe['channel_groups'][key]['channels']
            nodes            = np.array(nodes, dtype=np.int32)
            idx              = np.argsort(nodes)
            nodes            = np.sort(nodes)
            inv_nodes[nodes] = np.argsort(nodes)

            def get_edges(i, channel_groups):
              edges = []
              pos_x, pos_y = channel_groups['geometry'][i]
              for c2 in channel_groups['channels']:
                  pos_x2, pos_y2 = channel_groups['geometry'][c2]
                  if (((pos_x - pos_x2)**2 + (pos_y - pos_y2)**2) <= probe['radius']**2):
                      edges += [c2]
              return edges

            for count, i in enumerate(electrodes):
                for key in probe['channel_groups'].keys():
                  if nodes[i] in probe['channel_groups'][key]['channels']:
                    masks[count, idx[inv_nodes[get_edges(nodes[i], probe['channel_groups'][key])]]] = 1
        return templates, masks

    def _read_amplitudes(basename, n_templates, n_spikes, spike_clusters):
        amplitudes = np.empty_like(spike_clusters, dtype=np.float32)
        spike_ids = np.arange(n_spikes, dtype=np.int32)
        spc = _spikes_per_cluster(spike_ids, spike_clusters)

        with open_h5(basename + '.amplitudes.mat', 'r') as f:
            for i in range(n_templates):
                amplitudes_i = f.read('/temp_' + str(i))[0,...]
                amplitudes[spc[i]] = amplitudes_i
        return amplitudes


    def _truncate(fn, extension='.dat', offset=None, n_channels=None, itemsize=None, dtype=None, chunk_size=50000):
        """Eventually truncate a file at the end to ensure it has a correct shape.
        """
        data = np.memmap(fn, dtype=dtype, offset=offset)
        N    = data.shape[0]

        if np.mod(N, n_channels) != 0:

            fn_copy   = fn + extension
            N         = int(N/n_channels)
            chunk_len = chunk_size
            n_samples = int(N/chunk_len)
            offset    = 0

            if op.exists(fn_copy):
                return fn_copy, np.memmap(fn_copy, dtype=dtype, offset=offset, shape=(N, n_channels))

            # Create the end-truncated file.
            info("Truncating...")
            f = open(fn_copy, 'w')
            for i in range(n_samples):
                f.write(data[i*chunk_len*n_channels:(i+1)*chunk_len*n_channels])
            f.close()
        else:
            fn_copy = fn
            N       = int(N/n_channels)

        data    = np.memmap(fn_copy, dtype=dtype, offset=offset, shape=(N, n_channels))
        return fn_copy, data


    def _read_filtered(filename, offset_value, n_channels=None, dtype=None):
        fn     = filename
        offset = int(detect_header(filename, offset_value))
        info("Header: {} bytes.".format(offset))
        dtype = np.dtype(dtype)
        filename, data = _truncate(fn,
                          offset=offset,
                          n_channels=n_channels,
                          itemsize=dtype.itemsize,
                          dtype=dtype)
        return filename, data


    class Converter(object):
        def __init__(self,
                     basename,
                     filename,
                     N_t,
                     n_channels=None,
                     n_total_channels=None,
                     prb_file=None,
                     dtype=None,
                     sample_rate=None,
                     dc_offset=0,
                     gain=0.01,
                     offset_value=0
                     ):

            self.n_features_per_channel = 3
            self.n_total_channels = n_total_channels
            self.extract_s_after = self.extract_s_before = extract_s_before = extract_s_after = int(N_t - 1)//2

            # Filtering parameters for PCA (these are ignored if filtered_datfile == True)
            filter_low = 500.
            filter_high = 0.95 * .5 * sample_rate
            filter_butter_order = 3

            self.basename = basename
            self.kwik_path = basename + '.kwik'
            self.dtype = dtype
            self.prb_file = prb_file
            self.probe = load_probe(prb_file)

            self.sample_rate = sample_rate
            self.filtered_datfile = filtered_datfile

            self._sd = SpikeDetekt(probe=self.probe,
                                   n_features_per_channel=
                                   self.n_features_per_channel,
                                   pca_n_waveforms_max=10000,
                                   extract_s_before=extract_s_before,
                                   extract_s_after=extract_s_after,
                                   sample_rate=sample_rate,
                                   )
            self.n_samples_w = extract_s_before + extract_s_after + 1

            # A xxx.filtered.trunc file may be created if needed.
            self.file, self.traces_f = _read_filtered(filename,
                                           offset_value=offset_value,
                                           n_channels=n_total_channels,
                                           dtype=dtype,
                                           )
            self.n_samples, self.n_total_channels = self.traces_f.shape
            self.n_channels = n_channels
            assert n_total_channels == self.n_total_channels
            info("Loaded traces: {}.".format(self.traces_f.shape))

            # Load spikes.
            self.spike_samples, self.spike_clusters = _read_spikes(basename)
            self.n_spikes = len(self.spike_samples)
            assert len(self.spike_clusters) == self.n_spikes
            info("Loaded {} spikes.".format(self.n_spikes))

            # Chunks when computing features.
            self.chunk_size = 2500
            self.n_chunks   = int(np.ceil(self.n_spikes / self.chunk_size))

            # Load templates and masks.
            self.templates, self.template_masks = _read_templates(basename, self.probe, self.n_total_channels, self.n_channels)
            self.n_templates = len(self.templates)
            info("Loaded templates: {}.".format(self.templates.shape))

            # Load amplitudes.
            self.amplitudes = _read_amplitudes(basename, self.n_templates, self.n_spikes, self.spike_clusters)

            if extract_features:
                # The WaveformLoader fetches and filters waveforms from the raw traces dynamically.
                n_samples = (extract_s_before, extract_s_after)
                b_filter = bandpass_filter(rate=self.sample_rate,
                                           low=filter_low,
                                           high=filter_high,
                                           order=filter_butter_order)

                def filter(x):
                  return apply_filter(x, b_filter)

                filter_margin = filter_butter_order * 3

                nodes            = []
                for key in self.probe['channel_groups'].keys():
                  nodes += self.probe['channel_groups'][key]['channels']
                nodes    = np.array(nodes, dtype=np.int32)


                if filtered_datfile:
                  self._wl = WaveformLoader(traces=self.traces_f,
                                            n_samples=self.n_samples_w,
                                            dc_offset=dc_offset,
                                            scale_factor=gain,
                                            channels=nodes
                                            )
                else:
                  self._wl = WaveformLoader(traces=self.traces_f,
                                            n_samples=self.n_samples_w,
                                            filter=filter,
                                            filter_margin=filter_margin,
                                            dc_offset=dc_offset,
                                            scale_factor=gain,
                                            channels=nodes
                                            )

                # A virtual (n_spikes, n_samples, n_channels) array that is
                # memmapped to the filtered data file.
                self.waveforms = SpikeLoader(self._wl, self.spike_samples)

                assert self.waveforms.shape == (self.n_spikes,
                                                self.n_samples_w,
                                                self.n_channels)
                assert self.template_masks.shape == (self.n_templates, self.n_channels)

        def iter_spikes(self):
            for idx in range(0, self.n_chunks):
                i = idx * self.chunk_size
                j = (idx + 1) * self.chunk_size
                j_clip = min(j, self.n_spikes)
                yield (i, j_clip)

        def compute_pcs(self):
            k = self.n_spikes // self._sd._kwargs['pca_n_waveforms_max']

            # Find the masks of the selection of spikes.
            clu = self.spike_clusters[::k]
            masks = self.template_masks[clu]

            w, m = self.waveforms[::k], masks
            self.pcs = self._sd.waveform_pcs(w, m)
            return self.pcs

        def compute_features(self):
            pr = ProgressReporter()
            pr.set_progress_message('Computing features: {progress:.1f}%.')
            pr.set_complete_message('All features computed.')
            pr.value_max = self.n_chunks

            for i, j in self.iter_spikes():
                n = j - i

                # info("Extracting waveforms {} to {}...".format(i, j))
                w = self.waveforms[i:j]
                assert w.shape == (n, self.n_samples_w, self.n_channels)

                # info("Computing features of spikes {} to {}...".format(i, j))
                f = self._sd.features(w, self.pcs)
                assert f.shape == (n, self.n_channels, self.n_features_per_channel)

                yield f

                pr.increment()

        def compute_masks(self):
            for i, j in self.iter_spikes():
                n = j - i

                clu = self.spike_clusters[i:j]
                m = self.template_masks[clu]
                assert m.shape == (n, self.n_channels)

                yield m

        def create_kwik(self):
            # Create an empty Kwik file.
            info("Starting the conversion to Kwik...")
            create_kwik(kwik_path=self.kwik_path,
                        raw_data_files=[self.file],
                        prb_file=self.prb_file,
                        n_channels=self.n_total_channels,
                        sample_rate=self.sample_rate,
                        dtype=self.dtype,
                        nfeatures_per_channel=self.n_features_per_channel,
                        extract_s_after = self.extract_s_after,
                        extract_s_before = self.extract_s_before,
                        overwrite=True,
                        )

            # Compute PCs and features.
            if extract_features:
                info("Computing PCs...")
                self.compute_pcs()

                info("Computing features of all spikes...")
                # WARNING: watch out RAM usage here. We cannot use a generator because
                # the KwiKCreator only accepts lists at the moment.
                features = (f for f in self.compute_features())
                masks    = (m for m in self.compute_masks())
            else:
                info("Skipping PCA...")
                features = None
                masks = None
                self.n_features_per_channel = None

            # Add clusters.
            creator = KwikCreator(self.kwik_path)

            info("Adding the clusters in the kwik file.")
            creator.add_clustering(group=1,
                                   name='main',
                                   spike_clusters=self.spike_clusters,
                                   #template_waveforms=self.templates,
                                   #template_masks=self.template_masks,
                                   #template_amplitudes=self.amplitudes,
                                   )

            # Add spikes.
            info("Adding the spikes in the kwik file.")
            creator.add_spikes(group=1,
                               spike_samples=self.spike_samples,
                               masks=masks,
                               features=features,
                               n_channels = self.n_channels,
                               n_features = self.n_features_per_channel
                               )

            # Add template amplitudes. We add these to the .kwik file, not the
            # .kwx, since they're lightweight enough that you can delete them
            # afterwards!


            info("Kwik file successfully created!")

        def template_explorer(self, name='templates'):
            """Mini GUI to explore the templates."""

            from phy.plot.waveforms import plot_waveforms
            from vispy.app import run

            p         = {}
            positions = []
            nodes     = []
            for key in c.probe['channel_groups'].keys():
              p.update(c.probe['channel_groups'][key]['geometry'])
              nodes     +=  c.probe['channel_groups'][key]['channels']
              positions += [p[channel] for channel in c.probe['channel_groups'][key]['channels']]
            idx       = np.argsort(nodes)
            positions = np.array(positions)[idx]

            self._n = 2
            wave    = np.zeros((0, self.n_samples_w, self.n_channels))
            w = plot_waveforms(channel_positions=positions,
                               waveforms=wave,
                               overlap=True,
                               alpha=1.,
                               probe_scale=(1.9, 1.0),
                               box_scale=(0.066, 0.01),
                               )

            # Show templates.
            if name == 'templates':
                templates = self.templates
                masks = self.template_masks

            # Show waveforms.
            elif name == 'waveforms':
                templates = self.waveforms
                masks     = self.template_masks[self.spike_clusters]

            @w.connect
            def on_key_press(e):
                if e.key == 'space':
                    self._n += 1 if ('Shift' not in e.modifiers) else -1
                    if name == 'templates':
                        info("Template {}.".format(self._n))
                        w.set_data(waveforms=templates[self._n],
                                   masks=masks[self._n],
                                   )
                    elif name == 'waveforms':
                        sample = self.spike_samples[self._n]
                        cluster = self.spike_clusters[self._n]
                        info("Waveform {}, template={}, sample={}.".format(self._n,
                             cluster, sample))

                        wav = np.vstack((templates[self._n],
                                         self.templates[cluster][:-1][None, ...]))

                        m = np.vstack((masks[self._n],
                                       self.template_masks[cluster][None, ...]))
                        w.set_data(waveforms=wav,
                                   masks=m,
                                   spike_clusters=[0, 1],
                                   )
            run()

    basename         = params.get('data', 'file_out_suff')
    basename, ext    = os.path.splitext(basename)
    prb_file         = params.get('data', 'mapping')
    n_channels       = params.getint('data', 'N_e')
    N_t              = params.getint('data', 'N_t')
    n_total_channels = params.getint('data', 'N_total')
    sample_rate      = params.getint('data', 'sampling_rate')
    dtype            = params.get('data', 'data_dtype')
    dc_offset        = params.getint('data', 'dtype_offset')
    gain             = params.getfloat('data', 'gain')
    data_offset      = params.get('data', 'data_offset')

    c = Converter(basename, filename, N_t,
                  n_channels=n_channels,
                  n_total_channels=n_total_channels,
                  dc_offset=dc_offset,
                  prb_file=prb_file,
                  sample_rate=sample_rate,
                  dtype=dtype,
                  gain=gain,
                  offset_value=data_offset
                  )

    # Uncomment to have a look at the templates or waveforms.
    #c.template_explorer('templates')  # 'waveforms' or 'templates'
    #exit()

    if not os.path.exists(basename + '.kwik'):
        # Conversion.
        c.create_kwik()

    # Try to open the kwik file after the conversion.
    model = KwikModel(c.kwik_path)
    model.describe()