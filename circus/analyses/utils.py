import h5py
import numpy as np
import os
import re
import scipy as sp
import scipy.ndimage

from circus.shared.files import load_data
from circus.shared.probes import get_nodes_and_edges


def load_snippets(time_step_ids, params):

    nb_time_steps = params.getint('detection', 'N_t')
    do_spatial_whitening = params.getboolean('whitening', 'spatial')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    spatial_whitening = load_data(params, 'spatial_whitening') if do_spatial_whitening else None
    temporal_whitening = load_data(params, 'temporal_whitening') if do_temporal_whitening else None

    data_file = params.get_data_file()
    chunk_size = nb_time_steps
    nodes, edges = get_nodes_and_edges(params)

    data_file.open()

    _ = data_file.analyze(chunk_size)  # i.e. count chunks in sources

    snippets = []
    for time_step_id in time_step_ids:
        t_start = time_step_id - int(nb_time_steps - 1) // 2
        idx = data_file.get_idx(t_start, chunk_size)
        padding = (0, nb_time_steps - 1)
        data, t_offset = data_file.get_data(idx, chunk_size, padding=padding, nodes=nodes)
        data = data[(t_start - t_offset) % chunk_size:(t_start - t_offset) % chunk_size + nb_time_steps, :]
        if do_spatial_whitening:
            data = np.dot(data, spatial_whitening)
        if do_temporal_whitening:
            data = sp.ndimage.filters.convolve1d(data, temporal_whitening, axis=0, mode='constant')
        snippets.append(data)
    snippets = np.array(snippets)

    data_file.close()

    return snippets


def plot_snippets(ax, snippets, params, color='black', vmin=None, vmax=None):

    nb_channels = params.getint('data', 'N_e')
    nb_time_steps = params.getint('detection', 'N_t')

    probe = params.probe
    nodes, edges = get_nodes_and_edges(params)

    positions = []
    for i in probe['channel_groups'][1]['geometry'].keys():
        positions.append(probe['channel_groups'][1]['geometry'][i])
    positions = np.array(positions)
    vmin = np.abs(np.min(snippets)) if vmin is None else vmin
    vmax = np.abs(np.max(snippets)) if vmax is None else vmax
    dx = np.median(np.diff(np.unique(positions[:, 0])))  # horizontal inter-electrode distance
    dy = np.median(np.diff(np.unique(positions[:, 1])))  # vertical inter-electrode distance
    x_scaling = 0.8 * dx / 1.0
    y_scaling = 0.8 * dy / np.abs(vmax - vmin)

    ax.set_aspect('equal')
    for channel_id in range(0, nb_channels):
        x_c, y_c = positions[nodes[channel_id]]
        x = x_scaling * np.linspace(-0.5, + 0.5, num=nb_time_steps) + x_c
        for snippet in snippets:
            y = y_scaling * snippet[:, channel_id] + y_c
            ax.plot(x, y, color=color)

    return


def plot_snippet(ax, snippet, params, color='black', vmin=None, vmax=None, label=None):

    nb_channels = params.getint('data', 'N_e')
    nb_time_steps = params.getint('detection', 'N_t')

    probe = params.probe
    nodes, edges = get_nodes_and_edges(params)

    positions = []
    for i in probe['channel_groups'][1]['geometry'].keys():
        positions.append(probe['channel_groups'][1]['geometry'][i])
    positions = np.array(positions)
    vmin = np.abs(np.min(snippet)) if vmin is None else vmin
    vmax = np.abs(np.max(snippet)) if vmax is None else vmax
    dx = np.median(np.diff(np.unique(positions[:, 0])))  # horizontal inter-electrode distance
    dy = np.median(np.diff(np.unique(positions[:, 1])))  # vertical inter-electrode distance
    x_scaling = 0.8 * dx / 1.0
    y_scaling = 0.8 * dy / np.abs(vmax - vmin)

    ax.set_aspect('equal')
    for channel_id in range(0, nb_channels):
        x_c, y_c = positions[nodes[channel_id]]
        x = x_scaling * np.linspace(-0.5, + 0.5, num=nb_time_steps) + x_c
        y = y_scaling * snippet[:, channel_id] + y_c
        plot_kwargs = {
            'color': color,
            'label': label if channel_id == 0 else None,
        }
        ax.plot(x, y, **plot_kwargs)

    return


def load_template(template_id, params, extension=''):

    nb_channels = params.getint('data', 'N_e')
    nb_time_steps = params.getint('detection', 'N_t')

    templates = load_data(params, 'templates', extension=extension)
    template = templates[:, template_id]
    template = template.toarray()
    template = template.reshape(nb_channels, nb_time_steps)
    template = template.transpose()

    return template


def plot_template(ax, template, params, color='black', vmin=None, vmax=None, label=None):

    nb_channels = params.getint('data', 'N_e')
    nb_time_steps = params.getint('detection', 'N_t')

    probe = params.probe
    nodes, edges = get_nodes_and_edges(params)

    positions = []
    for i in probe['channel_groups'][1]['geometry'].keys():
        positions.append(probe['channel_groups'][1]['geometry'][i])
    positions = np.array(positions)
    vmin = np.abs(np.min(template)) if vmin is None else vmin
    vmax = np.abs(np.max(template)) if vmax is None else vmax
    dx = np.median(np.diff(np.unique(positions[:, 0])))  # horizontal inter-electrode distance
    dy = np.median(np.diff(np.unique(positions[:, 1])))  # vertical inter-electrode distance
    x_scaling = 0.8 * dx / 1.0
    y_scaling = 0.8 * dy / np.abs(vmax - vmin)

    ax.set_aspect('equal')
    for channel_id in range(0, nb_channels):
        if np.any(template[:, channel_id] != 0.0):
            x_c, y_c = positions[nodes[channel_id]]
            x = x_scaling * np.linspace(-0.5, + 0.5, num=nb_time_steps) + x_c
            y = y_scaling * template[:, channel_id] + y_c
            plot_kwargs = {
                'color': color,
                'label': label,
            }
            ax.plot(x, y, **plot_kwargs)
            label = None  # i.e. label first plot only

    return


def load_clusters_data(params, extension=''):

    file_out_suff = params.get('data', 'file_out_suff')
    path = "{}.clusters{}.hdf5".format(file_out_suff, extension)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with h5py.File(path, mode='r', libver='earliest') as file:
        data = dict()
        p = re.compile('_\d*$')  # noqa
        for key in file.keys():
            m = p.search(key)
            if m is None:
                data[key] = file[key][:]
            else:
                k_start, k_stop = m.span()
                key_ = key[0:k_start]
                channel_nb = int(key[k_start + 1:k_stop])
                if key_ not in data:
                    data[key_] = dict()
                data[key_][channel_nb] = file[key][:]

    return data
