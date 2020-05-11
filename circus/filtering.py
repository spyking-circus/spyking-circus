"""
filtering.py

author: Pierre Yeger
e-mail: pierre.yger <at> inserm.fr

Executes filtering and trigger sections on the data. 
"""
from scipy import signal
from circus.shared import plot
from circus.shared.utils import *
from circus.shared.probes import get_nodes_and_edges
from circus.shared.messages import print_and_log, init_logging
from circus.shared.files import get_artefact
from circus.shared.mpi import detect_memory


def check_if_done(params, flag, logger):
    """
    Read filter_done in [noedits] section of the param file

    Parameters
    ----------
    params : CircusParser object 
        the parser objects with filtering options from param file.

    flag : str
        'filter_done', 'artefacts_done', 'median_done' or 'ground_done'
    
    logger : logging object
        log message id

    Return
    ------
    bool
        True if data has been filtered, false if data was not
        filtered, or started if filtering started.
    """

    value = params.get('noedits', flag).lower().strip()
    if value == 'false':
        return False
    elif value == 'true':
        return True
    elif value == 'started':
        common_sentence = 'Data are likely to be corrupted, please recopy raw data'
        particular_sentence = 'And set the flag %s in the [noedits] section to False' % flag
        if comm.rank == 0:
            if flag == 'filter_done':
                msg = ['Code was interrupted while filtering', common_sentence, particular_sentence]
            elif flag == 'artefacts_done':
                msg = ['Code was interrupted while removing artefacts', common_sentence, particular_sentence]
            elif flag == 'median_done':
                msg = ['Code was interrupted while removing median', common_sentence, particular_sentence]
            elif flag == 'ground_done':
                msg = ['Code was interrupted while removing ground', common_sentence, particular_sentence]
            else:
                msg = ['Code was interrupted']
            print_and_log(msg, 'error', logger)
            sys.exit(0)

    return


def main(params, nb_cpu, nb_gpu, use_gpu):

    logger = init_logging(params.logfile)
    logger = logging.getLogger('circus.filtering')
    #################################################################
    do_filter = params.getboolean('filtering', 'filter')
    filter_done = check_if_done(params, 'filter_done', logger)
    artefacts_done = check_if_done(params, 'artefacts_done', logger)
    median_done = check_if_done(params, 'median_done', logger)
    ground_done = check_if_done(params, 'ground_done', logger)
    clean_artefact = params.getboolean('triggers', 'clean_artefact')
    remove_median = params.getboolean('filtering', 'remove_median')
    common_ground = params.getint('filtering', 'common_ground')
    remove_ground = common_ground >= 0
    nodes, edges = get_nodes_and_edges(params)
    #################################################################

    def filter_file(data_file_in, data_file_out, do_filtering, do_remove_median, do_remove_ground):
        """
        Performs a high-pass and low-pass Butterworth filter on the data file.

        Parameters
        ----------
        
        data_file_in : 

        data_file_out : 

        do_filtering : bool

        do_remove_median : bool
 
        do_remove_median : bool
        """

        try:
            cut_off = params.getfloat('filtering', 'cut_off', check=False)
            cut_off = [cut_off, 0.95 * (params.rate / 2.0)]  # Nyquist
        except Exception:
            cut_off = params.get('filtering', 'cut_off', check=False)
            cut_off = cut_off.split(',')
            try:
                cut_off[0] = float(cut_off[0])
            except Exception:
                if comm.rank == 0:
                    print_and_log(['First value of cut off must be a valid number'], 'error', logger)
                sys.exit(0)

            cut_off[1] = cut_off[1].replace(' ', '')
            if cut_off[1] == 'auto':
                cut_off[1] = 0.95 * (params.rate / 2.0)
            else:
                try:
                    cut_off[1] = float(cut_off[1])
                except Exception:
                    if comm.rank == 0:
                        print_and_log(['Second value of cut off must either auto, or a valid a number'], 'error', logger)
                    sys.exit(0)

        chunk_size = detect_memory(params, filtering=True)
        butter_order = params.getint('filtering', 'butter_order')
        nb_chunks, _ = data_file_in.analyze(chunk_size)

        b, a = signal.butter(butter_order, np.array(cut_off)/(params.rate/2.), 'pass')
        all_chunks = numpy.arange(nb_chunks, dtype=numpy.int64)
        to_process = all_chunks[comm.rank::comm.size]
        loc_nb_chunks = len(to_process)
        N_total = params.nb_channels
        process_all_channels = numpy.all(nodes == numpy.arange(N_total))
        duration = int(0.1*params.rate)

        if comm.rank == 0:
            to_write = []
            if do_filtering:
                to_write += ["Filtering with a Butterworth filter (order %d) in [%g, %g] Hz" % (butter_order, cut_off[0], cut_off[1])]
            if do_remove_median:
                to_write += ["Median over all channels is subtracted to each channels"]
            if do_remove_ground:
                to_write += ["Channel %s is used as a reference channel" % common_ground]

            print_and_log(to_write, 'default', logger)

        to_explore = range(comm.rank, nb_chunks, comm.size)

        if comm.rank == 0:
            to_explore = get_tqdm_progressbar(params, to_explore)

        if data_file_in == data_file_out:
            data_file_in.open(mode='r+')
        else:
            data_file_in.open(mode='r')
            data_file_out.open(mode='r+')

        for count, gidx in enumerate(to_explore):

            is_first = data_file_in.is_first_chunk(gidx, nb_chunks)
            is_last = data_file_in.is_last_chunk(gidx, nb_chunks)

            if not (is_first and is_last):
                if is_first:
                    padding = (0, duration)
                elif is_last:
                    padding = (-duration, 0)
                else:
                    padding = (-duration, duration)
            else:
                padding = (0, 0)

            local_chunk, t_offset =  data_file_in.get_data(gidx, chunk_size, padding)

            if do_filtering:
                local_chunk = signal.filtfilt(b, a, local_chunk, axis=0)
                local_chunk -= numpy.median(local_chunk, 0)
                if not is_last:
                    local_chunk = local_chunk[numpy.abs(padding[0]):-numpy.abs(padding[1])]
                else:
                    local_chunk = local_chunk[numpy.abs(padding[0]):]
            else:
                local_chunk = local_chunk[numpy.abs(padding[0]):-numpy.abs(padding[1])]

            if do_remove_median:
                if not process_all_channels:
                    global_median = numpy.median(numpy.take(local_chunk, nodes, axis=1), 1)
                else:
                    global_median = numpy.median(local_chunk, 1)

                local_chunk -= global_median[:, numpy.newaxis]

            if common_ground > -1:
                ground = local_chunk[:, common_ground]
                local_chunk -= ground[:, numpy.newaxis]

            if data_file_in != data_file_out and data_file_in.is_first_chunk(gidx, nb_chunks):
                if data_file_in.is_stream:
                    g_offset = t_offset - numpy.sum(data_file_in._times[:data_file_in._get_streams_index_by_time(t_offset) + 1])
                else:
                    g_offset = t_offset - data_file_in.t_start
            else:
                g_offset = t_offset

            data_file_out.set_data(g_offset, local_chunk)

        sys.stderr.flush()
        comm.Barrier()

    def compute_artefacts(data_file):
        """
        Compute artefact locations based on the [triggers] section of the params file.

        Parameters
        ----------
        data_file :

        Return
        ------
        dict
            A dictionary with the location of the artefacts
        """

        trig_in_ms = params.getboolean('triggers', 'trig_in_ms')
        artefacts = numpy.loadtxt(params.get('triggers', 'trig_file'), comments=['#', '//'])
        windows = numpy.loadtxt(params.get('triggers', 'trig_windows'), comments=['#', '//'])
        make_plots = params.get('triggers', 'make_plots')
        plot_path = os.path.join(params.get('data', 'file_out_suff'), 'plots')

        if len(windows.shape) == 1:
            windows = windows.reshape(1, 2)

        if len(artefacts.shape) == 1:
            artefacts = artefacts.reshape(1, 2)

        if trig_in_ms:
            if comm.rank == 0:
                print_and_log(['Artefact times are read in ms'], 'debug', logger)
            artefacts[:, 1] *= numpy.int64(data_file.sampling_rate*1e-3)
            windows[:, 1] *= numpy.int64(data_file.sampling_rate*1e-3)
        else:
            if comm.rank == 0:
                print_and_log(['Artefact times are read in timesteps'], 'debug', logger)

        artefacts = artefacts.astype(numpy.int64)
        windows = windows.astype(numpy.int64)

        nb_stimuli = len(numpy.unique(artefacts[:, 0]))
        mytest = numpy.all(numpy.in1d(numpy.unique(artefacts[:, 0]), numpy.unique(windows[:, 0])))

        if not mytest:
            if comm.rank == 0:
                print_and_log(['Error in the trigger file: not all artefacts are defined'], 'error', logger)
            sys.exit(0)

        all_labels = artefacts[:, 0]
        all_times = artefacts[:, 1]
        mask = (all_times >= 0) & (all_times + numpy.max(windows[:, 1]) < data_file.t_stop)
        all_times = numpy.compress(mask, all_times)
        all_labels = numpy.compress(mask, all_labels)

        local_labels = numpy.unique(all_labels)[comm.rank::comm.size]

        if comm.rank == 0:
            to_write = ["Computing averaged artefacts from %d stimuli" % nb_stimuli]
            print_and_log(to_write, 'default', logger)
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            local_labels = get_tqdm_progressbar(params, local_labels)

        comm.Barrier()
        # First we need to get the average artefacts
        art_dict = {}
        for count, artefact in enumerate(local_labels):
            indices = numpy.where(all_labels == artefact)[0].astype(numpy.uint32)
            tmp = numpy.where(windows[:, 0] == artefact)[0]
            tau = numpy.int64(windows[tmp, 1])
            pspikes = all_times[indices]
            times = numpy.sort(numpy.random.permutation(pspikes)[:500])
            if len(numpy.where(numpy.diff(times) < tau)[0]) > 0:
                if comm.rank == 0:
                    print_and_log(['Stimulation times for artefact %d are too close!' % artefact], 'error', logger)
                sys.exit(0)

            art_dict[artefact] = get_artefact(params, times, tau, nodes)
            if make_plots not in ['None', '']:
                save = [plot_path, '%d.%s' % (artefact, make_plots)]
                plot.view_artefact(art_dict[artefact], save=save)

        sys.stderr.flush()
        return art_dict

    def remove_artefacts(data_file, art_dict):
        """
        Remove artefact times based on the [triggers] section of the params file.

        Parameters
        ----------

        data_file :

        art_dict : dict
            a dictionary with the artefact times.
        """

        trig_in_ms = params.getboolean('triggers', 'trig_in_ms')
        artefacts = numpy.loadtxt(params.get('triggers', 'trig_file'), comments=['#', '//'])
        windows = numpy.loadtxt(params.get('triggers', 'trig_windows'), comments=['#', '//'])
        make_plots = params.get('triggers', 'make_plots')
        plot_path = os.path.join(params.get('data', 'file_out_suff'), 'plots')

        if len(windows.shape) == 1:
            windows = windows.reshape(1, 2)

        if len(artefacts.shape) == 1:
            artefacts = artefacts.reshape(1, 2)

        if trig_in_ms:
            if comm.rank == 0:
                print_and_log(['Artefact times are read in ms'], 'debug', logger)
            artefacts[:, 1] *= numpy.int64(data_file.sampling_rate * 1e-3)
            windows[:, 1] *= numpy.int64(data_file.sampling_rate * 1e-3)
        else:
            if comm.rank == 0:
                print_and_log(['Artefact times are read in timesteps'], 'debug', logger)

        artefacts = artefacts.astype(numpy.int64)
        windows = windows.astype(numpy.int64)
        nb_stimuli = len(numpy.unique(artefacts[:, 0]))
        mytest = numpy.all(numpy.in1d(numpy.unique(artefacts[:, 0]), numpy.unique(windows[:, 0])))

        if not mytest:
            if comm.rank == 0:
                print_and_log(['Error in the trigger files: not all artefacts are defined'], 'error', logger)
            sys.exit(0)

        all_labels = artefacts[:, 0]
        all_times = artefacts[:, 1]
        local_labels = numpy.unique(all_labels)[comm.rank::comm.size]

        mask = numpy.in1d(all_labels, local_labels)
        all_times = numpy.compress(mask, all_times)
        all_labels = numpy.compress(mask, all_labels)

        mask = (all_times >= 0) & (all_times < data_file.t_stop)
        all_times = numpy.compress(mask, all_times)
        all_labels = numpy.compress(mask, all_labels)

        if comm.rank == 0:
            to_write = ["Removing artefacts from %d stimuli" % nb_stimuli]
            print_and_log(to_write, 'default', logger)
            all_times = get_tqdm_progressbar(params, all_times)

        comm.Barrier()

        for count, time in enumerate(all_times):

            label = all_labels[count]
            tmp = numpy.where(windows[:, 0] == label)[0][0]
            tau = numpy.int64(windows[tmp, 1])

            if (data_file.t_stop - time) < tau:
                tau = max_offset - time

            local_chunk = data_file.get_snippet(time, tau)
            for idx, i in enumerate(nodes):
                local_chunk[:, i] -= art_dict[label][idx, :tau]
            data_file.set_data(time, local_chunk)

        comm.Barrier()
        sys.stderr.flush()

    if comm.rank == 0:
        print_and_log(['Initializing the filtering step...'], 'debug', logger)

    if params.getboolean('data', 'overwrite'):
        if comm.rank == 0:
            print_and_log(['Reading the input file...'], 'debug', logger)

        data_file_in = params.get_data_file()
        data_file_out = data_file_in
    else:
        if comm.rank == 0:
            print_and_log(['Overwrite is set to False, so creating a new datafile...'], 'debug', logger)

        if comm.rank == 0:
            print_and_log(['Reading the input file...'], 'debug', logger)

        if os.path.exists(params.get('data', 'data_file_no_overwrite')):
            has_been_created = True
        else:
            has_been_created = False

        if not has_been_created and (filter_done or median_done or artefacts_done):
            if comm.rank == 0:
                print_and_log(['The filtering is done but file not present. See no_edits section'], 'error', logger)
            sys.exit(0)

        if not has_been_created:
            data_file_in = params.get_data_file(source=True, has_been_created=has_been_created)
        else:
            data_file_in = params.get_data_file(source=False, has_been_created=has_been_created)

        if comm.rank == 0:
            print_and_log(['Reading the output file and allocating ressources...'], 'debug', logger)

        description = data_file_in.get_description()
        description['data_dtype'] = 'float32'
        description['dtype_offset'] = 0
        description['data_offset'] = 0

        comm.Barrier()
        data_file_out = params.get_data_file(is_empty=not has_been_created, params=description)

        if comm.rank == 0:
            print_and_log(['Allocating space for filtered files...'], 'debug', logger)

        if not has_been_created:
            data_file_out.allocate(shape=data_file_in.shape)

        comm.Barrier()

    if clean_artefact:
        if not (os.path.exists(params.get('triggers', 'trig_file')) and os.path.exists(params.get('triggers', 'trig_windows'))):
            if comm.rank == 0:
                print_and_log(['trig_file or trig_windows file can not be found'], 'error', logger)
            sys.exit(0)

    to_write = []

    if do_filter and filter_done:
        do_filter = False
        to_write += ["Filtering has already been done"]
    if remove_median and median_done:
        remove_median = False
        to_write += ["Median over all channels has already been removed"]
    if remove_ground and ground_done:
        remove_ground = False
        to_write += ["Common ground %s has alread been subtracted" % common_ground]

    if comm.rank == 0 and len(to_write) > 0:
        print_and_log(to_write, 'info', logger)

    if params.getboolean('data', 'overwrite'):
        data_file_in.open(mode='r+')
    else:
        data_file_in.open(mode='r')
        data_file_out.open(mode='r+')

    if do_filter or remove_median or remove_ground:
        if comm.rank == 0:
            if do_filter:
                params.write('noedits', 'filter_done', 'Started')
            if remove_median:
                params.write('noedits', 'median_done', 'Started')
            if remove_ground:
                params.write('noedits', 'ground_done', 'Started')
        filter_file(data_file_in, data_file_out, do_filter, remove_median, remove_ground)

    if comm.rank == 0:
        if do_filter:
            params.write('noedits', 'filter_done', 'True')
        if remove_median:
            params.write('noedits', 'median_done', 'True')
        if remove_ground:
            params.write('noedits', 'ground_done', 'True')

    if clean_artefact and artefacts_done:
        clean_artefact = False
        if comm.rank == 0:
            print_and_log(['Artefacts have already been removed'], 'debug', logger)

    if clean_artefact:
        art_dict = compute_artefacts(data_file_in)
        if comm.rank == 0:
            params.write('noedits', 'artefacts_done', 'Started')
        remove_artefacts(data_file_out, art_dict)

    if comm.rank == 0:
        if clean_artefact:
            params.write('noedits', 'artefacts_done', 'True')

    data_file_in.close()
    if not params.getboolean('data', 'overwrite'):
        data_file_out.close()

    comm.Barrier()
