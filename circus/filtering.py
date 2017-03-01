from scipy import signal
from .shared import plot
from .shared.utils import *
from circus.shared.probes import get_nodes_and_edges
from circus.shared.messages import print_and_log, init_logging
from circus.shared.files import get_artefact

def main(params, nb_cpu, nb_gpu, use_gpu):


    logger         = init_logging(params.logfile)
    logger         = logging.getLogger('circus.filtering')
    #################################################################
    do_filter      = params.getboolean('filtering', 'filter')
    filter_done    = params.getboolean('noedits', 'filter_done')
    artefacts_done = params.getboolean('noedits', 'artefacts_done')
    median_done    = params.getboolean('noedits', 'median_done')
    clean_artefact = params.getboolean('triggers', 'clean_artefact')
    remove_median  = params.getboolean('filtering', 'remove_median')
    nodes, edges   = get_nodes_and_edges(params)
    #################################################################


    def filter_file(data_file_in, data_file_out, do_filtering, do_remove_median):

        try:
            cut_off    = params.getfloat('filtering', 'cut_off')
            cut_off    = [cut_off, 0.95*(params.rate/2.)]
        except Exception:
            cut_off        = params.get('filtering', 'cut_off')
            cut_off        = cut_off.split(',')
            try:
                cut_off[0] = float(cut_off[0])
            except Exception:
                if comm.rank == 0:
                    print_and_log(['First value of cut off must be a valid number'], 'error', logger)
                sys.exit(1)

            cut_off[1] = cut_off[1].replace(' ', '')
            if cut_off[1] == 'auto':
                cut_off[1] = 0.95*(params.rate/2.)
            else:
                try:
                    cut_off[1] = float(cut_off[1])
                except Exception:
                    if comm.rank == 0:
                        print_and_log(['Second value of cut off must either auto, or a valid a number'], 'error', logger)
                    sys.exit(1)

        chunk_size    = params.getint('data', 'chunk_size')
        nb_chunks, _  = data_file_in.analyze(chunk_size)

        b, a          = signal.butter(3, np.array(cut_off)/(params.rate/2.), 'pass')
        all_chunks    = numpy.arange(nb_chunks, dtype=numpy.int64)
        to_process    = all_chunks[comm.rank::comm.size]
        loc_nb_chunks = len(to_process)
        N_total       = params.nb_channels

        if comm.rank == 0:
            to_write = []
            if do_filtering:
                to_write += ["Filtering the signal with a Butterworth filter in (%g, %g) Hz" %(cut_off[0],cut_off[1])]
            if do_remove_median:
                to_write += ["Median over all channels is substracted to each channels"]

            print_and_log(to_write, 'default', logger)

        to_explore = xrange(comm.rank, nb_chunks, comm.size)

        if comm.rank == 0:
            to_explore = get_tqdm_progressbar(to_explore)

        for count, gidx in enumerate(to_explore):

            local_chunk, t_offset =  data_file_in.get_data(gidx, chunk_size)

            if do_filtering:
                for i in nodes:    
                    try:           
                        local_chunk[:, i]  = signal.filtfilt(b, a, local_chunk[:, i])
                    except Exception:
                        pass
                local_chunk[:, i] -= numpy.median(local_chunk[:, i]) 

            if do_remove_median:
                if not numpy.all(nodes == numpy.arange(N_total)):
                    global_median = numpy.median(numpy.take(local_chunk, nodes, axis=1), 1)
                else:
                    global_median = numpy.median(local_chunk, 1)
                for i in nodes:
                    local_chunk[:, i] -= global_median

            if data_file_in != data_file_out and data_file_in.is_first_chunk(gidx, nb_chunks):
                if data_file_in.is_stream:
                    g_offset = t_offset - numpy.sum(data_file_in._times[:data_file_in._get_streams_index_by_time(t_offset)+1])
                else:
                    g_offset = t_offset - data_file_in.t_start
            else:
                g_offset = t_offset

            data_file_out.set_data(g_offset, local_chunk)



        comm.Barrier()


    def compute_artefacts(data_file):

        chunk_size     = params.getint('data', 'chunk_size')
        trig_in_ms     = params.getboolean('triggers', 'trig_in_ms')
        artefacts      = numpy.loadtxt(params.get('triggers', 'trig_file'))
        windows        = numpy.loadtxt(params.get('triggers', 'trig_windows'))
        make_plots     = params.get('triggers', 'make_plots')
        plot_path      = os.path.join(params.get('data', 'data_file_noext'), 'plots')

        if len(windows.shape) == 1:
            windows = windows.reshape(1, 2)

        if trig_in_ms:
            if comm.rank == 0:
                print_and_log(['Artefact times are read in ms'], 'debug', logger)
            artefacts[:, 1] *= numpy.int64(data_file.sampling_rate*1e-3)
            windows[:, 1]   *= numpy.int64(data_file.sampling_rate*1e-3)
        else:
            if comm.rank == 0:
                print_and_log(['Artefact times are read in timesteps'], 'debug', logger)
            artefacts        = artefacts.astype(numpy.int64)
            windows          = windows.astype(numpy.int64)

        nb_stimuli       = len(numpy.unique(artefacts[:, 0]))
        mytest           = nb_stimuli == len(windows)

        if not mytest:
            if comm.rank == 0:
                print_and_log(['Error in the trigger files'], 'error', logger)
            sys.exit(1)

        all_labels   = artefacts[:, 0].astype(numpy.int32)
        all_times    = artefacts[:, 1].astype(numpy.int32)

        mask         = (all_times >= 0) & (all_times + numpy.max(windows[:,1]) < data_file.t_stop)
        all_times    = numpy.compress(mask, all_times)
        all_labels   = numpy.compress(mask, all_labels)

        local_labels = numpy.unique(all_labels)[comm.rank::comm.size]

        if comm.rank == 0:
            to_write = ["Computing averaged artefacts from %d stimuli" %(nb_stimuli)]
            print_and_log(to_write, 'default', logger)
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            local_labels = get_tqdm_progressbar(local_labels)

        comm.Barrier()
        # First we need to get the average artefacts
        art_dict = {}
        for count, artefact in enumerate(local_labels):
            indices  = numpy.where(all_labels == artefact)[0].astype(numpy.int32)
            tmp      = numpy.where(windows[:, 0] == artefact)[0]
            tau      = numpy.int64(windows[tmp, 1])
            pspikes  = all_times[indices]
            times    = numpy.sort(numpy.random.permutation(pspikes)[:500])
            if len(numpy.where(numpy.diff(times) < tau)[0]) > 0:
                if comm.rank == 0:
                    print_and_log(['Stimulation times for artefact %d are too close!' %artefact], 'error', logger)
                sys.exit(1)

            art_dict[artefact] = get_artefact(params, times, tau, nodes)
            if make_plots not in ['None', '']:
                save     = [plot_path, '%d.%s' %(artefact, make_plots)]
                plot.view_artefact(art_dict[artefact], save=save)


        return art_dict


    def remove_artefacts(data_file, art_dict):

        chunk_size     = params.getint('data', 'chunk_size')
        trig_in_ms     = params.getboolean('triggers', 'trig_in_ms')
        artefacts      = numpy.loadtxt(params.get('triggers', 'trig_file')).astype(numpy.int64)
        windows        = numpy.loadtxt(params.get('triggers', 'trig_windows')).astype(numpy.int64)
        make_plots     = params.get('triggers', 'make_plots')
        plot_path      = os.path.join(params.get('data', 'data_file_noext'), 'plots')

        if len(windows.shape) == 1:
            windows = windows.reshape(1, 2)

        if trig_in_ms:
            if comm.rank == 0:
                print_and_log(['Artefact times are read in ms'], 'debug', logger)
            artefacts[:, 1] *= numpy.int64(data_file.sampling_rate*1e-3)
            windows[:, 1]   *= numpy.int64(data_file.sampling_rate*1e-3)
        else:
            if comm.rank == 0:
                print_and_log(['Artefact times are read in timesteps'], 'debug', logger)
            artefacts        = artefacts.astype(numpy.int64)
            windows          = windows.astype(numpy.int64)
 
        nb_stimuli       = len(numpy.unique(artefacts[:, 0]))
        mytest           = nb_stimuli == len(windows)

        if not mytest:
            if comm.rank == 0:
                print_and_log(['Error in the trigger files'], 'error', logger)
            sys.exit(1)

        all_labels   = artefacts[:, 0].astype(numpy.int32)
        all_times    = artefacts[:, 1].astype(numpy.int32)
        local_labels = numpy.unique(all_labels)[comm.rank::comm.size]

        mask       = numpy.in1d(all_labels, local_labels)
        all_times  = numpy.compress(mask, all_times)
        all_labels = numpy.compress(mask, all_labels)

        mask       = (all_times >= 0) & (all_times < data_file.t_stop)
        all_times  = numpy.compress(mask, all_times)
        all_labels = numpy.compress(mask, all_labels)

        if comm.rank == 0:
            to_write = ["Removing artefacts from %d stimuli" %(nb_stimuli)]
            print_and_log(to_write, 'default', logger)
            all_times = get_tqdm_progressbar(all_times)

        comm.Barrier()

        for count, time in enumerate(all_times):

            label = all_labels[count]
            tmp   = numpy.where(windows[:, 0] == label)[0]
            tau   = numpy.int64(windows[tmp, 1])

            if (data_file.t_stop - time) < tau:
                tau   = max_offset - time

            local_chunk   = data_file.get_snippet(time, tau)

            for idx, i in enumerate(nodes):
                local_chunk[:, i] -= art_dict[label][idx, :tau]
            data_file.set_data(time, local_chunk)

        comm.Barrier()


    if comm.rank == 0:
        print_and_log(['Initializing the filtering step...'], 'debug', logger)

    if params.getboolean('data', 'overwrite'):
        if comm.rank == 0:
            print_and_log(['Reading the input file...'], 'debug', logger)

        data_file_in  = params.get_data_file()
        data_file_out = data_file_in
    else:
        if comm.rank == 0:
            print_and_log(['Overwrite is set to False, so creating a new datafile...'], 'debug', logger)

        if comm.rank == 0:
            print_and_log(['Reading the input file...'], 'debug', logger)

        data_file_in = params.get_data_file(source=True, has_been_created=False)

        import copy
        tmp_params   = copy.deepcopy(data_file_in._params)

        if comm.rank == 0:
            print_and_log(['Reading the output file and allocating ressources...'], 'debug', logger)

        description                 = data_file_in.get_description()
        description['data_dtype']   = 'float32'
        description['dtype_offset'] = 0
        description['data_offset']  = 0

        data_file_out = params.get_data_file(is_empty=True, params=description)

        data_file_out.allocate(shape=data_file_in.shape)
        data_file_in._params = tmp_params
        if data_file_in.is_stream:
            for source in data_file_in._sources:
                source._params = tmp_params   

    if clean_artefact:
        if not (os.path.exists(params.get('triggers', 'trig_file')) and os.path.exists(params.get('triggers', 'trig_windows'))):
            if comm.rank == 0:
                print_and_log(['trig_file or trig_windows file can not be found'], 'error', logger)
            sys.exit(1)

    to_write = []

    if do_filter and filter_done:
        do_filter = False
        to_write += ["Filtering has already been done"]
    if remove_median and median_done:
        remove_median = False
        to_write += ["Median over all channels has already been substracted to each channels"]

    if comm.rank == 0:
        print_and_log(to_write, 'debug', logger)

    if params.getboolean('data', 'overwrite'):
        data_file_in.open(mode='r+')
    else:
        data_file_in.open()
        data_file_out.open(mode='r+')

    if do_filter or remove_median:
        filter_file(data_file_in, data_file_out, do_filter, remove_median)

    if comm.rank == 0:
        if do_filter:
            params.write('noedits', 'filter_done', 'True')
        if remove_median:
            params.write('noedits', 'median_done', 'True')


    if clean_artefact and artefacts_done:
        clean_artefact = False
        if comm.rank == 0:
            print_and_log(['Artefacts have already been removed'], 'debug', logger)

    if clean_artefact:
        art_dict   = compute_artefacts(data_file_in)
        remove_artefacts(data_file_out, art_dict)

    if comm.rank == 0:
        if clean_artefact:
            params.write('noedits', 'artefacts_done', 'True')

    data_file_in.close()
    if not params.getboolean('data', 'overwrite'):
        data_file_out.close()

    comm.Barrier()
