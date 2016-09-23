from scipy import signal
from .shared import plot
from .shared.utils import *
from circus.shared.probes import get_nodes_and_edges
from circus.shared.messages import print_error, print_info, print_and_log


def main(params, nb_cpu, nb_gpu, use_gpu):

    #################################################################
    multi_files    = params.getboolean('data', 'multi-files')
    do_filter      = params.getboolean('filtering', 'filter')
    filter_done    = params.getboolean('noedits', 'filter_done')
    clean_artefact = params.getboolean('triggers', 'clean_artefact')
    remove_median  = params.getboolean('filtering', 'remove_median')
    nodes, edges   = get_nodes_and_edges(params)
    #################################################################

    if clean_artefact:
        if not (os.path.exists(params.get('triggers', 'trig_file')) and os.path.exists(params.get('triggers', 'trig_windows'))):
            io.print_and_log(['trig_file or trig_windows file can not be found'], 'error', params)
            sys.exit(0)

    if do_filter or multi_files or clean_artefact or remove_median:

        def filter_file(data_file_in, data_file_out=None, offset=0, perform_filtering=True, display=True):

            try:
                cut_off    = params.getfloat('filtering', 'cut_off')
                cut_off    = [cut_off, 0.95*(params.rate/2.)]
            except Exception:
                cut_off        = params.get('filtering', 'cut_off')
                cut_off        = cut_off.split(',')
                try:
                    cut_off[0] = float(cut_off[0])
                except Exception:
                    print_and_log(['First value of cut off must be a valid number'], 'error', params)
                    sys.exit(0)
                
                cut_off[1] = cut_off[1].replace(' ', '')
                if cut_off[1] == 'auto':
                    cut_off[1] = 0.95*(params.rate/2.)
                else:
                    try:
                        cut_off[1] = float(cut_off[1])
                    except Exception:
                        print_and_log(['Second value of cut off must either auto, or a valid a number'], 'error', params)
                        sys.exit(0)

            if filter_done:
                if comm.rank == 0:
                    to_write = []
                    if do_filter:
                        to_write += ["Filtering has already been done in band [%dHz, %dHz]" %(cut_off[0], cut_off[1])]
                    if remove_median:
                        to_write += ["Median over all channels was substracted to each channels"]
                    if display:
                        print_and_log(to_write, 'info', params)
                return

            if data_file_out is None:
                same_file = True
                data_file_in.open(mode='r+')
                data_file_out = data_file_in
            else:
                same_file = False
                data_file_in.open()
                data_file_out.open(mode='r+')

            chunk_size     = params.getint('data', 'chunk_size')
            nb_chunks, last_chunk_len = data_file_in.analyze(chunk_size)
            
            b, a          = signal.butter(3, np.array(cut_off)/(params.rate/2.), 'pass')
            all_chunks    = numpy.arange(nb_chunks, dtype=numpy.int64)
            to_process    = all_chunks[comm.rank::comm.size]
            loc_nb_chunks = len(to_process)

            goffset       = data_file_in.duration

            if comm.rank == 0:
                if perform_filtering:
                    to_write = ["Filtering the signal with a Butterworth filter in (%g, %g) Hz" %(cut_off[0],cut_off[1])]
                elif multi_files:
                    to_write = ["Concatenating multi files without filtering"]
                if remove_median:
                    to_write += ["Median over all channels is substracted to each channels"]
                if display:
                    print_and_log(to_write, 'default', params)

                pbar = get_progressbar(loc_nb_chunks)

            for count, gidx in enumerate(to_process):

                local_chunk =  data_file_in.get_data(gidx, chunk_size)
                
                for i in nodes:
                    if perform_filtering:
                        try:
                            local_chunk[:, i]  = signal.filtfilt(b, a, local_chunk[:, i])
                        except Exception:
                            pass
                    local_chunk[:, i] -= numpy.median(local_chunk[:, i]) 

                if remove_median:
                    if not numpy.all(nodes == numpy.arange(data_in.N_tot)):
                        global_median = numpy.median(numpy.take(local_chunk, nodes, axis=1), 1)
                    else:
                        global_median = numpy.median(local_chunk, 1)
                    for i in nodes:
                        local_chunk[:, i] -= global_median

                data_file_out.set_data(gidx*chunk_size, local_chunk)

                if comm.rank == 0:
                    pbar.update(count)

            if comm.rank == 0:
                pbar.finish()

            comm.Barrier()

            data_file_in.close()
            if not same_file:           
                data_file_out.close()

            return goffset + offset

        def compute_artefacts(data_file, max_offset):

            chunk_size     = params.getint('data', 'chunk_size')
            artefacts      = numpy.loadtxt(params.get('triggers', 'trig_file'))
            windows        = numpy.loadtxt(params.get('triggers', 'trig_windows'))
            make_plots     = params.get('triggers', 'make_plots')
            plot_path      = os.path.join(params.get('data', 'data_file_noext'), 'plots')

            if len(windows.shape) == 1:
                windows = windows.reshape(1, 2)

            artefacts[:, 1] *= numpy.int64(data_file.rate*1e-3)
            windows[:, 1]   *= numpy.int64(data_file.rate*1e-3)
            nb_stimuli       = len(numpy.unique(artefacts[:, 0]))
            mytest           = nb_stimuli == len(windows)

            if not mytest:
                print_and_log(['Error in the trigger files'], 'error', params)
                sys.exit(0)

            all_labels   = artefacts[:, 0]
            all_times    = artefacts[:, 1]

            mask         = (all_times >= 0) & (all_times + numpy.max(windows[:,1]) < max_offset)
            all_times    = numpy.compress(mask, all_times)
            all_labels   = numpy.compress(mask, all_labels)

            local_labels = numpy.unique(all_labels)[comm.rank::comm.size]

            if comm.rank == 0:
                to_write = ["Computing averaged artefacts from %d stimuli" %(nb_stimuli)]
                print_and_log(to_write, 'default', params)
                pbar = get_progressbar(len(local_labels))
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path)

            comm.Barrier()
            # First we need to get the average artefacts
            art_dict = {}
            for count, artefact in enumerate(local_labels):
                indices  = numpy.where(all_labels == artefact)[0].astype(numpy.int32)
                tmp      = numpy.where(windows[:, 0] == artefact)[0]
                tau      = windows[tmp, 1]
                pspikes  = all_times[indices]
                times    = numpy.sort(numpy.random.permutation(pspikes)[:500])
                if len(numpy.where(numpy.diff(times) < tau)[0]) > 0:
                    if comm.rank == 0:
                        print_and_log(['Stimulation times for artefact %d are too close!' %artefact], 'error', params)
                    sys.exit(0)
                art_dict[artefact] = get_artefact(params, times, tau, nodes)
                if make_plots not in ['None', '']:
                    save     = [plot_path, '%d.%s' %(artefact, make_plots)]
                    plot.view_artefact(art_dict[artefact], save=save)

                if comm.rank == 0:
                    pbar.update(count)

            if comm.rank == 0:
                pbar.finish()

            data_file.close()

            return art_dict


        def remove_artefacts(art_dict, data_file, max_offset):

            chunk_size     = params.getint('data', 'chunk_size')
            artefacts      = numpy.loadtxt(params.get('triggers', 'trig_file')).astype(numpy.int64)
            windows        = numpy.loadtxt(params.get('triggers', 'trig_windows')).astype(numpy.int64)
            make_plots     = params.get('triggers', 'make_plots')
            plot_path      = os.path.join(params.get('data', 'data_file_noext'), 'plots')

            if len(windows.shape) == 1:
                windows = windows.reshape(1, 2)

            artefacts[:, 1] *= numpy.int64(data_file.rate*1e-3)
            windows[:, 1]   *= numpy.int64(data_file.rate*1e-3)
            nb_stimuli       = len(numpy.unique(artefacts[:, 0]))
            mytest           = nb_stimuli == len(windows)

            if not mytest:
                print_and_log(['Error in the trigger files'], 'error', params)
                sys.exit(0)

            all_labels   = artefacts[:, 0]
            all_times    = artefacts[:, 1]
            local_labels = numpy.unique(all_labels)[comm.rank::comm.size]

            if comm.rank == 0:
                to_write = ["Removing artefacts from %d stimuli" %(nb_stimuli)]
                print_and_log(to_write, 'default', params)
                pbar = get_progressbar(len(all_times))

            comm.Barrier()
            
            count    = 0
    
            mask       = numpy.in1d(all_labels, local_labels)
            all_times  = numpy.compress(mask, all_times)
            all_labels = numpy.compress(mask, all_labels)

            mask       = (all_times >= 0) & (all_times < max_offset)
            all_times  = numpy.compress(mask, all_times)
            all_labels = numpy.compress(mask, all_labels)

            for label, time in zip(all_labels, all_times):

                tmp      = numpy.where(windows[:, 0] == label)[0]
                tau      = windows[tmp, 1]
                if (max_offset - time) < tau:
                    tau   = max_offset - time

                local_chunk   = data_file.get_snippet(time, tau)

                for idx, i in enumerate(nodes):
                    local_chunk[:, i] -= art_dict[label][idx, :tau]
                       
                data_file.set_data(time, local_chunk)

                count        += 1

                if comm.rank == 0:
                    pbar.update(count)

            if comm.rank == 0:
                pbar.finish()

            comm.Barrier()
            data_file.close()

        if not multi_files:  

            data_file = params.get_data_file()
            goffset   = filter_file(data_file)

            if clean_artefact:
                art_dict   = compute_artefacts(data_file, goffset)
                remove_artefacts(art_dict, data_file, goffset)

        else:

            all_files     = params.get_multi_files()
            combined_file = params.get('data', 'data_file')
            data_file     = params.get_data_file(multi=True, force_raw=False)
            comm.Barrier()

            times         = io.data_stats(params, show=False, export_times=True)
            data_out      = params.get_data_file(force_raw=True, is_empty=True)

            data_out.allocate(shape=(times[-1][1], data_out.nb_channels), data_dtype=numpy.float32)
            
            print_and_log(['Output file: %s' %combined_file], 'debug', params)
            goffset = 0
            
            for data_file in all_files:

                params.set('data', 'data_multi_file', data_file)
                data_in = params.get_data_file(multi=True, force_raw=False)

                print_and_log(['Input file for filtering: %s' %params.get('data', 'data_file') ], 'debug', params)
                goffset = filter_file(data_in, data_out, goffset, perform_filtering=do_filter, display=(goffset == 0))

            params.set('data', 'data_file', combined_file)

            if clean_artefact:
                art_dict   = compute_artefacts(data_out, goffset)
                remove_artefacts(art_dict, data_out, goffset)

        if comm.rank == 0 and (do_filter or clean_artefact):
            params.write('noedits', 'filter_done', 'True')

        sys.exit(0)

    comm.Barrier()
