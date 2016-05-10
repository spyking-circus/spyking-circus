from scipy import signal
from .shared import plot
from .shared.utils import *


def main(filename, params, nb_cpu, nb_gpu, use_gpu):

    #################################################################
    multi_files    = params.getboolean('data', 'multi-files')
    data_offset    = params.getint('data', 'data_offset')
    dtype_offset   = params.getint('data', 'dtype_offset')
    data_dtype     = params.get('data', 'data_dtype')
    do_filter      = params.getboolean('filtering', 'filter')
    filter_done    = params.getboolean('noedits', 'filter_done')
    clean_artefact = params.getboolean('triggers', 'clean_artefact')
    cut_off        = params.getint('filtering', 'cut_off')
    sampling_rate  = params.getint('data', 'sampling_rate')
    remove_median  = params.getboolean('filtering', 'remove_median')
    nodes, edges   = io.get_nodes_and_edges(params)
    #################################################################

    if filter_done:
        if comm.rank == 0:
            to_write = ["Filtering has already been done with cut off at %dHz" %cut_off]
            if remove_median:
                to_write += ["Median over all channels was substracted to each channels"]
            io.print_and_log(to_write, 'info', params)

    elif do_filter:

        def filter_file(params, comm, mpi_input, mpi_output, offset=0):

            N_total        = params.getint('data', 'N_total')
            cut_off        = params.getint('filtering', 'cut_off')
            chunk_size     = params.getint('whitening', 'chunk_size')

            borders, nb_chunks, chunk_len, last_chunk_len = io.analyze_data(params, chunk_size)
            if last_chunk_len > 0:
                nb_chunks += 1

            b, a          = signal.butter(3, (cut_off/(sampling_rate/2.), 0.95), 'pass')
            all_chunks    = numpy.arange(nb_chunks)
            to_process    = all_chunks[comm.rank::comm.size]
            loc_nb_chunks = len(to_process)

            if comm.rank == 0:
                to_write = ["Filtering the signal with a Butterworth filter in (%g, %g) Hz" %(cut_off, int(0.95*(sampling_rate/2)))]
                if remove_median:
                    to_write += ["Median over all channels is substracted to each channels"]
                io.print_and_log(to_write, 'info', params)
                pbar = get_progressbar(loc_nb_chunks)

            for count, gidx in enumerate(to_process):

                if (last_chunk_len > 0) and (gidx == (nb_chunks - 1)):
                    data_len   = last_chunk_len
                    chunk_size = last_chunk_len//N_total
                else:
                    data_len   = chunk_len

                local_chunk   = numpy.zeros(data_len, dtype=data_dtype)
                mpi_input.Read_at(gidx*chunk_len, local_chunk)
                local_shape   = chunk_size
                local_chunk   = local_chunk.reshape(local_shape, N_total)
                local_chunk   = local_chunk.astype(numpy.float32)
                local_chunk  -= dtype_offset
                for i in nodes:
                    try:
                        local_chunk[:, i]  = signal.filtfilt(b, a, local_chunk[:, i])
                    except Exception:
                        pass
                    local_chunk[:, i] -= numpy.median(local_chunk[:, i]) 
                if remove_median:
                    if not numpy.all(nodes == numpy.arange(N_total)):
                        global_median = numpy.median(numpy.take(local_chunk, nodes, axis=1), 1)
                    else:
                        global_median = numpy.median(local_chunk, 1)
                    for i in nodes:
                        local_chunk[:, i] -= global_median

                local_chunk  += dtype_offset
                local_chunk   = local_chunk.astype(data_dtype)
                local_chunk   = local_chunk.ravel()

                mpi_output.Write_at(gidx*chunk_len+offset, local_chunk)

                if comm.rank == 0:
                    pbar.update(count)

            if comm.rank == 0:
                pbar.finish()

            comm.Barrier()

        def compute_artefacts(params, comm):

            cut_off        = params.getint('filtering', 'cut_off')
            chunk_size     = params.getint('whitening', 'chunk_size')
            artefacts      = numpy.loadtxt(params.get('triggers', 'trig_file'))
            windows        = numpy.loadtxt(params.get('triggers', 'trig_windows'))
            make_plots     = params.get('triggers', 'make_plots')
            plot_path      = os.path.join(params.get('data', 'data_file_noext'), 'plots')

            if len(windows.shape) == 1:
                windows = windows.reshape(1, 2)

            artefacts[:, 1] *= int(sampling_rate*1e-3)
            windows[:, 1]   *= int(sampling_rate*1e-3)
            nb_stimuli       = len(numpy.unique(artefacts[:, 0]))
            mytest           = nb_stimuli == len(windows)

            if not mytest:
                io.print_and_log(['Error in the trigger files'], 'error', params)
                sys.exit(0)

            all_labels   = artefacts[:, 0]
            all_times    = artefacts[:, 1]
            local_labels = numpy.unique(all_labels)[comm.rank::comm.size]

            if comm.rank == 0:
                to_write = ["Computing averaged artefacts from %d stimuli" %(nb_stimuli)]
                io.print_and_log(to_write, 'info', params)
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
                        io.print_and_log(['Stimulation times for artefact %d are too close!' %artefact], 'error', params)
                    sys.exit(0)
                art_dict[artefact] = io.get_artefact(params, times, tau, nodes)
                if make_plots not in ['None', '']:
                    save     = [plot_path, '%d.%s' %(artefact, make_plots)]
                    plot.view_artefact(art_dict[count], save=save)

                if comm.rank == 0:
                    pbar.update(count)

            if comm.rank == 0:
                pbar.finish()

            return art_dict


        def remove_artefacts(params, comm, art_dict, mpi_file, max_offset):

            N_total        = params.getint('data', 'N_total')
            cut_off        = params.getint('filtering', 'cut_off')
            chunk_size     = params.getint('whitening', 'chunk_size')
            artefacts      = numpy.loadtxt(params.get('triggers', 'trig_file'))
            windows        = numpy.loadtxt(params.get('triggers', 'trig_windows'))
            make_plots     = params.get('triggers', 'make_plots')
            plot_path      = os.path.join(params.get('data', 'data_file_noext'), 'plots')

            if len(windows.shape) == 1:
                windows = windows.reshape(1, 2)

            artefacts[:, 1] *= int(sampling_rate*1e-3)
            windows[:, 1]   *= int(sampling_rate*1e-3)
            nb_stimuli       = len(numpy.unique(artefacts[:, 0]))
            mytest           = nb_stimuli == len(windows)

            if not mytest:
                io.print_and_log(['Error in the trigger files'], 'error', params)
                sys.exit(0)

            all_labels   = artefacts[:, 0]
            all_times    = artefacts[:, 1]
            local_labels = numpy.unique(all_labels)[comm.rank::comm.size]

            if comm.rank == 0:
                to_write = ["Removing artefacts from %d stimuli" %(nb_stimuli)]
                io.print_and_log(to_write, 'info', params)
                pbar = get_progressbar(len(all_times))

            comm.Barrier()
            
            count    = 0
            
            for label, time in zip(all_labels, all_times):

                if (time >= 0) and (time < max_offset) and (label in local_labels):

                    tmp      = numpy.where(windows[:, 0] == label)[0]
                    tau      = windows[tmp, 1]
                    mshape   = tau
                    data_len = tau * N_total
                    if (max_offset - time) < tau:
                        data_len = (max_offset - time)*N_total
                        mshape   = max_offset - time

                    local_chunk   = numpy.zeros(data_len, dtype=data_dtype)
                    mpi_file.Read_at(N_total * time, local_chunk)
                    local_chunk   = local_chunk.reshape(mshape, N_total)
                    local_chunk   = local_chunk.astype(numpy.float32)
                    local_chunk  -= dtype_offset
                    for idx, i in enumerate(nodes):
                        local_chunk[:, i] -= art_dict[label][idx, :mshape]
                        
                    local_chunk  += dtype_offset
                    local_chunk   = local_chunk.astype(data_dtype)
                    local_chunk   = local_chunk.ravel()

                    mpi_file.Write_at(N_total*time, local_chunk)

                count        += 1

                if comm.rank == 0:
                    pbar.update(count)

            if comm.rank == 0:
                pbar.finish()

            comm.Barrier()

        myfile   = MPI.File()
        data_mpi = get_mpi_type(data_dtype)

        if not multi_files:            
            mpi_in = myfile.Open(comm, params.get('data', 'data_file'), MPI.MODE_RDWR)
            mpi_in.Set_view(data_offset, data_mpi, data_mpi)
            offset = (mpi_in.size//data_mpi.size)
            filter_file(params, comm, mpi_in, mpi_in)

            if clean_artefact:
                art_dict   = compute_artefacts(params, comm)
                remove_artefacts(params, comm, art_dict, mpi_in, offset)

            mpi_in.Close()
        else:
            all_files = io.get_multi_files(params)
            all_times = io.data_stats(params, show=False, export_times=True)

            if comm.rank == 0:
                io.copy_header(data_offset, params.get('data', 'data_multi_file'), params.get('data', 'data_file'))
                
            comm.Barrier()
            
            combined_file = params.get('data', 'data_file')
            mpi_out       = myfile.Open(comm, combined_file, MPI.MODE_RDWR)
            mpi_out.Set_view(data_offset, data_mpi, data_mpi)
            io.write_to_logger(params, ['Output file: %s' %params.get('data', 'data_file') ], 'debug')

            offset   = 0

            for data_file in all_files:
                mpi_in = myfile.Open(comm, data_file, MPI.MODE_RDWR)
                if params.getboolean('data', 'MCS'):
                    data_offset, nb_channels = io.detect_header(data_file, 'MCS')
                mpi_in.Set_view(data_offset, data_mpi, data_mpi) 
                params.set('data', 'data_file', data_file)
                io.write_to_logger(params, ['Input file for filtering: %s' %params.get('data', 'data_file') ], 'debug')
                filter_file(params, comm, mpi_in, mpi_out, offset)
                offset += (mpi_in.size//data_mpi.size)               
                mpi_in.Close()

            params.set('data', 'data_file', combined_file)

            if clean_artefact:
                art_dict   = compute_artefacts(params, comm)
                remove_artefacts(params, comm, art_dict, mpi_out, offset)

            mpi_out.Close()

        if comm.rank == 0:
            io.change_flag(filename, 'filter_done', 'True')

    comm.Barrier()
