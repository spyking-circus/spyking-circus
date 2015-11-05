from scipy import signal

from .shared.utils import *


def main(filename, params, nb_cpu, nb_gpu, use_gpu):

    #################################################################
    sampling_rate  = params.getint('data', 'sampling_rate')
    N_total        = params.getint('data', 'N_total')
    chunk_size     = params.getint('data', 'chunk_size')
    data_file      = params.get('data', 'data_file')
    data_offset    = params.getint('data', 'data_offset')
    dtype_offset   = params.getint('data', 'dtype_offset')
    data_dtype     = params.get('data', 'data_dtype')
    do_filter      = params.getboolean('filtering', 'filter')
    filter_done    = params.getboolean('noedits', 'filter_done')
    cut_off        = params.getint('filtering', 'cut_off')
    #################################################################

    if filter_done:
        if comm.rank == 0:
            io.print_info(["Filtering has already been done with cut off at %dHz" %cut_off])

    elif do_filter:

        borders, nb_chunks, chunk_len, last_chunk_len = io.analyze_data(params)
        if last_chunk_len > 0:
            nb_chunks += 1

        b, a          = signal.butter(3, (cut_off/(sampling_rate/2.), 0.95), 'pass')
        all_chunks    = numpy.arange(nb_chunks)
        to_process    = all_chunks[numpy.arange(comm.rank, nb_chunks, comm.size)]
        loc_nb_chunks = len(to_process)

        if comm.rank == 0:
            print "Filtering the signal with a Butterworth filter in", (cut_off, int(0.95*(sampling_rate/2))), "Hz"
            pbar = get_progressbar(loc_nb_chunks)

        myfile   = MPI.File()
        data_mpi = get_mpi_type(data_dtype)
        g        = myfile.Open(comm, data_file, MPI.MODE_RDWR)
        g.Set_view(data_offset, data_mpi, data_mpi)

        for count, gidx in enumerate(to_process):

            if (last_chunk_len > 0) and (gidx == (nb_chunks - 1)):
                data_len   = last_chunk_len
                chunk_size = last_chunk_len/N_total
            else:
                data_len   = chunk_len

            local_chunk   = numpy.zeros(data_len, dtype=data_dtype)
            g.Read_at(gidx*chunk_len, local_chunk)
            local_shape   = chunk_size
            local_chunk   = local_chunk.reshape(local_shape, N_total)
            local_chunk   = local_chunk.astype(numpy.float32)
            local_chunk  -= dtype_offset
            for i in xrange(N_total):
                try:
                    local_chunk[:, i] = signal.filtfilt(b, a, local_chunk[:, i])
                except Exception:
                    pass
            local_chunk  += dtype_offset
            local_chunk   = local_chunk.astype(data_dtype)
            local_chunk   = local_chunk.reshape(local_shape * N_total)

            g.Write_at(gidx*chunk_len, local_chunk)

            if comm.rank == 0:
                pbar.update(count)

        g.Close()

        if comm.rank == 0:
            pbar.finish()
            io.change_flag(filename, 'filter_done', 'True')

    comm.Barrier()
