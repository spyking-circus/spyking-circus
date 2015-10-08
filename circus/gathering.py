def main(filename, params, nb_cpu, nb_gpu, use_gpu):
    io.collect_data(nb_cpu, params, erase=False)
