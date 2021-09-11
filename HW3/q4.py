import numpy as np
import matplotlib.pyplot as plt
from q2 import Percolation


def find_q_inf(step_size=.05, sample_numbers=100, sample_lengths=(10, 100, 200), file_name=None):
    if file_name is None:
        p_s = np.arange(0, 1 + step_size, step_size)
        samples_clusters_size_ensemble = np.zeros((len(sample_lengths), len(p_s), sample_numbers))
        for (sample, length) in enumerate(sample_lengths):
            print(str(sample) + ": ")
            model = Percolation(length)
            for (p_i, p) in enumerate(p_s):
                print("   " + str(p))
                for sample_number in range(sample_numbers):
                    model.render(p)
                    clusters, sizes = model.infinity_clusters()
                    if len(clusters) != 0:
                        samples_clusters_size_ensemble[sample][p_i][sample_number] = sizes[0]

        data = {
            'p_s': p_s,
            'size_s': samples_clusters_size_ensemble,
            'lengths': sample_lengths,
            'numbers': sample_numbers
        }
        np.save("data/q4_" + str(step_size) + "_" + str(sample_numbers) + "_" + str(sample_lengths) + "_ensemble", data)
    else:
        data = np.load('data/q4_' + file_name + '_ensemble.npy', allow_pickle=True).tolist()
        p_s = data['p_s']
        samples_clusters_size_ensemble = data['size_s']
        sample_lengths = data['lengths']
        sample_numbers = data['numbers']

    clusters_size_avg = np.average(samples_clusters_size_ensemble, axis=2)
    clusters_size_std = np.std(samples_clusters_size_ensemble, ddof=1, axis=2) / np.sqrt(sample_numbers)

    for sample in range(len(sample_lengths)):
        length = sample_lengths[sample]
        plt.errorbar(p_s, clusters_size_avg[sample] / (length * length), yerr=clusters_size_std[sample] / (length * length), linestyle='--', elinewidth=.8)

    plt.xlabel(r'$p$')
    plt.ylabel(r'$Q_\infty$')
    plt.axis([-.05, 1.05, -.05, 1.05])
    legends = list(map(lambda length: r'L = ' + str(length), sample_lengths))
    plt.legend(legends)
    plt.savefig("images/q4_" + str(step_size) + "_" + str(sample_numbers) + "_" + str(sample_lengths) + '.png')
    plt.show()


find_q_inf(0.01, 200, (10, 100, 200))
