import numpy as np
import matplotlib.pyplot as plt
from q2 import Percolation


def find_q(step_size=.05, sample_numbers=100, sample_lengths=(10, 100, 200), file_name=None):
    if file_name is None:
        p_s = np.arange(0, 1 + step_size, step_size)
        samples_q_s_ensemble = np.zeros((len(sample_lengths), len(p_s), sample_numbers))
        for (sample, length) in enumerate(sample_lengths):
            print(str(sample) + ": ")
            model = Percolation(length)
            for (p_i, p) in enumerate(p_s):
                print("   " + str(p))
                for sample_number in range(sample_numbers):
                    model.render(p)
                    if model.is_percolated():
                        samples_q_s_ensemble[sample][p_i][sample_number] = 1

        data = {
            'p_s': p_s,
            'q_s': samples_q_s_ensemble,
            'lengths': sample_lengths,
            'numbers': sample_numbers
        }
        np.save("data/q3_" + str(step_size) + "_" + str(sample_numbers) + "_" + str(sample_lengths) + "_ensemble", data)
    else:
        data = np.load('data/q3_' + file_name + '_ensemble.npy', allow_pickle=True).tolist()
        p_s = data['p_s']
        samples_q_s_ensemble = data['q_s']
        sample_lengths = data['lengths']
        sample_numbers = data['numbers']

    q_s_avg = np.average(samples_q_s_ensemble, axis=2)
    q_s_std = np.std(samples_q_s_ensemble, ddof=1, axis=2) / np.sqrt(sample_numbers)

    for sample in range(len(sample_lengths)):
        plt.errorbar(p_s, q_s_avg[sample], yerr=q_s_std[sample], linestyle='--', elinewidth=.8)

    plt.xlabel(r'$p$')
    plt.ylabel(r'$Q$')
    plt.axis([-.05, 1.05, -.05, 1.05])
    legends = list(map(lambda length: r'L = ' + str(length), sample_lengths))
    plt.legend(legends)
    plt.savefig("images/q3_" + str(step_size) + "_" + str(sample_numbers) + "_" + str(sample_lengths) + '.png')
    plt.show()


find_q(0.01, 200, (10, 100, 200))
