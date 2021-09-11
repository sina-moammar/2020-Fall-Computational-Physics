import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from q2 import Percolation


def find_correlation_radius(step_size=.05, p_start=0.0, p_end=1.0, sample_numbers=100,
                            sample_lengths=(10, 20, 40, 80, 160), file_name=None):
    if file_name is None:
        p_s = np.arange(p_start, p_end + step_size, step_size)
        samples_correlation_radii_ensemble = np.zeros((len(sample_lengths), len(p_s), sample_numbers))
        for (sample, length) in enumerate(sample_lengths):
            print(str(sample) + ": ")
            model = Percolation(length)
            for (p_i, p) in enumerate(p_s):
                print("   " + str(p))
                for sample_number in range(sample_numbers):
                    model.render(p)
                    samples_correlation_radii_ensemble[sample][p_i][sample_number] = model.correlation_length()

        data = {
            'p_s': p_s,
            'radii': samples_correlation_radii_ensemble,
            'lengths': sample_lengths,
            'numbers': sample_numbers
        }
        np.save("data/q5_" + str(p_start) + "_" + str(step_size) + "_" + str(p_end) + "_" + str(sample_numbers) + "_" + str(sample_lengths) + "_ensemble", data)
    else:
        data = np.load('data/q5_' + file_name + '_ensemble.npy', allow_pickle=True).tolist()
        p_s = data['p_s']
        samples_correlation_radii_ensemble = data['radii']
        sample_lengths = data['lengths']
        sample_numbers = data['numbers']

    correlation_radii_avg = np.average(samples_correlation_radii_ensemble, axis=2)
    correlation_radii_std = np.std(samples_correlation_radii_ensemble, ddof=1, axis=2) / np.sqrt(sample_numbers)

    p_c_s = np.zeros(len(sample_lengths))
    for sample in range(len(sample_lengths)):
        p_c_s[sample] = p_s[np.argmax(correlation_radii_avg[sample])]
        plt.errorbar(p_s, correlation_radii_avg[sample], yerr=correlation_radii_std[sample], linestyle='--', elinewidth=.8)

    plt.xlabel(r'$p$')
    plt.ylabel(r'$\xi$')
    legends = list(map(lambda length: r'L = ' + str(length), sample_lengths))
    plt.legend(legends)
    plt.savefig("images/q5_" + str(p_start) + "_" + str(step_size) + "_" + str(p_end) + "_" + str(sample_numbers) + "_" + str(sample_lengths) + '.png')
    plt.show()

    return p_c_s


def p_infinity_model(p, p_inf, A, v):
    return (A * np.abs(p - p_inf)) ** (-v)


if __name__ == "__main__":
    length_s = np.array([10, 20, 40, 80, 160])
    p_c_s = np.zeros(len(length_s))
    print(find_correlation_radius(0.01, 0, 1, 100, (10, 20, 40, 80, 160)))
    p_c_s[0] = find_correlation_radius(0.001, 0.45, 0.54, 600, (10,))[0]
    p_c_s[1] = find_correlation_radius(0.001, 0.5, 0.57, 300, (20,))[0]
    p_c_s[2] = find_correlation_radius(0.001, 0.53, 0.59, 100, (40,))[0]
    p_c_s[3] = find_correlation_radius(0.001, 0.55, 0.6, 30, (80,))[0]
    p_c_s[4] = find_correlation_radius(0.001, 0.56, 0.6, 10, (160,))[0]

    fit_para, fit_error = curve_fit(p_infinity_model, p_c_s, length_s, p0=(.6, 1, 1.4))
    fit_error = np.diag(fit_error)
    print('p_c(∞) = ' + str(fit_para[0]) + ' ± ' + str(fit_error[0]))
