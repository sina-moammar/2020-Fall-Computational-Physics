import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def model_line_func(t, a, b):
    return a * t + b


def test_correlation(number=4, start_power=14, end_power=24, base=2):
    length_s = base ** np.arange(start_power, end_power)
    random_numbers = np.random.randint(0, 10, length_s[-1])

    std_s = np.zeros(len(length_s))
    for length_index, length in enumerate(length_s):
        next_indexes = np.roll(random_numbers[:length] == number, 1)
        next_numbers = random_numbers[:length][next_indexes]
        numbers, counts = np.unique(next_numbers, return_counts=True)
        std_s[length_index] = np.std(counts) / length

    std_s_ln = np.log(std_s)
    length_s_ln = np.log(length_s)
    std_para, std_error = curve_fit(model_line_func, length_s_ln, std_s_ln)
    std_error = np.diag(std_error)
    std_fit = np.exp(std_para[1] + length_s_ln * std_para[0])

    next_indexes = np.roll(random_numbers == number, 1)
    next_numbers = random_numbers[next_indexes]
    plt.hist(next_numbers, bins=np.arange(-0.5, 10, 1), rwidth=.8)
    plt.axhline(y=(length_s[-1] / 100), color='r', linestyle='-', linewidth=2)
    plt.xticks(np.arange(0, 10, 1))
    plt.xlabel(r'numbers')
    plt.ylabel(r'counts')
    plt.savefig("images/q2_histogram_" + str(number) + "_" + str(start_power) + "_" + str(end_power) + "_" + str(base) + '.png')
    plt.show()

    plt.plot(length_s, std_s, linestyle='', marker='o')
    plt.plot(length_s, std_fit, linestyle='--')
    plt.xlabel(r'N')
    plt.ylabel(r'$\frac{\sigma}{N}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig("images/q2_sigma_" + str(number) + "_" + str(start_power) + "_" + str(end_power) + "_" + str(base) + '.png')
    plt.show()

    print('Sigma Slope: ' + str(std_para[0]) + ' ± ' + str(std_error[0]))


test_correlation()
