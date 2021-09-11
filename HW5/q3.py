import numpy as np
import matplotlib.pyplot as plt


def gaussian_func(mean, sigma, x):
    return 1 / np.sqrt(2 * np.pi * (sigma ** 2)) * np.exp(-((x - mean) ** 2) / (2 * (sigma ** 2)))


def test_central_limit(n_s=(5, 10, 100, 1000), sample_numbers=10**6):

    for n in n_s:
        random_numbers = np.random.randint(0, 10, (sample_numbers, n), dtype=np.int8)
        sample_sum_s = np.sum(random_numbers, axis=1)

        sigma = np.std(sample_sum_s)
        mean = np.mean(sample_sum_s)
        min_sum = np.min(sample_sum_s)
        max_sum = np.max(sample_sum_s)
        x = np.linspace(min_sum, max_sum + 1, 1000)
        gaussian_fit = gaussian_func(mean, sigma, x)
        plt.hist(sample_sum_s, bins=np.arange(min_sum - 0.5, max_sum + 1, 1), density=True)
        plt.plot(x, gaussian_fit, linewidth=3)
        plt.xlabel(r'sums')
        plt.ylabel(r'density')
        plt.savefig("images/q3_" + str(n) + "_" + str(sample_numbers) + '.png')
        plt.show()


test_central_limit()
