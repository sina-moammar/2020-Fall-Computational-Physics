import numpy as np
import matplotlib.pyplot as plt


def gaussian_func(sigma, x):
    return 1 / np.sqrt(2 * np.pi * (sigma ** 2)) * np.exp(-(x ** 2) / (2 * (sigma ** 2)))


def gaussian_random_generator(sigma=5, numbers=100000):
    uniform_random_numbers = np.random.rand(numbers, 2)
    rho = sigma * np.sqrt(-2 * np.log(1 - uniform_random_numbers[:, 0]))
    theta = 2 * np.pi * uniform_random_numbers[:, 1]

    gaussian_random = rho * np.array([np.cos(theta), np.sin(theta)])
    gaussian_random_numbers = gaussian_random.flatten()

    min_value = np.min(gaussian_random_numbers)
    max_value = np.max(gaussian_random_numbers)
    x = np.linspace(min_value, max_value + 1, 1000)
    y = gaussian_func(sigma, x)
    plt.hist(gaussian_random_numbers, bins=np.arange(min_value - 0.5, max_value + 1, 1), density=True)
    plt.plot(x, y, linewidth=3)
    plt.xlabel(r'numbers')
    plt.ylabel(r'density')
    plt.savefig("images/q4_" + str(sigma) + "_" + str(numbers) + '.png')
    plt.show()


gaussian_random_generator()
