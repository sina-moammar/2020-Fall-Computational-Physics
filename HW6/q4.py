import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def gaussian_func(mean, sigma, x):
    return 1 / np.sqrt(2 * np.pi * (sigma ** 2)) * np.exp(-((x - mean) ** 2) / (2 * (sigma ** 2)))


def model_exp_func(t, a):
    return np.exp(a * t)


class Metropolis:
    def __init__(self, distribution_func):
        self.distribution_func = distribution_func
        self.generated_numbers = []
        self.acceptance_count = 0
        self.time = 0
        self.step = 0
        self.x_0 = 0

    def generate(self, time, x_0, step):
        self.time = time
        self.step = step
        self.x_0 = x_0
        self.acceptance_count = 0
        self.generated_numbers = np.zeros(time + 1)
        self.generated_numbers[0] = x_0
        x_random_numbers = np.random.rand(time)
        probability_random_numbers = np.random.rand(time)
        prev_func_value = self.distribution_func(x_0)

        for current_time in range(time):
            next_x = self.generated_numbers[current_time] + (2 * step) * (x_random_numbers[current_time] - .5)
            next_func_value = self.distribution_func(next_x)
            move_probability = next_func_value / prev_func_value
            if probability_random_numbers[current_time] < move_probability:
                self.generated_numbers[current_time + 1] = next_x
                self.acceptance_count += 1
                prev_func_value = next_func_value
            else:
                self.generated_numbers[current_time + 1] = self.generated_numbers[current_time]

    def get_correlation_for(self, time):
        correlation_matrix = np.corrcoef(self.generated_numbers[:self.time - time], self.generated_numbers[time:self.time])
        return correlation_matrix[0, 1]

    def get_correlation_length(self, batch_count=10000):
        _THRESHOLD = .0001
        max_index = self.time - batch_count
        times = np.arange(0, max_index + 1)
        correlations = np.zeros(len(times))
        count = 0

        for index, time in enumerate(times):
            correlations[index] = self.get_correlation_for(time)
            if correlations[index] < _THRESHOLD:
                break
            count += 1

        times = times[:count]
        correlations = correlations[:count]

        correlation_fit_para, correlation_fit_error = curve_fit(model_exp_func, times, correlations)
        correlation_fit_error = np.diag(correlation_fit_error)
        correlation_fit = np.exp(correlation_fit_para[0] * times)

        plt.plot(times, correlations, linestyle='', marker='.', markersize=5)
        plt.plot(times, correlation_fit, linestyle='--', color='g')
        plt.xlabel(r'$j$')
        plt.ylabel(r'$C(j)$')
        plt.legend(['Samples', 'Fitted Curve'])
        plt.savefig('images/q4_c_j_' + str(self.x_0) + '_' + str(self.time) + '_' + str(self.step) + '.jpg')
        plt.show()

        correlation_length = -1 / correlation_fit_para[0]
        correlation_length_error = (correlation_length ** 2) * correlation_fit_error[0]

        return correlation_length, correlation_length_error

    def get_acceptance_rate(self):
        return self.acceptance_count / self.time


if __name__ == "__main__":
    sigma = 2
    sample_numbers = 1000000
    x_0 = 0
    distribution_func = lambda x: gaussian_func(x_0, sigma, x)
    steps = np.array([1.01, 2.04, 3.15, 4.41, 5.88, 7, 7.77, 10.6, 15.93, 31.92])

    for step in steps:
        model = Metropolis(distribution_func)
        model.generate(sample_numbers, x_0, step)
        a_r = model.get_acceptance_rate()
        correlation_length, correlation_length_error = model.get_correlation_length()

        generated_random_numbers = model.generated_numbers[::int(np.ceil(2 * correlation_length))]
        x_min = np.min(generated_random_numbers)
        x_max = np.max(generated_random_numbers)
        bins = int((x_max - x_min) / (sigma / 11))
        x_s = np.linspace(x_min, x_max, bins)
        gaussian_distribution = gaussian_func(x_0, sigma, x_s)
        plt.hist(generated_random_numbers, bins=bins, density=True)
        plt.plot(x_s, gaussian_distribution, linewidth=3)
        plt.xlabel(r'$x$')
        plt.ylabel(r'density')
        plt.savefig('images/q4_' + str(x_0) + '_' + str(sample_numbers) + '_' + str(step) + '.jpg')
        plt.show()

        print(f'Step: {step}')
        print(f'Acceptance Rate: {a_r}')
        print(f'Correlation Length: {correlation_length} ± {correlation_length_error}')
