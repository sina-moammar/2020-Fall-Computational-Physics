import numpy as np
import time


def monte_carlo(func, low, high, scale_coefficient=None, random_generator=None, sample_numbers=None, density_func=None):
    start_time = time.time()
    integral_func = func
    if density_func is not None:
        integral_func = lambda x: func(x) / density_func(x)

    if sample_numbers is None:
        sample_numbers = int(np.abs((high - low) * 1000))

    if random_generator is None:
        random_generator = lambda n: (high - low) * np.random.rand(n) + low

    if scale_coefficient is None:
        scale_coefficient = high - low

    random_x = random_generator(sample_numbers)
    func_values = integral_func(random_x)
    integral_value = scale_coefficient * np.mean(func_values)
    value_error = scale_coefficient * np.std(func_values) / np.sqrt(sample_numbers)

    end_time = time.time()

    return integral_value, value_error, (end_time - start_time)


if __name__ == "__main__":
    func = lambda x: np.exp(-(x ** 2))
    low = 0
    high = 2
    sample_numbers = 10 ** np.arange(3, 9)

    for sample in sample_numbers:
        simple_sampling_value, simple_sampling_error, simple_sampling_runtime = monte_carlo(
            func, low, high, sample_numbers=sample)

        random_generator_1 = lambda n: -np.log(np.exp(-low) - (np.exp(-low) - np.exp(-high)) * np.random.rand(n))
        density_func_1 = lambda x: np.exp(-x)
        scale_coefficient_1 = np.exp(-low) - np.exp(-high)
        important_sampling_value_1, important_sampling_error_1, important_sampling_runtime_1 = monte_carlo(
            func, low, high, scale_coefficient_1, random_generator_1, sample, density_func_1)

        random_generator_2 = lambda n: np.tan((np.arctan(high) - np.arctan(low)) * np.random.rand(n))
        density_func_2 = lambda x: 1 / (x ** 2 + 1)
        scale_coefficient_2 = np.arctan(high) - np.arctan(low)
        important_sampling_value_2, important_sampling_error_2, important_sampling_runtime_2 = monte_carlo(
            func, low, high, scale_coefficient_2, random_generator_2, sample, density_func_2)

        print(f'Sample Numbers: {sample}:')

        print(f'Simple Sampling:\n'
              f'\tvalue: {simple_sampling_value}\n'
              f'\terror: {simple_sampling_error}\n'
              f'\ttime: {simple_sampling_runtime}')

        print(f'Important Sampling: g(x) = e^(-x)\n'
              f'\tvalue: {important_sampling_value_1}\n'
              f'\terror: {important_sampling_error_1}\n'
              f'\ttime: {important_sampling_runtime_1}')

        print(f'Important Sampling: g(x) = 1 / (x^2 + 1)\n'
              f'\tvalue: {important_sampling_value_2}\n'
              f'\terror: {important_sampling_error_2}\n'
              f'\ttime: {important_sampling_runtime_2}')

        print(end='\n')
