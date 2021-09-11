import numpy as np


def compute_com(radius, density_func, sample_numbers=10000):
    random_x = (2 * radius) * (np.random.rand(sample_numbers) - .5)
    random_y = (2 * radius) * (np.random.rand(sample_numbers) - .5)
    random_z = (2 * radius) * (np.random.rand(sample_numbers) - .5)

    valid_indexes = random_x ** 2 + random_y ** 2 + random_z ** 2 < radius ** 2
    random_valid_x = random_x[valid_indexes]
    random_valid_y = random_y[valid_indexes]
    random_valid_z = random_z[valid_indexes]

    densities = density_func(random_valid_x, random_valid_y, random_valid_z)
    average_density = np.mean(densities)

    weighted_x = densities * random_valid_x
    center_of_mass_x = np.mean(weighted_x) / average_density
    center_of_mass_x_error = np.std(weighted_x) / average_density / np.sqrt(len(random_x))

    weighted_y = densities * random_valid_y
    center_of_mass_y = np.mean(weighted_y) / average_density
    center_of_mass_y_error = np.std(weighted_y) / average_density / np.sqrt(len(random_y))

    weighted_z = densities * random_valid_z
    center_of_mass_z = np.mean(weighted_z) / average_density
    center_of_mass_z_error = np.std(weighted_z) / average_density / np.sqrt(len(random_z))

    return [[center_of_mass_x, center_of_mass_y, center_of_mass_z],
            [center_of_mass_x_error, center_of_mass_y_error, center_of_mass_z_error]]


if __name__ == "__main__":
    radius = 5
    sample_numbers = 100000000

    def density_func(radius, x, y, z):
        return (z + radius) / (2 * radius) + 1

    coordination, errors = compute_com(radius, lambda x, y, z: density_func(radius, x, y, z), sample_numbers=sample_numbers)
    print(f'X = {coordination[0]} ± {errors[0]}')
    print(f'Y = {coordination[1]} ± {errors[1]}')
    print(f'Z = {coordination[2]} ± {errors[2]}')
