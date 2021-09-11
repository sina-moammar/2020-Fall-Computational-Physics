import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.constants as CONSTANTS
import math
import json
import os


class SingleAtomMD:
    def __init__(self, length, number, dimension, sigma, mass, epsilon, v_max, h=10 ** -3, saving_period=10,
                 is_reduced_units=True, name='MD'):
        self.sigma = sigma
        self.mass = mass
        self.epsilon = epsilon
        self.number = number
        self.dimension = dimension
        self.h = h
        self.h_half = h / 2
        self.reduced_time = 0
        self.name = name
        self.saving_period = saving_period
        if is_reduced_units:
            self.reduced_length = length
            self.length = length * self.sigma
            self.reduced_v_max = v_max
        else:
            self.length = length
            self.reduced_length = length / self.sigma
            self.reduced_v_max = v_max / np.sqrt(self.epsilon / self.mass)
        self.reduced_length_half = self.reduced_length / 2
        self._file_base_name = f'{self.name}_{self.reduced_length}_{self.number}_{self.dimension}_{self.h}'

        self.positions = np.zeros((self.dimension, self.number))
        self._place_atoms_left_side_regularly()
        self._periodic_boundaries()

        self.velocities = np.zeros((self.dimension, self.number))
        self._assign_initial_velocities()
        self._center_of_mass_frame()

        self._positions_diff = np.zeros((dimension, self.number, self.number))
        self._update_positions_diff()

        self._distance_matrix_2 = np.zeros((self.number, self.number))
        self._distance_matrix_6 = np.zeros((self.number, self.number))
        self._distance_matrix_12 = np.zeros((self.number, self.number))
        self._update_distance_matrices()

        self.accelerators = np.zeros((self.dimension, self.number))
        self._update_accelerators()

        self._reduced_temperature = None
        self._reduced_potential_energy = None
        self._reduced_kinetic_energy = None
        self._reduced_volume = None
        self._reduced_pressure = None
        self._number_of_left_side_atoms = None

        self._initialize_files()

    def initiate_from_positions_velocities(self, positions, velocities):
        self.positions[:][:] = positions
        self.velocities[:][:] = velocities
        self._update_positions_and_distances()
        self._update_accelerators()
        self.reduced_time = 0

        self._initialize_files()

    def _initialize_files(self):
        file = open(f'data/{self._file_base_name}.info', 'w')
        info = {
            'sigma': self.sigma,
            'mass': self.mass,
            'epsilon': self.epsilon,
            'length': self.reduced_length,
            'number': self.number,
            'dimension': self.dimension,
            'v_max': self.reduced_v_max,
            'h': self.h,
            'data_size': np.dtype(float).itemsize,
            'saving_period': self.saving_period,
            'trajectory': ['positions', 'velocities'],
            'data': ['temperature', 'potential_energy', 'kinetic_energy', 'energy', 'pressure', 'number_of_left_side_atoms']
        }
        json.dump(info, file, indent=True)
        file.close()

        file = open(f'data/{self._file_base_name}.traj', 'wb')
        self._update_trajectory_file(file)
        file.close()

        file = open(f'data/{self._file_base_name}.data', 'wb')
        self._update_data_file(file)
        file.close()

    def _update_positions_diff(self):
        for axis in range(self.dimension):
            tiled_positions = np.tile(self.positions[axis], (self.number, 1))
            np.subtract(tiled_positions, np.transpose(tiled_positions), self._positions_diff[axis])

        far_positive_indexes = self._positions_diff > self.reduced_length_half
        self._positions_diff[far_positive_indexes] -= self.reduced_length
        far_negative_indexes = self._positions_diff < -self.reduced_length_half
        self._positions_diff[far_negative_indexes] += self.reduced_length

    def _update_positions_and_distances(self):
        self._periodic_boundaries()
        self._update_positions_diff()
        self._update_distance_matrices()

    def _periodic_boundaries(self):
        out_indexes = self.positions > self.reduced_length
        self.positions[out_indexes] -= self.reduced_length
        out_indexes = self.positions < 0
        self.positions[out_indexes] += self.reduced_length

    def _update_accelerators(self):
        self.accelerators = 24 * np.sum(
            (-2 * self._distance_matrix_12 + self._distance_matrix_6) * self._distance_matrix_2 * self._positions_diff,
            axis=2)

    def _update_distance_matrices(self):
        self._distance_matrix_2 = 1 / np.sum(np.square(self._positions_diff), axis=0)
        np.fill_diagonal(self._distance_matrix_2, 0)
        self._distance_matrix_6 = self._distance_matrix_2 * self._distance_matrix_2 * self._distance_matrix_2
        self._distance_matrix_12 = self._distance_matrix_6 * self._distance_matrix_6

    @staticmethod
    def _write_list_in_file(file, list):
        file.write(list.tobytes())

    def _update_trajectory_file(self, file):
        for position in self.positions:
            self._write_list_in_file(file, position)
        for velocity in self.velocities:
            self._write_list_in_file(file, velocity)

    def _get_data_list(self):
        return np.array([
            self.get_reduced_temperature(),
            self.get_reduced_potential_energy(),
            self.get_reduced_kinetic_energy(),
            self.get_reduced_energy(),
            self.get_reduced_pressure(),
            self.number_of_left_side_atoms()
        ], dtype=float)

    def _update_data_file(self, file):
        self._write_list_in_file(file, self._get_data_list())

    def _time_step(self):
        print(f'\rTime: {self.reduced_time}', end='')
        accelerators_changes = self.accelerators * self.h_half
        self.positions += (self.velocities + accelerators_changes) * self.h
        self.velocities += accelerators_changes
        self._update_positions_and_distances()
        self._update_accelerators()
        self.velocities += self.accelerators * self.h_half
        self.reduced_time += 1

        self._reduced_temperature = None
        self._reduced_potential_energy = None
        self._reduced_kinetic_energy = None
        self._reduced_pressure = None
        self._number_of_left_side_atoms = None

    def render(self, reduced_time, flush_period=10000):
        trajectory_file = open(f'data/{self._file_base_name}.traj', 'ab')
        data_file = open(f'data/{self._file_base_name}.data', 'ab')
        flush_numbers = int(reduced_time / flush_period)
        extra_time = reduced_time % flush_period
        for flush_step in range(flush_numbers):
            for step in range(flush_period):
                self._time_step()
                if self.reduced_time % self.saving_period == 0:
                    self._update_trajectory_file(trajectory_file)
                    self._update_data_file(data_file)
            trajectory_file.flush()
            os.fsync(trajectory_file.fileno())
            data_file.flush()
            os.fsync(data_file.fileno())
        for step in range(extra_time):
            self._time_step()
            if self.reduced_time % self.saving_period == 0:
                self._update_trajectory_file(trajectory_file)
                self._update_data_file(data_file)

        print(end='\n')
        trajectory_file.close()
        data_file.close()

        return DataAnalysis(self._file_base_name)

    def _center_of_mass_frame(self):
        velocity_mean = np.mean(self.velocities, axis=1)
        for axis in range(self.dimension):
            self.velocities[axis] -= velocity_mean[axis]

    def _place_atoms_left_side_regularly(self):
        half_size = math.ceil((self.number / (2 ** (self.dimension - 1))) ** (1 / self.dimension))
        grid_distance = self.reduced_length / 2 / half_size
        grid_sizes = np.array([1, half_size, *[2 * half_size for i in range(self.dimension - 1)]])

        for axis in range(self.dimension):
            self.positions[axis] = np.tile(np.repeat(np.arange(grid_sizes[axis + 1]), np.prod(grid_sizes[:axis + 1])), np.prod(grid_sizes[axis:]))[:self.number]

        self.positions *= grid_distance
        self.positions += grid_distance / 2

    def _assign_initial_velocities(self):
        max_amp = np.ones(self.number) * self.reduced_v_max
        for axis in range(self.dimension - 1):
            random_thetas = np.random.rand(self.number) * (2 * np.pi)
            self.velocities[axis] = max_amp * np.cos(random_thetas)
            max_amp = max_amp * np.sin(random_thetas)
        self.velocities[-1] = max_amp

    def get_reduced_temperature(self):
        if self._reduced_temperature is None:
            if self._reduced_kinetic_energy is None:
                self._reduced_temperature = np.sum(np.square(self.velocities)) / ((self.number - 1) * self.dimension)
            else:
                self._reduced_temperature = self.get_reduced_kinetic_energy() / (.5 * (self.number - 1) * self.dimension)

        return self._reduced_temperature

    def get_temperature(self):
        return self.get_reduced_temperature() * self.epsilon / CONSTANTS.Boltzmann

    def get_reduced_potential_energy(self):
        if self._reduced_potential_energy is None:
            self._reduced_potential_energy = 2 * np.sum(self._distance_matrix_12 - self._distance_matrix_6)

        return self._reduced_potential_energy

    def get_potential_energy(self):
        return self.get_reduced_potential_energy() * self.epsilon

    def get_reduced_kinetic_energy(self):
        if self._reduced_kinetic_energy is None:
            if self._reduced_temperature is None:
                self._reduced_kinetic_energy = 0.5 * np.sum(np.square(self.velocities))
            else:
                self._reduced_kinetic_energy = self.get_reduced_temperature() * (.5 * (self.number - 1) * self.dimension)

        return self._reduced_kinetic_energy

    def get_kinetic_energy(self):
        return self.get_reduced_kinetic_energy() * self.epsilon

    def get_reduced_energy(self):
        return self.get_reduced_kinetic_energy() + self.get_reduced_potential_energy()

    def get_energy(self):
        return self.get_reduced_energy() * self.epsilon

    def get_reduced_volume(self):
        if self._reduced_volume is None:
            self._reduced_volume = np.power(self.reduced_length, self.dimension)

        return self._reduced_volume

    def get_volume(self):
        return self.get_reduced_volume() * np.power(self.sigma, self.dimension)

    def get_reduced_pressure(self):
        if self._reduced_pressure is None:
            self._reduced_pressure = (self.number * self.get_reduced_temperature() + (12 / self.dimension) * np.sum(
                (-2 * self._distance_matrix_12 + self._distance_matrix_6))) / self.get_reduced_volume()

        return self._reduced_pressure

    def get_pressure(self):
        return self.get_reduced_pressure() * (self.epsilon / (np.power(self.sigma, self.dimension)))

    def number_of_left_side_atoms(self):
        if self._number_of_left_side_atoms is None:
            self._number_of_left_side_atoms = np.count_nonzero(self.positions[0] < self.reduced_length_half)

        return self._number_of_left_side_atoms


class DataAnalysis:
    def __init__(self, file_base_name):
        self.file_base_name = file_base_name
        info = self._process_info_file()
        self.sigma = info['sigma']
        self.mass = info['mass']
        self.epsilon = info['epsilon']
        self.reduced_length = info['length']
        self.number = info['number']
        self.dimension = info['dimension']
        self.reduced_v_max = info['v_max']
        self.h = info['h']
        self.data_size = info['data_size']
        self.saving_period = info['saving_period']
        self.trajectory_struct = info['trajectory']
        self.data_struct = info['data']
        self._pos_vel_size = self.dimension * self.number * self.data_size
        self._traj_batch_size = len(self.trajectory_struct) * self._pos_vel_size
        self._data_batch_size = len(self.data_struct) * self.data_size
        self.sample_numbers = int(os.stat(f'data/{self.file_base_name}.data').st_size / self._data_batch_size)

        self._relaxation_index = None
        self._reduced_temperature = None
        self._reduced_temperature_error = None
        self._reduced_potential_energy = None
        self._reduced_potential_energy_error = None
        self._reduced_kinetic_energy = None
        self._reduced_kinetic_energy_error = None
        self._reduced_energy = None
        self._reduced_energy_error = None
        self._reduced_pressure = None
        self._reduced_pressure_error = None
        self._number_of_left_side_atoms = None
        self._number_of_left_side_atoms_error = None
        self._equilibrium_index = int(self.sample_numbers / 2)

    def _process_info_file(self):
        file = open(f'data/{self.file_base_name}.info', 'r')
        info = json.load(file)
        file.close()
        return info

    def auto_correlation(self, show=False):
        if show or self._relaxation_index is None:
            THERESHOLD = np.exp(-1)
            velocity_diff = self.trajectory_struct.index('velocities') * self._pos_vel_size
            temperature_diff = self.data_struct.index('temperature') * self.data_size
            trajectory_file = open(f'data/{self.file_base_name}.traj', 'rb')
            data_file = open(f'data/{self.file_base_name}.data', 'rb')

            steps = np.arange(0, int(self.sample_numbers / 5), dtype=int)
            C_v = np.zeros(len(steps))
            for step in steps:
                T_total = 0
                for sample in range(self.sample_numbers - step):
                    trajectory_file.seek(sample * self._traj_batch_size + velocity_diff)
                    data_file.seek(sample * self._data_batch_size + temperature_diff)
                    T_total += np.frombuffer(data_file.read(self.data_size))
                    velocities_1 = np.frombuffer(trajectory_file.read(self._pos_vel_size))
                    trajectory_file.seek((sample + step) * self._traj_batch_size + velocity_diff)
                    velocities_2 = np.frombuffer(trajectory_file.read(self._pos_vel_size))
                    C_v[step] += np.sum(velocities_1 * velocities_2)
                C_v[step] /= (T_total * self.dimension * (self.number - 1))
                if C_v[step] <= THERESHOLD and self._relaxation_index is None:
                    self._relaxation_index = step
                    if not show:
                        break

            if self._relaxation_index is None:
                self._relaxation_index = math.ceil(-steps[-1] / np.log(C_v[-1]))

            times = steps * (self.saving_period * self.h)

            if show:
                plt.plot(times, C_v)
                plt.xlabel(r'Time $(\times \tau)$')
                plt.ylabel(r'$C_v$')
                plt.savefig(f'images/{self.file_base_name}_c_v.jpg')
                plt.show()

            trajectory_file.close()
            data_file.close()

        return self._relaxation_index * self.saving_period * self.h

    def animate(self, interval=50, suffix=None):
        file = open(f'data/{self.file_base_name}.traj', 'rb')
        data = open(f'data/{self.file_base_name}.data', 'rb')
        positions_diff = self.trajectory_struct.index('positions') * self._pos_vel_size

        def update_frame(sample, frame, ax):
            file.seek(sample * self._traj_batch_size + positions_diff)
            positions = np.frombuffer(file.read(self._pos_vel_size))
            frame.set_data(positions[:self.number], positions[self.number:2 * self.number])
            # ax.set_title(r'Time $(\times \tau):$' + f'{round(sample * self.saving_period * self.h, 2)}')
            data.seek(sample * self._data_batch_size)
            ax.set_title(r'T :' + f'{round(np.frombuffer(data.read(self.data_size))[0] * self.epsilon / CONSTANTS.Boltzmann, 2)} K')
            return frame,

        fig = plt.figure()
        ax = plt.gca()
        frame, = plt.plot([], [], linestyle='', marker='o')
        plt.xlim(0, self.reduced_length)
        plt.ylim(0, self.reduced_length)
        ani = animation.FuncAnimation(fig, update_frame, self.sample_numbers, fargs=(frame, ax), interval=interval,
                                      blit=True)
        ani.save(f'animations/{self.file_base_name}{f"_{suffix}" if suffix else ""}.mp4')
        file.close()

    @staticmethod
    def _equilibrium_time(data):
        accuracy = 5
        micro_max_number = 200
        sampling_step = int(max(len(data) / (micro_max_number * accuracy), 1))
        reduced_data = data[::sampling_step]
        macro_step = int(max(len(reduced_data) / accuracy, 1))
        end_part = reduced_data[-macro_step:]
        end_mean = np.mean(end_part)
        end_std = np.std(end_part)
        diff = end_mean - reduced_data[0]
        threshold = end_mean - end_std * np.sign(diff)
        check_condition_func = (lambda x: x >= threshold) if diff > 0 else (lambda x: x <= threshold)

        for index in range(macro_step, len(reduced_data) - macro_step, macro_step):
            mean_value = np.mean(reduced_data[index - macro_step:index])
            if check_condition_func(mean_value):
                return int(index * sampling_step)

        return len(data)

    def _data_property_values(self, property_name):
        data = open(f'data/{self.file_base_name}.data', 'rb')
        property_diff = self.data_struct.index(property_name) * self.data_size
        values = np.zeros(self.sample_numbers)
        samples = range(self.sample_numbers)

        for sample in samples:
            data.seek(sample * self._data_batch_size + property_diff)
            values[sample] = np.frombuffer(data.read(self.data_size))

        data.close()

        return values

    def _calc_data_property_mean(self, property_name):
        values = self._data_property_values(property_name)
        valid_values = values[self._equilibrium_index:]
        mean_value = np.mean(valid_values)
        error = np.std(valid_values, ddof=1) / np.sqrt(len(values))
        return mean_value, error

    def get_reduced_temperature(self):
        if self._reduced_temperature is None:
            self._reduced_temperature, self._reduced_temperature_error = self._calc_data_property_mean('temperature')

        return np.array([self._reduced_temperature, self._reduced_temperature_error])

    def get_temperature(self):
        return self.get_reduced_temperature() * (self.epsilon / CONSTANTS.Boltzmann)

    def get_reduced_potential_energy(self):
        if self._reduced_potential_energy is None:
            self._reduced_potential_energy, self._reduced_potential_energy_error = self._calc_data_property_mean('potential_energy')

        return np.array([self._reduced_potential_energy, self._reduced_potential_energy_error])

    def get_potential_energy(self):
        return self.get_reduced_potential_energy() * self.epsilon

    def get_reduced_kinetic_energy(self):
        if self._reduced_kinetic_energy is None:
            self._reduced_kinetic_energy, self._reduced_kinetic_energy_error = self._calc_data_property_mean('kinetic_energy')

        return np.array([self._reduced_kinetic_energy, self._reduced_kinetic_energy_error])

    def get_kinetic_energy(self):
        return self.get_reduced_kinetic_energy() * self.epsilon

    def get_reduced_energy(self):
        if self._reduced_energy is None:
            self._reduced_energy, self._reduced_energy_error = self._calc_data_property_mean('energy')

        return np.array([self._reduced_energy, self._reduced_energy_error])

    def get_energy(self):
        return self.get_reduced_energy() * self.epsilon

    def get_reduced_pressure(self):
        if self._reduced_pressure is None:
            self._reduced_pressure, self._reduced_pressure_error = self._calc_data_property_mean('pressure')

        return np.array([self._reduced_pressure, self._reduced_pressure_error])

    def get_pressure(self):
        return self.get_reduced_pressure() * (self.epsilon / (np.power(self.sigma, self.dimension)))

    def get_number_of_left_side_atoms(self):
        if self._number_of_left_side_atoms is None:
            self._number_of_left_side_atoms, self._number_of_left_side_atoms_error = self._calc_data_property_mean('number_of_left_side_atoms')

        return np.array([self._number_of_left_side_atoms, self._number_of_left_side_atoms_error])

    def get_reduced_temperature_values(self):
        return self._data_property_values('temperature')

    def get_temperature_values(self):
        return self.get_reduced_temperature_values() * self.epsilon / CONSTANTS.Boltzmann

    def get_reduced_potential_energy_values(self):
        return self._data_property_values('potential_energy')

    def get_potential_energy_values(self):
        return self.get_reduced_potential_energy_values() * self.epsilon

    def get_reduced_kinetic_energy_values(self):
        return self._data_property_values('kinetic_energy')

    def get_kinetic_energy_values(self):
        return self.get_reduced_kinetic_energy_values() * self.epsilon

    def get_reduced_energy_values(self):
        return self._data_property_values('energy')

    def get_energy_values(self):
        return self.get_reduced_energy_values() * self.epsilon

    def get_reduced_pressure_values(self):
        return self._data_property_values('pressure')

    def get_pressure_values(self):
        return self.get_reduced_pressure_values() * (self.epsilon / (np.power(self.sigma, self.dimension)))

    def get_number_of_left_side_atoms_values(self):
        return self._data_property_values('number_of_left_side_atoms')

    def plot_energies(self):
        kinetic_energies = self.get_reduced_kinetic_energy_values()
        potential_energies = self.get_reduced_potential_energy_values()
        total_energies = self.get_reduced_energy_values()
        times = np.arange(self.sample_numbers) * self.saving_period * self.h

        plt.plot(times, kinetic_energies)
        plt.plot(times, potential_energies)
        plt.plot(times, total_energies)
        plt.xlabel(r'Time $(\times \tau)$')
        plt.ylabel(r'Energy $(\times \epsilon)$')
        plt.legend(['Kinetic Energy', 'Potential Energy', 'Total Energy'])
        plt.savefig(f'images/{self.file_base_name}_energies.jpg')
        plt.show()

    def plot_left_side_atom_numbers(self):
        numbers = self.get_number_of_left_side_atoms_values()
        times = np.arange(self.sample_numbers) * self.saving_period * self.h

        plt.axhline(.5, color='orange', linestyle='--')
        plt.plot(times, numbers / self.number)
        plt.xlabel(r'Time $(\times \tau)$')
        plt.ylabel(r'Fraction')
        plt.savefig(f'images/{self.file_base_name}_left_side_numbers.jpg')
        plt.show()

    def plot_temperature(self):
        temperatures = self.get_reduced_temperature_values()
        times = np.arange(self.sample_numbers) * self.saving_period * self.h

        plt.plot(times, temperatures)
        plt.xlabel(r'Time $(\times \tau)$')
        plt.ylabel(r'T $(\times \epsilon / k_B)$')
        plt.savefig(f'images/{self.file_base_name}_temperatures.jpg')
        plt.show()

    def plot_pressure(self):
        pressures = self.get_reduced_pressure_values()
        times = np.arange(self.sample_numbers) * self.saving_period * self.h

        plt.plot(times, pressures)
        plt.xlabel(r'Time $(\times \tau)$')
        plt.ylabel(r'P $(\times \epsilon / \sigma^' + f'{self.dimension}' + ')$')
        plt.savefig(f'images/{self.file_base_name}_pressures.jpg')
        plt.show()
