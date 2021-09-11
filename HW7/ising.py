import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as CONSTANTS


class Ising2D:
    _diff_energy_exp_map = {
        4: 1,
        8: 1,
    }

    def __init__(self, length, J):
        self.length = length
        self.size = length * length
        self.J = J
        self.reduced_T = np.Inf
        self.grid = np.random.choice((-1, 1), (length, length))
        self.reduced_energy = self._reduced_energy()
        self.total_magnetisation = self._total_magnetisation()

    @staticmethod
    def model_exp_func(t, a):
        return np.exp(-t / a)

    def _set_diff_energy_exp(self):
        self._diff_energy_exp_map = {
            4: np.exp(-4 / self.reduced_T),
            8: np.exp(-8 / self.reduced_T),
        }

    @staticmethod
    def _auto_correlation(data, diff_s):
        result = np.zeros(len(diff_s))

        for index, diff in enumerate(diff_s):
            result[index] = np.corrcoef(data[:len(data) - diff], data[diff:])[0, 1]
            if np.isnan(result[index]):
                result[index] = 1 if diff == 0 else 0

        return result

    @staticmethod
    def _exp_length(x_s, y_s, show=False):
        try:
            x_length = x_s[-1] - x_s[0]
            x_length = 1 if x_length == 0 else x_length
            fit_para, fit_error = curve_fit(Ising2D.model_exp_func, x_s / x_length, y_s, p0=(.5,))
            fit_para *= x_length
        except:
            fit_para = [0]

        if show:
            fit = np.zeros(len(x_s)) if fit_para[0] == 0 else np.exp(x_s / -fit_para[0])

            plt.plot(x_s, y_s, linestyle='', marker='.', markersize=5)
            plt.plot(x_s, fit, linestyle='--', color='g')
            plt.legend(['Samples', 'Fitted Curve'])
            plt.show()

        return fit_para[0]

    @staticmethod
    def _relaxation_time(data, show=False, max_diff=None):
        accuracy = 10
        max_number = 10 ** 4
        threshold = np.exp(-1)
        sampling_step = int(max(len(data) / (max_number * accuracy), 1))
        reduced_data = data[::sampling_step]
        center = 0
        max_diff = int(np.ceil(len(reduced_data) / accuracy)) if max_diff is None else max_diff
        while center < threshold:
            middle_index = int(max_diff / 2)
            center = Ising2D._auto_correlation(reduced_data, [middle_index])[0]
            max_diff = int(np.ceil(max_diff / 2))

        max_diff *= 2 * sampling_step
        samples_numbers = 50
        batch_size = int(max(max_diff / samples_numbers, 1))

        diff_s = np.arange(0, max_diff, batch_size)
        auto_correlations = Ising2D._auto_correlation(data, diff_s)

        return Ising2D._exp_length(diff_s, auto_correlations, show=show)

    @staticmethod
    def _mean_slope(data):
        accuracy = 2
        step = int(max(len(data) / accuracy, 1))
        return (np.mean(data[-step:]) - np.mean(data[:step])) / (len(data) - step)

    @staticmethod
    def _equilibrium_time(data):
        accuracy = 10
        micro_max_number = 100
        sampling_step = int(max(len(data) / (micro_max_number * np.power(accuracy, 2)), 1))
        reduced_data = data[::sampling_step]
        macro_step = int(max(len(reduced_data) / accuracy, 1))
        micro_step = int(max(macro_step / accuracy, 1))
        end_part = reduced_data[-macro_step:]
        end_mean = np.mean(end_part)
        end_std = np.std(end_part)
        diff = end_mean - reduced_data[0]
        threshold = end_mean - end_std * np.sign(diff)
        check_condition_func = (lambda x: x >= threshold) if diff > 0 else (lambda x: x <= threshold)

        # slope_threshold = .2
        # y_diff = np.max(reduced_data) - np.min(reduced_data)
        # y_diff = 1 if y_diff == 0 else y_diff
        # x_diff = len(reduced_data)
        # slope = Ising2D._mean_slope(end_part) / y_diff * x_diff
        # print(f'\nslope: {slope}')
        # if np.abs(slope) > slope_threshold:
        #     return None

        total = np.sum(reduced_data[:macro_step])
        for index in range(macro_step, len(reduced_data) - macro_step, micro_step):
            if check_condition_func(total / macro_step):
                return int((index - macro_step / 2) * sampling_step)
            else:
                total += np.sum(reduced_data[index:index + micro_step]) - np.sum(
                    reduced_data[index - macro_step:index - macro_step + micro_step])

        return None

    def spatial_correlation_length(self, show=False):
        grid_mean_square = np.mean(self.grid) ** 2
        grid_var = np.var(self.grid)
        threshold = 0.1 * grid_var + grid_mean_square
        max_diff = int(np.ceil(self.length / 2))
        correlations = np.zeros(max_diff)
        diff = 0

        while diff < max_diff:
            grid_rolled = np.roll(self.grid, -diff, axis=1)
            correlations[diff] = np.mean(self.grid * grid_rolled)
            if correlations[diff] < threshold:
                diff += 1
                break
            else:
                diff += 1

        correlations = correlations[:diff]

        if grid_var != 0:
            correlations = (correlations - grid_mean_square) / grid_var
            return Ising2D._exp_length(np.arange(len(correlations)), correlations, show=show)
        else:
            return 0

    def _monte_carlo_move(self):
        acceptance_count = 0
        random_indexes = np.random.randint(0, self.length, (self.size, 2))

        for MP_move in range(self.size):
            row, col = random_indexes[MP_move]
            diff_energy = self._diff_energy(row, col)

            if diff_energy <= 0 or np.random.rand() < self._diff_energy_exp(diff_energy):
                self.grid[row, col] = -self.grid[row, col]
                self.reduced_energy += diff_energy
                acceptance_count += 1

        return acceptance_count

    def render(self, T, sample_numbers, reduced_unit=True):
        self.reduced_T = T if reduced_unit else T / (self.J / CONSTANTS.Boltzmann)
        self._set_diff_energy_exp()
        acceptance_count = 0
        relaxation_time = 100
        reduced_energies = np.zeros(relaxation_time)
        MP_step = 0

        while True:
            while MP_step < relaxation_time:
                self._monte_carlo_move()
                reduced_energies[MP_step] = self.reduced_energy
                MP_step += 1

            new_relaxation_time = int(np.ceil(Ising2D._relaxation_time(reduced_energies)))
            if new_relaxation_time > relaxation_time:
                reduced_energies = np.append(reduced_energies, np.zeros(new_relaxation_time - relaxation_time))
                relaxation_time = new_relaxation_time
            else:
                relaxation_time = new_relaxation_time
                break
        relaxation_time *= 2

        count = 0
        total_count = 0
        is_equilibrium_found = False
        reduced_energies = np.zeros(sample_numbers)
        correlation_length_s = np.zeros(sample_numbers)
        total_magnetisation_s = np.zeros(sample_numbers)
        sample_limit = sample_numbers
        while True:
            while count < sample_limit:
                print(f'\rSample {count + 1} from {sample_limit}', end='')
                for micro_step in range(relaxation_time):
                    acceptance_count += self._monte_carlo_move()

                reduced_energies[count] = self.reduced_energy
                self.total_magnetisation = self._total_magnetisation()
                total_magnetisation_s[count] = self.total_magnetisation
                correlation_length_s[count] = self.spatial_correlation_length()
                count += 1
                total_count += 1

            if is_equilibrium_found:
                break
            else:
                equilibrium_time = Ising2D._equilibrium_time(reduced_energies)
                if equilibrium_time is None:
                    count = 0
                else:
                    count = 0
                    sample_limit = equilibrium_time
                    is_equilibrium_found = True

        print('\rCompleted!')

        acceptance_rate = acceptance_count / (total_count * self.size)
        render_data = RenderData(
            self.length, self.size, self.J, self.reduced_T, sample_numbers, acceptance_rate, reduced_energies,
            total_magnetisation_s, correlation_length_s
        )
        return render_data

    def _reduced_energy(self):
        return -np.sum(self.grid * (np.roll(self.grid, 1, axis=1) + np.roll(self.grid, 1, axis=0)))

    def _total_magnetisation(self):
        return np.abs(np.sum(self.grid))

    def _diff_energy(self, row, col):
        return 2 * self.grid[row][col] * (
                self.grid[(row + 1) % self.length][col] + self.grid[row - 1][col] +
                self.grid[row][(col + 1) % self.length] + self.grid[row][col - 1]
        )

    def _diff_energy_exp(self, diff_energy):
        return self._diff_energy_exp_map.get(diff_energy)

    def show(self):
        fig, ax = plt.subplots()
        plt.tick_params(axis='both', which='both', bottom=False, left=False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_aspect('equal', 'box')
        plt.pcolormesh(self.grid, cmap='CMRmap_r')
        plt.savefig('ising_' + str(self.length) + '_' + str(self.reduced_T) + '.png', pad_inches=0, bbox_inches='tight')
        plt.show()


class RenderData:
    def __init__(self, length, size, J, reduced_T, sample_numbers, acceptance_rate, reduced_energies,
                 total_magnetisation_s, correlation_length_s):
        self.length = length
        self.size = size
        self.J = J
        self.reduced_T = reduced_T
        self.sample_numbers = sample_numbers
        self.acceptance_rate = acceptance_rate
        self.reduced_energies = reduced_energies
        self.total_magnetisation_s = total_magnetisation_s
        self.correlation_length_s = correlation_length_s
        self._reduced_energy = None
        self._reduced_energy_error = None
        self._magnetisation = None
        self._magnetisation_error = None
        self._correlation_length = None
        self._correlation_length_error = None
        self._reduced_heat_capacity = None
        self._reduced_heat_capacity_error = None
        self._reduced_susceptibility = None
        self._reduced_susceptibility_error = None

    def save(self):
        data = {
            'length': self.length,
            'size': self.size,
            'J': self.J,
            'T': self.reduced_T,
            'sample_numbers': self.sample_numbers,
            'reduced_energies': self.reduced_energies,
            'total_magnetisation_s': self.total_magnetisation_s,
            'correlation_length_s': self.correlation_length_s,
            'acceptance_rate': self.acceptance_rate,
            '_reduced_energy': self._reduced_energy,
            '_reduced_energy_error': self._reduced_energy_error,
            '_magnetisation': self._magnetisation,
            '_magnetisation_error': self._magnetisation_error,
            '_correlation_length': self._correlation_length,
            '_correlation_length_error': self._correlation_length_error,
            '_reduced_heat_capacity': self._reduced_heat_capacity,
            '_reduced_heat_capacity_error': self._reduced_heat_capacity_error,
            '_reduced_susceptibility': self._reduced_susceptibility,
            '_reduced_susceptibility_error': self._reduced_susceptibility_error,
        }

        np.save("ising_render_data_" + str(self.length) + '_' + str(self.reduced_T) + '_' + str(self.sample_numbers), data)

    @staticmethod
    def load(file_name):
        data = np.load(file_name, allow_pickle=True).tolist()
        render_data = RenderData(
            data['length'],
            data['size'],
            data['J'],
            data['T'],
            data['sample_numbers'],
            data['acceptance_rate'],
            data['reduced_energies'],
            data['total_magnetisation_s'],
            data['correlation_length_s'],
        )

        render_data._reduced_energy = data['_reduced_energy']
        render_data._reduced_energy_error = data['_reduced_energy_error']
        render_data._magnetisation = data['_magnetisation']
        render_data._magnetisation_error = data['_magnetisation_error']
        render_data._correlation_length = data['_correlation_length']
        render_data._correlation_length_error = data['_correlation_length_error']
        render_data._reduced_heat_capacity = data['_reduced_heat_capacity']
        render_data._reduced_heat_capacity_error = data['_reduced_heat_capacity_error']
        render_data._reduced_susceptibility = data['_reduced_susceptibility']
        render_data._reduced_susceptibility_error = data['_reduced_susceptibility_error']

        return render_data

    def get_reduced_energy(self):
        if self._reduced_energy is None:
            self._reduced_energy = np.mean(self.reduced_energies)

        return self._reduced_energy

    def get_reduced_energy_error(self):
        if self._reduced_energy_error is None:
            self._reduced_energy_error = np.std(self.reduced_energies, ddof=1) / np.sqrt(self.sample_numbers)

        return self._reduced_energy_error

    def get_reduced_heat_capacity(self):
        if self._reduced_heat_capacity is None:
            self._reduced_heat_capacity = np.var(self.reduced_energies) / np.power(self.reduced_T, 2)

        return self._reduced_heat_capacity

    def get_reduced_heat_capacity_error(self):
        if self._reduced_heat_capacity_error is None:
            self._reduced_heat_capacity_error = RenderData.calc_bootstrap_error(self.reduced_energies, np.var) / np.power(self.reduced_T, 2)

        return self._reduced_heat_capacity_error

    def get_reduced_susceptibility(self):
        if self._reduced_susceptibility is None:
            self._reduced_susceptibility = np.var(self.total_magnetisation_s) / self.reduced_T

        return self._reduced_susceptibility

    def get_reduced_susceptibility_error(self):
        if self._reduced_susceptibility_error is None:
            self._reduced_susceptibility_error = RenderData.calc_bootstrap_error(self.total_magnetisation_s, np.var) / self.reduced_T

        return self._reduced_susceptibility_error

    def get_magnetisation(self):
        if self._magnetisation is None:
            self._magnetisation = np.mean(self.total_magnetisation_s)

        return self._magnetisation

    def get_magnetisation_error(self):
        if self._magnetisation_error is None:
            self._magnetisation_error = np.std(self.total_magnetisation_s, ddof=1) / np.sqrt(self.sample_numbers)

        return self._magnetisation_error

    def get_correlation_length(self):
        if self._correlation_length is None:
            self._correlation_length = np.mean(self.correlation_length_s)

        return self._correlation_length

    def get_correlation_length_error(self):
        if self._correlation_length_error is None:
            self._correlation_length_error = np.std(self.correlation_length_s, ddof=1) / np.sqrt(self.sample_numbers)

        return self._correlation_length_error

    def get_energy(self):
        return self.get_reduced_energy() * self.J

    def get_energy_error(self):
        return self.get_reduced_energy_error() * self.J

    def get_heat_capacity(self):
        return self.get_reduced_heat_capacity() * CONSTANTS.Boltzmann

    def get_heat_capacity_error(self):
        return self.get_reduced_heat_capacity_error() * CONSTANTS.Boltzmann

    def get_susceptibility(self):
        return self.get_reduced_susceptibility() / self.J

    def get_susceptibility_error(self):
        return self.get_reduced_susceptibility_error() / self.J

    @staticmethod
    def calc_bootstrap_error(data, func):
        ensemble_numbers = 100
        ensemble_values = np.zeros(ensemble_numbers)

        for ensemble_number in range(ensemble_numbers):
            random_numbers = np.random.randint(0, len(data), len(data))
            ensemble_values[ensemble_number] = func(data[random_numbers])

        return np.std(ensemble_values)
