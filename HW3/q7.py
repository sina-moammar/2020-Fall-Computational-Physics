import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def model_line_func(t, a, b):
    return a * t + b


class Percolation:
    def __init__(self, length):
        self.length = length
        self.grid = np.zeros((length + 2, length + 2), dtype=int)
        self.p = 0
        self.active_cells = []
        self.size = 0

    def load(self, file_name):
        data = np.load('q7_' + file_name + '_ensemble.npy', allow_pickle=True).tolist()
        self.length = data['length']
        self.grid = data['grid']
        self.p = data['p']
        self.size = data['size']
        self.active_cells = data['active']

    def render(self, p, save=False):
        self.p = p
        self.active_cells = np.zeros(self.length * self.length, dtype=np.complex64)
        self.grid = np.zeros((self.length + 2, self.length + 2), dtype=np.int8)
        self.grid[1:self.length + 1, 1:self.length + 1] = np.ones((self.length, self.length), dtype=np.short) * -1
        seed_index = int(self.length / 2) + 1
        active_cells = [seed_index + seed_index * 1j]
        self.grid[seed_index][seed_index] = 1
        self.active_cells[0] = seed_index + seed_index * 1j
        self.size = 1

        while len(active_cells) > 0:
            neighbors = []
            for i in range(len(active_cells)):
                row = int(active_cells[i].imag)
                col = int(active_cells[i].real)
                if self.grid[row - 1][col] == -1:
                    neighbors.append(col + (row - 1) * 1j)
                if self.grid[row + 1][col] == -1:
                    neighbors.append(col + (row + 1) * 1j)
                if self.grid[row][col - 1] == -1:
                    neighbors.append(col - 1 + row * 1j)
                if self.grid[row][col + 1] == -1:
                    neighbors.append(col + 1 + row * 1j)

            neighbors = np.array(neighbors, dtype=np.complex64)
            neighbors = np.unique(neighbors)
            # print(neighbors)
            probability = np.random.rand(len(neighbors))
            active_cells_indexes = probability < p
            closed_cells_indexes = np.bitwise_not(active_cells_indexes)
            new_active = neighbors[active_cells_indexes]
            new_closed = neighbors[closed_cells_indexes]

            if len(new_closed) > 0:
                self.grid[new_closed.imag.astype(int), new_closed.real.astype(int)] = 0
            if len(new_active) > 0:
                self.grid[new_active.imag.astype(int), new_active.real.astype(int)] = 1

            active_cells = new_active
            new_cells_length = len(new_active)
            self.active_cells[self.size:self.size + new_cells_length] = new_active
            self.size += new_cells_length

        self.active_cells = self.active_cells[:self.size]

        if save:
            data = {
                'length': self.length,
                'grid': self.grid,
                'p': self.p,
                'active': self.active_cells,
                'size': self.size
            }
            np.save("data/q7_" + str(self.length) + "_" + str(p) + "_ensemble", data)

    def correlation_length(self):
        return np.std(self.active_cells)

    def show(self):
        image = self.grid[1:-1, 1:-1]
        fig, ax = plt.subplots()
        plt.tick_params(axis='both', which='both', bottom=False, left=False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_aspect('equal', 'box')
        plt.pcolormesh(image, cmap='CMRmap_r')
        plt.savefig('images/q7_' + str(self.length) + '_' + str(self.p) + '.png', pad_inches=0, bbox_inches='tight')
        plt.show()


def calc_size_radius(p_start=.5, step_size=.01, p_end=.6, length=10000, sample_numbers=100, file_name=None):
    if file_name is None:
        p_s = np.arange(p_start, p_end, step_size)
        model = Percolation(length)
        correlation_length_s_ensemble = np.zeros((len(p_s), sample_numbers))
        size_s_ensemble = np.zeros((len(p_s), sample_numbers))
        for (p_i, p) in enumerate(p_s):
            print(str(p) + ": ")
            for sample in range(sample_numbers):
                print("   " + str(sample))
                model.render(p)
                correlation_length_s_ensemble[p_i][sample] = model.correlation_length()
                size_s_ensemble[p_i][sample] = model.size

        data = {
            'p_s': p_s,
            'length': length,
            'numbers': sample_numbers,
            'radii': correlation_length_s_ensemble,
            'sizes': size_s_ensemble
        }
        np.save("data/q7_" + str(p_start) + "_" + str(step_size) + "_" + str(p_end) + "_" + str(sample_numbers) + "_" + str(length) + "_ensemble", data)
    else:
        data = np.load('q7_' + file_name + '_ensemble.npy', allow_pickle=True).tolist()
        p_s = data['p_s']
        length = data['length']
        sample_numbers = data['numbers']
        correlation_length_s_ensemble = data['radii']
        size_s_ensemble = data['sizes']

    correlation_radii_avg = np.average(correlation_length_s_ensemble, axis=1)
    correlation_radii_std = np.std(correlation_length_s_ensemble, ddof=1, axis=1) / np.sqrt(sample_numbers)
    size_s_avg = np.average(size_s_ensemble, axis=1)
    size_s_std = np.std(size_s_ensemble, ddof=1, axis=1) / np.sqrt(sample_numbers)

    size_s_avg_ln = np.log(size_s_avg)
    correlation_radii_avg_ln = np.log(correlation_radii_avg)
    size_fit_para, size_fit_error = curve_fit(model_line_func, correlation_radii_avg_ln, size_s_avg_ln)
    size_fit_error = np.diag(size_fit_error)
    size_fit = np.exp(size_fit_para[1] + correlation_radii_avg_ln * size_fit_para[0])

    plt.errorbar(p_s, correlation_radii_avg, yerr=correlation_radii_std, linestyle='--', elinewidth=.8)
    plt.xlabel(r'$p$')
    plt.ylabel(r'$\xi$')
    plt.savefig("images/q7_radii_" + str(p_start) + "_" + str(step_size) + "_" + str(p_end) + "_" + str(sample_numbers) + "_" +
                str(length) + '.png')
    plt.show()

    plt.errorbar(p_s, size_s_avg, yerr=size_s_std, linestyle='--', elinewidth=.8)
    plt.xlabel(r'$p$')
    plt.ylabel(r'$S$')
    plt.savefig("images/q7_sizes_" + str(p_start) + "_" + str(step_size) + "_" + str(p_end) + "_" + str(sample_numbers) + "_" +
                str(length) + '.png')
    plt.show()

    plt.errorbar(correlation_radii_avg, size_s_avg, xerr=correlation_radii_std, yerr=size_s_std, linestyle='--',
                 elinewidth=.8)
    plt.plot(correlation_radii_avg, size_fit, linestyle='--')
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$S$')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig("images/q7_size_radii_" + str(p_start) + "_" + str(step_size) + "_" + str(p_end) + "_" + str(sample_numbers) +
                "_" + str(length) + '.png')
    plt.show()

    print('Slope: ' + str(size_fit_para[0]) + ' ± ' + str(size_fit_error[0]))


model = Percolation(100)
model.render(0.5)
model.show()
model.render(0.55)
model.show()
model.render(0.59)
model.show()

calc_size_radius(0.5, 0.01, 0.6, 10000, 1000)
calc_size_radius(0.5, 0.01, 0.58, 10000, 1000)
