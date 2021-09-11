import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class RandomWalk2D:
    def __init__(self):
        self.positions = []
        self.steps = []
        self.time = 0

    def render(self, time, step_length=1):
        self.time = time
        self.positions = np.zeros(time + 1, dtype=np.complex64)
        self.steps = np.random.choice([1, -1, 1j, -1j], time) * step_length

    def calc_positions(self):
        for time, step in enumerate(self.steps):
            self.positions[time + 1] = self.positions[time] + step

        return self.positions


def model_line_func(t, a, b):
    return a * t + b


def calc_var(time=100, step=10, l_s=(0.5, 1, 2), sample_numbers=100000, file_name=None):
    if file_name is None:
        times = np.arange(0, time, step, dtype=int)
        model = RandomWalk2D()
        sample_positions_ensemble = np.zeros((len(l_s), sample_numbers, len(times)), dtype=np.complex64)

        for (l_i, l) in enumerate(l_s):
            print(str(l) + ':')
            for sample_number in range(sample_numbers):
                print('\r \t' + str(sample_number), end='')
                model.render(time, l)
                sample_positions_ensemble[l_i][sample_number] = model.calc_positions()[times]

            print(end='\n')

        data = {
            'times': times,
            'time': time,
            'step': step,
            'l_s': l_s,
            'positions': sample_positions_ensemble,
            'numbers': sample_numbers
        }
        np.save("data/q5_" + str(l_s) + "_" + str(time) + "_" + str(step) + "_" + str(sample_numbers) + "_ensemble", data)
    else:
        data = np.load('data/q5_' + file_name + '_ensemble.npy', allow_pickle=True).tolist()
        times = data['times']
        time = data['time']
        step = data['step']
        l_s = data['l_s']
        sample_positions_ensemble = data['positions']
        sample_numbers = data['numbers']

    positions_var = np.var(sample_positions_ensemble, axis=1)

    for (l_i, l) in enumerate(l_s):
        position_var = positions_var[l_i]

        var_fit_para, var_fit_error = curve_fit(model_line_func, times, position_var)
        var_fit_error = np.diag(var_fit_error)
        var_fit = var_fit_para[1] + var_fit_para[0] * times

        plt.plot(times, var_fit, linestyle='--', color='g')
        plt.plot(times, position_var, linestyle='', marker='.', markersize=5)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$<r^2(t)>$')
        plt.legend(['Fitted line', 'l = ' + str(l)])
        plt.savefig(
            "images/q5_var_" + str(l) + "_" + str(time) + "_" + str(step) + "_" + str(sample_numbers) + '.png')
        plt.show()

        print('p = ' + str(l) + ':')
        print('\t var(r) slope: ' + str(var_fit_para[0]) + ' ± ' + str(var_fit_error[0]))


calc_var(100, 10, (0.5, 1, 2), 100000)
