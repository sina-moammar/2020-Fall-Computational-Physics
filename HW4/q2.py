import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class RandomWalk:
    def __init__(self):
        self.positions = []
        self.steps = []
        self.time = 0
        self.probability = .5

    def render(self, time, p=.5):
        self.time = time
        self.probability = p
        self.positions = np.zeros(time + 1, dtype=np.int)
        self.steps = np.random.choice([1, -1], time, p=(p, 1 - p))

    def calc_positions(self):
        for time, step in enumerate(self.steps):
            self.positions[time + 1] = self.positions[time] + step

        return self.positions


def model_line_func(t, a, b):
    return a * t + b


def calc_mean_positions(time=100, step=10, p_s=(0.2, 0.5, 0.8), sample_numbers=100000, file_name=None):
    if file_name is None:
        times = np.arange(0, time, step, dtype=int)
        model = RandomWalk()
        sample_positions_ensemble = np.zeros((len(p_s), sample_numbers, len(times)), dtype=np.int8)

        for (p_i, p) in enumerate(p_s):
            print(str(p) + ':')
            for sample_number in range(sample_numbers):
                print('\r\t' + str(sample_number), end='')
                model.render(time, p)
                sample_positions_ensemble[p_i][sample_number] = model.calc_positions()[times]

            print(end='\n')

        data = {
            'times': times,
            'step': step,
            'p_s': p_s,
            'positions': sample_positions_ensemble,
            'numbers': sample_numbers
        }
        np.save("data/q2_" + str(p_s) + "_" + str(time) + '_' + str(step) + "_" + str(sample_numbers) + "_ensemble", data)
    else:
        data = np.load('data/q2_' + file_name + '_ensemble.npy', allow_pickle=True).tolist()
        times = data['times']
        step = data['step']
        p_s = data['p_s']
        sample_positions_ensemble = data['positions']
        sample_numbers = data['numbers']

    positions_avg = np.average(sample_positions_ensemble, axis=1)
    positions_var = np.var(sample_positions_ensemble, axis=1)

    for (p_i, p) in enumerate(p_s):
        position_avg = positions_avg[p_i]
        position_var = positions_var[p_i]
        position_fit_para, position_fit_error = curve_fit(model_line_func, times, position_avg)
        position_fit_error = np.diag(position_fit_error)
        position_fit = position_fit_para[1] + position_fit_para[0] * times

        var_fit_para, var_fit_error = curve_fit(model_line_func, times, position_var)
        var_fit_error = np.diag(var_fit_error)
        var_fit = var_fit_para[1] + var_fit_para[0] * times

        plt.plot(times, position_fit, linestyle='--', color='g')
        plt.plot(times, position_avg, linestyle='', marker='.', markersize=5)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$<x(t)>$')
        plt.legend(['Fitted line', 'p = ' + str(p)])
        plt.savefig(
            "images/q2_x_" + str(p) + "_" + str(time) + "_" + str(step) + "_" + str(sample_numbers) + '.png')
        plt.show()

        plt.plot(times, var_fit, linestyle='--', color='g')
        plt.plot(times, position_var, linestyle='', marker='.', markersize=5)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\sigma^2(t)$')
        plt.legend(['Fitted line', 'p = ' + str(p)])
        plt.savefig(
            "images/q2_var_" + str(p) + "_" + str(time) + "_" + str(step) + "_" + str(sample_numbers) + '.png')
        plt.show()

        print('p = ' + str(p) + ':')
        print('\t <x(t)> slope: ' + str(position_fit_para[0]) + ' ± ' + str(position_fit_error[0]))
        print('\t var(x) slope: ' + str(var_fit_para[0]) + ' ± ' + str(var_fit_error[0]))


calc_mean_positions(100, 10, (0.2, 0.5, 0.8), 100000)
