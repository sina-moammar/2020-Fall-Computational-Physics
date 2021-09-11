import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time


class RandomWalkWithAbsorbingBarrier:
    def __init__(self, length):
        self.length = length
        self.x_initial = 0
        self.life_time = 0

    def render(self, x_0):
        self.x_initial = x_0
        self.life_time = 0
        if not(0 < x_0 < (self.length - 1)):
            self.life_time = 0
            return

        probabilities = np.zeros(self.length)
        probabilities[int(x_0)] = 1
        death_probability_threshold = 1 - 10**(-14)
        total_death_probability = 0
        time = 0

        while total_death_probability < death_probability_threshold:
            time += 1
            each_side_probabilities = probabilities / 2
            probabilities[1:-1] = each_side_probabilities[2:] + each_side_probabilities[:-2]
            death_probability = each_side_probabilities[1] + each_side_probabilities[-2]
            total_death_probability += death_probability
            self.life_time += death_probability * time


def model_parabolic_func(t, a, b, c):
    return a * ((t - b) ** 2) + c


def calc_mean_life_time(length=20, step=1, file_name=None):
    if file_name is None:
        x_s = np.arange(0, length, step, dtype=int)
        model = RandomWalkWithAbsorbingBarrier(length)
        sample_life_times = np.zeros(len(x_s))

        for (x_i, x) in enumerate(x_s):
            print(x)
            model.render(x)
            sample_life_times[x_i] = model.life_time

        data = {
            'x_s': x_s,
            'length': length,
            'step': step,
            'life_times': sample_life_times,
        }
        np.save("data/q4_" + str(length) + '_' + str(step) + "_ensemble", data)
    else:
        data = np.load('data/q4_' + file_name + '_ensemble.npy', allow_pickle=True).tolist()
        x_s = data['x_s']
        length = data['length']
        step = data['step']
        sample_life_times = data['life_times']

    life_time_fit_para, life_time_fit_error = curve_fit(model_parabolic_func, x_s, sample_life_times)
    life_time_fit_error = np.diag(life_time_fit_error)
    life_time_fit = life_time_fit_para[0] * ((x_s - life_time_fit_para[1]) ** 2) + life_time_fit_para[2]

    plt.plot(x_s, life_time_fit, linestyle='--', color='g')
    plt.plot(x_s, sample_life_times, linestyle='', marker='.', markersize=5)
    plt.xlabel(r'$x_0$')
    plt.ylabel(r'<Life Time>')
    plt.legend(['Fitted line', 'Samples'])
    plt.savefig(
        "images/q4_life_time_" + str(length) + '_' + str(step) + "_" + '.png')
    plt.show()

    print('<Life_Time(x_0)> = ' + str(life_time_fit_para[0]) + ' * (x_0 - ' + str(life_time_fit_para[1]) + ') ^ 2 + '+
          str(life_time_fit_para[2]))


start = time.time()
calc_mean_life_time(20, 1)
end = time.time()

print('Time: ' + str(end - start))
