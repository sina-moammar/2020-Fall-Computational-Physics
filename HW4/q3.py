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
            return

        random_mean_size = 2 * x_0 * (self.length - 1 - x_0)
        position = x_0
        while True:
            steps = np.random.choice([1, -1], random_mean_size)
            for step in steps:
                position += step
                self.life_time += 1
                if not(0 < position < (self.length - 1)):
                    return


def model_parabolic_func(t, a, b, c):
    return a * ((t - b) ** 2) + c


def calc_mean_life_time(length=20, step=1, sample_numbers=100000, file_name=None):
    if file_name is None:
        x_s = np.arange(0, length, step, dtype=int)
        model = RandomWalkWithAbsorbingBarrier(length)
        sample_life_times_ensemble = np.zeros((len(x_s), sample_numbers), dtype=np.int)

        for (x_i, x) in enumerate(x_s):
            print(str(x) + ':')
            for sample_number in range(sample_numbers):
                print('\r\t' + str(sample_number), end="")
                model.render(x)
                sample_life_times_ensemble[x_i][sample_number] = model.life_time

            print(end='\n')

        data = {
            'x_s': x_s,
            'length': length,
            'step': step,
            'life_times': sample_life_times_ensemble,
            'numbers': sample_numbers
        }
        np.save("data/q3_" + str(length) + '_' + str(step) + "_" + str(sample_numbers) + "_ensemble", data)
    else:
        data = np.load('data/q3_' + file_name + '_ensemble.npy', allow_pickle=True).tolist()
        x_s = data['x_s']
        length = data['length']
        step = data['step']
        sample_life_times_ensemble = data['life_times']
        sample_numbers = data['numbers']

    life_time_avg = np.average(sample_life_times_ensemble, axis=1)

    life_time_fit_para, life_time_fit_error = curve_fit(model_parabolic_func, x_s, life_time_avg)
    life_time_fit_error = np.diag(life_time_fit_error)
    life_time_fit = life_time_fit_para[0] * ((x_s - life_time_fit_para[1]) ** 2) + life_time_fit_para[2]

    plt.plot(x_s, life_time_fit, linestyle='--', color='g')
    plt.plot(x_s, life_time_avg, linestyle='', marker='.', markersize=5)
    plt.xlabel(r'$x_0$')
    plt.ylabel(r'<Life Time>')
    plt.legend(['Fitted line', 'Samples'])
    plt.savefig(
        "images/q3_life_time_" + str(length) + '_' + str(step) + "_" + str(sample_numbers) + '.png')
    plt.show()

    print('<Life_Time(x_0)> = ' + str(life_time_fit_para[0]) + ' * (x_0 - ' + str(life_time_fit_para[1]) + ') ^ 2 + '+
          str(life_time_fit_para[2]))


start = time.time()
calc_mean_life_time(20, 1, 100000)
end = time.time()

print('Time: ' + str(end - start))
