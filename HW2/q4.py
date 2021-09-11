import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class RandomBallisticDeposition:
    def __init__(self, length=0):
        self.length = length
        self.time = 0
        self.samples_times = []
        self.samples_heights = []

    def render(self, time=None, samples_times=None, save=False):
        if time is None:
            self.time = samples_times[-1]
        else:
            self.time = time
            step_size = 10 * self.length
            samples_times = np.arange(step_size, time, step_size)
            samples_times = np.append(samples_times, time)
        self.samples_times = samples_times
        self.samples_heights = np.zeros((len(samples_times), self.length), dtype=int)

        prev_time = 0
        for (i, current_time) in enumerate(samples_times):
            randoms = np.random.randint(0, self.length, current_time - prev_time, dtype=int)
            indexes, counts = np.unique(randoms, return_counts=True)
            self.samples_heights[i][indexes] = self.samples_heights[i - 1][indexes] + counts
            prev_time = current_time

        if save:
            data = {
                'times': self.samples_times,
                'heights': self.samples_heights,
                'time': self.time,
                'length': self.length
            }
            np.save("data/q4_" + str(self.length) + '_' + str(self.time), data)

    def show(self):
        prev_height = np.zeros(self.length)
        x = np.arange(0, self.length)
        is_blue = True
        for heights_sample in self.samples_heights:
            plt.bar(x, heights_sample - prev_height, width=1, bottom=prev_height, color=('b' if is_blue else 'r'))
            prev_height = heights_sample
            is_blue = not is_blue

        plt.tick_params(axis='both', which='both', bottom=False, left=False)
        plt.gca().set_yticklabels([])
        plt.gca().set_xticklabels([])
        plt.axis('off')
        plt.axis('equal')
        plt.savefig('images/q4_' + str(self.length) + '_' + str(self.time) + '.png', pad_inches=0, bbox_inches='tight')
        plt.show()

    def load(self, file_name):
        data = np.load('data/' + file_name, allow_pickle=True).tolist()
        self.length = data['length']
        self.time = data['time']
        self.samples_times = data['times']
        self.samples_heights = data['heights']

    def make_ensemble(self, sample_times, num, file_name=None):
        ensemble_avg_s = np.zeros((num, len(sample_times)))
        ensemble_std_s = np.zeros((num, len(sample_times)))
        for t in range(num):
            self.render(samples_times=sample_times)
            ensemble_avg_s[t] = np.average(self.samples_heights, axis=1)
            ensemble_std_s[t] = np.std(self.samples_heights, axis=1)

        data = {
            'avgs': ensemble_avg_s,
            'stds': ensemble_std_s,
            'times': sample_times
        }
        if file_name is not None:
            np.save("data/q4_" + file_name + "_ensemble", data)

    def analyse(self, file_name):
        data = np.load('data/q4_' + file_name + '_ensemble.npy', allow_pickle=True).tolist()
        ensemble_avg_s = data['avgs']
        ensemble_std_s = data['stds']
        sample_times = data['times']
        sample_times_ln = np.log(sample_times)

        avg = np.average(ensemble_avg_s, axis=0)
        avg_ln = np.log(avg)
        avg_error = np.std(ensemble_avg_s, axis=0)
        avg_error[avg_error == 0] = 10 ** (-16)
        avg_error_ln = avg_error / avg
        std = np.average(ensemble_std_s, axis=0)
        std_ln = np.log(std)
        std_error = np.std(ensemble_std_s, axis=0)
        std_error[std_error == 0] = 10 ** (-16)
        std_error_ln = std_error / std

        def model_line_func(t, a, b):
            return b + a * t

        valid_indexes = [10, None]
        avg_fit_para, avg_fit_error = curve_fit(model_line_func, sample_times_ln[valid_indexes[0]:valid_indexes[1]],
                                                avg_ln[valid_indexes[0]:valid_indexes[1]],
                                                sigma=avg_error_ln[valid_indexes[0]:valid_indexes[1]])
        avg_fit_error = np.diag(avg_fit_error)
        avg_fit = np.exp(avg_fit_para[1] + sample_times_ln * avg_fit_para[0])

        std_fit_para, std_fit_error = curve_fit(model_line_func, sample_times_ln[valid_indexes[0]:valid_indexes[1]],
                                                std_ln[valid_indexes[0]:valid_indexes[1]],
                                                sigma=std_error_ln[valid_indexes[0]:valid_indexes[1]])
        std_fit_error = np.diag(std_fit_error)
        std_fit = np.exp(std_fit_para[1] + sample_times_ln * std_fit_para[0])

        plt.errorbar(sample_times, avg, yerr=avg_error, linestyle='', marker='.', ecolor='r', markersize=5)
        plt.plot(sample_times, avg_fit, linestyle='--', color='g')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Time')
        plt.ylabel('Average Height')
        plt.legend(['Fitted line', 'Samples'])
        plt.savefig('images/q4_' + file_name + '_avg.png')
        plt.show()

        plt.errorbar(sample_times, std, yerr=std_error, linestyle='', marker='.', ecolor='r', markersize=5)
        plt.plot(sample_times, std_fit, linestyle='--', color='g')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Time')
        plt.ylabel(r'$\omega$')
        plt.legend(['Fitted line', 'Samples'])
        plt.savefig('images/q4_' + file_name + '_omega.png')
        plt.show()

        print('Avg Slope: ', avg_fit_para[0], ' ± ', avg_fit_error[0])
        # print('Avg Coffi: ', avg_fit_para[1], ' ± ', avg_fit_error[1])
        print('Std Slope: ', std_fit_para[0], ' ± ', std_fit_error[0])
        # print('Std Coffi: ', std_fit_para[1], ' ± ', std_fit_error[1])


test = RandomBallisticDeposition(200)
test.render(20000, save=True)
test.show()
times = np.exp(np.arange(1, 14, .5)).astype(int)
test.make_ensemble(times, 200, file_name='200_200_1_14_.5')
test.analyse(file_name='200_200_1_14_.5')
