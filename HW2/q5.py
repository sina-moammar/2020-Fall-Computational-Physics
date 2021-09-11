import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class RandomBallisticDepositionWithRelaxation:
    def __init__(self, length):
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
        for (t, current_time) in enumerate(samples_times):
            self.samples_heights[t][:] = self.samples_heights[t - 1][:]
            randoms = np.random.randint(0, self.length, current_time - prev_time, dtype=int)
            for index in randoms:
                left = self.samples_heights[t][index - 1]
                mid = self.samples_heights[t][index]
                right = self.samples_heights[t][(index + 1) % self.length]
                min_height = np.min([left, mid, right])
                if min_height == mid:
                    final_index = index
                elif left == right:
                    final_index = np.random.choice([index - 1, index + 1])
                elif min_height == left:
                    final_index = index - 1
                else:
                    final_index = index + 1
                self.samples_heights[t][final_index % self.length] += 1
            prev_time = current_time

        if save:
            data = {
                'times': self.samples_times,
                'heights': self.samples_heights,
                'time': self.time,
                'length': self.length
            }
            np.save("data/q5_" + str(self.length) + '_' + str(self.time), data)

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
        plt.savefig('images/q5_' + str(self.length) + '_' + str(self.time) + '.png', pad_inches=0, bbox_inches='tight')
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
            print(t)
            self.render(samples_times=sample_times)
            ensemble_avg_s[t] = np.average(self.samples_heights, axis=1)
            ensemble_std_s[t] = np.std(self.samples_heights, axis=1)

        data = {
            'avgs': ensemble_avg_s,
            'stds': ensemble_std_s,
            'times': sample_times
        }
        if file_name is not None:
            np.save("data/q5_" + file_name + "_ensemble", data)

    def analyse(self, file_name):
        data = np.load('data/q5_' + file_name + '_ensemble.npy', allow_pickle=True).tolist()
        ensemble_avg_s = data['avgs']
        ensemble_std_s = data['stds']
        sample_times = data['times']
        sample_times_ln = np.log(sample_times)

        avg = np.average(ensemble_avg_s, axis=0)
        avg_ln = np.log(avg)
        avg_error = np.std(ensemble_avg_s, axis=0)
        avg_error[avg_error == 0] = 10**(-16)
        avg_error_ln = avg_error / avg
        std = np.average(ensemble_std_s, axis=0)
        std_ln = np.log(std)
        std_error = np.std(ensemble_std_s, axis=0)
        std_error[std_error == 0] = 10 ** (-16)
        std_error_ln = std_error / std

        def model_const_func(t, a):
            return a

        def model_line_func(t, a, b):
            return a * t + b

        valid_indexes = [9, -7]
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

        saturation_index = -6
        saturation_fit_para, saturation_fit_error = curve_fit(model_const_func, sample_times[saturation_index:],
                                                              std[saturation_index:],
                                                              sigma=std_error[saturation_index:])
        saturation_fit_error = np.diag(saturation_fit_error)
        saturation_fit = saturation_fit_para[0] * np.ones(len(sample_times))

        plt.errorbar(sample_times, avg, yerr=avg_error, linestyle='', marker='.', ecolor='r', markersize=5)
        plt.plot(sample_times, avg_fit, linestyle='--', color='g')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Time')
        plt.ylabel('Average Height')
        plt.legend(['Fitted line', 'Samples'])
        plt.savefig('images/q5_' + file_name + '_avg.png')
        plt.show()

        plt.errorbar(sample_times, std, yerr=std_error, linestyle='', marker='.', ecolor='r', markersize=5)
        plt.plot(sample_times, std_fit, linestyle='--', color='g')
        plt.plot(sample_times, saturation_fit, linestyle='-.', color='g')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Time')
        plt.ylabel(r'$\omega$')
        plt.legend(['Fitted line', 'Saturation', 'Samples'])
        plt.savefig('images/q5_' + file_name + '_omega.png')
        plt.show()

        print('Avg Slope: ', avg_fit_para[0], ' ± ', avg_fit_error[0])
        # print('Avg Coffi: ', avg_fit_para[1], ' ± ', avg_fit_error[1])
        print('Std Slope: ', std_fit_para[0], ' ± ', std_fit_error[0])
        # print('Std Coffi: ', std_fit_para[1], ' ± ', std_fit_error[1])
        print('Saturation: ', saturation_fit_para[0], ' ± ', np.abs(saturation_fit_error[0]))


def calc_z(saturation_values):
    def model_line_func(t, a, b):
        return a * t + b

    saturation_s = np.array(saturation_values)
    saturation_s_ln = np.log(saturation_s)
    length_s = np.array([200, 150, 100, 50])
    length_s_ln = np.log(length_s)

    sat_fit_para, sat_fit_error = curve_fit(model_line_func, length_s_ln, saturation_s_ln)
    sat_fit_error = np.diag(sat_fit_error)
    sat_fit = np.exp(sat_fit_para[1] + length_s_ln * sat_fit_para[0])
    print('Sat Slope: ', sat_fit_para[0], ' ± ', sat_fit_error[0])

    plt.plot(length_s, sat_fit, linestyle='--', color='g')
    plt.plot(length_s, saturation_s, linestyle='', marker='.', markersize=5)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Length')
    plt.ylabel(r'$\omega_s$')
    plt.legend(['Fitted line', 'Samples'])
    plt.savefig('images/q5_z.png')
    plt.show()


test = RandomBallisticDepositionWithRelaxation(50)
times = np.exp(np.arange(1, 13, .4)).astype(int)
test.make_ensemble(times, 50, file_name='50_50_1_13_.4')
test.analyse(file_name='50_50_1_13_.4')

test = RandomBallisticDepositionWithRelaxation(100)
times = np.exp(np.arange(1, 14, .4)).astype(int)
test.make_ensemble(times, 50, file_name='100_50_1_14_.4')
test.analyse(file_name='100_50_1_14_.4')

test = RandomBallisticDepositionWithRelaxation(150)
times = np.exp(np.arange(1, 14, .4)).astype(int)
test.make_ensemble(times, 50, file_name='150_50_1_14_.4')
test.analyse(file_name='150_50_1_14_.4')

test = RandomBallisticDepositionWithRelaxation(200)
times = np.exp(np.arange(1, 15, .4)).astype(int)
test.make_ensemble(times, 50, file_name='200_50_1_15_.4')
test.analyse(file_name='200_50_1_15_.4')
test.render(20000, save=True)
test.show()

calc_z([3.124, 2.655, 2.168, 1.5575])
