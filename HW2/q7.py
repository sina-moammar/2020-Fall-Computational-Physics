import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class NearestNeighborBallisticDepositionWithInitialCondition:
    def __init__(self, length):
        self.length = length
        self.time = 0
        self.samples_times = []
        self.samples_heights = []
        self.samples_cells = []

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
        self.samples_heights[-1][int(self.length / 2)] = 1
        self.samples_cells = []

        prev_time = 0
        for (t, current_time) in enumerate(samples_times):
            self.samples_heights[t][:] = self.samples_heights[t - 1][:]
            randoms = np.random.randint(0, self.length, current_time - prev_time, dtype=int)
            cells = np.zeros((current_time - prev_time, 2), dtype=int)
            for i in range(len(randoms)):
                index = randoms[i]
                left = self.samples_heights[t][index - 1]
                mid = self.samples_heights[t][index]
                mid = mid + 1 if mid > 0 else mid
                right = self.samples_heights[t][(index + 1) % self.length]
                new_height = np.max([left, mid, right])
                if new_height > 0:
                    self.samples_heights[t][index] = new_height
                    cells[i] = (index, new_height - 1)
            prev_time = current_time
            self.samples_cells.append(cells)

        if save:
            data = {
                'times': self.samples_times,
                'heights': self.samples_heights,
                'time': self.time,
                'length': self.length
            }
            np.save("data/q7_" + str(self.length) + '_' + str(self.time), data)

    def show(self):
        max_heights = [np.max(cells[:, 1]) for cells in self.samples_cells]
        max_height = np.max(max_heights)
        images = np.zeros((max_height + 1, self.length))
        images.nonzero()
        is_blue = True
        for sample_cells in self.samples_cells:
            images[sample_cells[:, 1], sample_cells[:, 0]] = 1 if is_blue else 2
            is_blue = not is_blue

        plt.tick_params(axis='both', which='both', bottom=False, left=False)
        plt.gca().set_yticklabels([])
        plt.gca().set_xticklabels([])
        plt.pcolormesh(images)
        plt.axis('equal')
        plt.axis('off')
        plt.savefig('images/q7_' + str(self.length) + '_' + str(self.time) + '.png', pad_inches=0, bbox_inches='tight')
        plt.show()

    def load(self, file_name):
        data = np.load('data/' + file_name, allow_pickle=True).tolist()
        self.length = data['length']
        self.time = data['time']
        self.samples_times = data['times']
        self.samples_heights = data['heights']

    def __calc_width(self):
        widths = np.zeros(len(self.samples_times))
        for t, heights in enumerate(self.samples_heights):
            non_zero_indexes = heights.nonzero()[0]
            if len(non_zero_indexes) > 0:
                widths[t] = non_zero_indexes[-1] - non_zero_indexes[0] + 1

        return widths


    def make_ensemble(self, sample_times, num, file_name=None):
        ensemble_avg_s = np.zeros((num, len(sample_times)))
        ensemble_std_s = np.zeros((num, len(sample_times)))
        ensemble_width_s = np.zeros((num, len(sample_times)))
        for t in range(num):
            print(t)
            self.render(samples_times=sample_times)
            ensemble_avg_s[t] = np.average(self.samples_heights, axis=1)
            ensemble_std_s[t] = np.std(self.samples_heights, axis=1)
            ensemble_width_s[t] = self.__calc_width()

        data = {
            'avgs': ensemble_avg_s,
            'stds': ensemble_std_s,
            'widths': ensemble_width_s,
            'times': sample_times
        }
        if file_name is not None:
            np.save("data/q7_" + file_name + "_ensemble", data)

    def analyse(self, file_name=None):
        data = np.load('data/q7_' + file_name + '_ensemble.npy', allow_pickle=True).tolist()
        ensemble_avg_s = data['avgs']
        ensemble_std_s = data['stds']
        ensemble_width_s = data['widths']
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
        width = np.average(ensemble_width_s, axis=0)
        width_ln = np.log(width)
        width_error = np.std(ensemble_width_s, axis=0)
        width_error[width_error == 0] = 10 ** (-16)
        width_error_ln = width_error / width

        def model_line_func(t, a, b):
            return a * t + b

        valid_indexes = [16, -2]
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

        width_fit_para, width_fit_error = curve_fit(model_line_func, sample_times_ln[valid_indexes[0]:valid_indexes[1]],
                                                width_ln[valid_indexes[0]:valid_indexes[1]],
                                                sigma=width_error_ln[valid_indexes[0]:valid_indexes[1]])
        width_fit_error = np.diag(width_fit_error)
        width_fit = np.exp(width_fit_para[1] + sample_times_ln * width_fit_para[0])

        plt.errorbar(sample_times, avg, yerr=avg_error, linestyle='', marker='.', ecolor='r', markersize=5)
        plt.plot(sample_times, avg_fit, linestyle='--', color='g')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Time')
        plt.ylabel('Average Height')
        plt.legend(['Fitted line', 'Samples'])
        plt.savefig('images/q7_' + file_name + '_avg.png')
        plt.show()

        plt.errorbar(sample_times, std, yerr=std_error, linestyle='', marker='.', ecolor='r', markersize=5)
        plt.plot(sample_times, std_fit, linestyle='--', color='g')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Time')
        plt.ylabel(r'$\omega$')
        plt.legend(['Fitted line', 'Samples'])
        plt.savefig('images/q7_' + file_name + '_omega.png')
        plt.show()

        plt.errorbar(sample_times, width, yerr=width_error, linestyle='', marker='.', ecolor='r', markersize=5)
        plt.plot(sample_times, width_fit, linestyle='--', color='g')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Time')
        plt.ylabel(r'Width')
        plt.legend(['Fitted line', 'Samples'])
        plt.savefig('images/q7_' + file_name + '_width.png')
        plt.show()

        print('Avg Slope: ', avg_fit_para[0], ' ± ', avg_fit_error[0])
        # print('Avg Coffi: ', avg_fit_para[1], ' ± ', avg_fit_error[1])
        print('Std Slope: ', std_fit_para[0], ' ± ', std_fit_error[0])
        # print('Std Coffi: ', std_fit_para[1], ' ± ', std_fit_error[1])
        print('Width Slope: ', width_fit_para[0], ' ± ', width_fit_error[0])
        # print('Width Coffi: ', width_fit_para[1], ' ± ', width_fit_error[1])


test = NearestNeighborBallisticDepositionWithInitialCondition(200)
test.render(18000, save=True)
test.show()
times = np.exp(np.arange(1, 9.5, .4)).astype(int)
test.make_ensemble(times, 100, file_name='200_100_1_9.5_.4')
test.analyse(file_name='200_100_1_9.5_.4')
