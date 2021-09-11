import numpy as np
import matplotlib.pyplot as plt


class DiffusionLimitedAggregation:
    def __init__(self, length):
        self.length = length
        self.grid = []
        self.time = 0
        self.height_limit_diff = 5

    def __is_stuck(self, i, j):
        return self.grid[j][(i + 1) % self.length] != 0 \
               or self.grid[j][(i - 1) % self.length] != 0 \
               or self.grid[j + 1][i] != 0 \
               or self.grid[j - 1][i] != 0

    def __final_position(self, i, j):
        lower_limit_height = j
        upper_limit_height = lower_limit_height + self.height_limit_diff
        random_mean_steps = lower_limit_height ** 2
        steps_states = np.array([(0, 1), (0, -1), (1, 0), (-1, 0)])

        while True:
            random_steps_indexes = np.random.randint(0, len(steps_states), random_mean_steps)
            random_steps = steps_states[random_steps_indexes]
            for step_i, step_j in random_steps:
                if j > upper_limit_height:
                    return None
                elif self.__is_stuck(i, j):
                    return i, j

                i = (i + step_i) % self.length
                j = j + step_j

    def render(self, time, save=False):
        self.time = time
        self.grid = [[0 for i in range(self.length)] for j in range(self.height_limit_diff + 3)]
        self.grid[0] = [1] * self.length
        lower_limit_height = 1
        particles_count = self.length
        color_count = 2 * self.length

        while True:
            random_initial_position = np.random.randint(0, self.length, 2 * self.time)
            for initial_position in random_initial_position:
                new_position = self.__final_position(initial_position, lower_limit_height)
                if new_position is None:
                    continue
                else:
                    particles_count += 1
                    print('\r' + str(particles_count - self.length), end='')
                    self.grid[new_position[1]][new_position[0]] = int(particles_count / color_count) + 1

                    if new_position[1] >= lower_limit_height:
                        lower_limit_height = new_position[1] + 1
                        self.grid.append([0] * self.length)

                    if particles_count == self.length + time:
                        if save:
                            data = {
                                'length': self.length,
                                'time': self.time,
                                'grid': self.grid,
                            }
                            np.save("data/q6_" + str(self.length) + '_' + str(self.time), data)
                        return

    def load(self, file_name):
        data = np.load('data/q6_' + file_name + '.npy', allow_pickle=True).tolist()
        self.length = data['length']
        self.time = data['time']
        self.grid = data['grid']

    def show(self):
        fig, ax = plt.subplots()
        plt.tick_params(axis='both', which='both', bottom=False, left=False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_aspect('equal', 'box')
        plt.pcolormesh(self.grid, cmap='nipy_spectral')
        plt.savefig('images/q6_' + str(self.length) + '_' + str(self.time) + '.png', pad_inches=0, bbox_inches='tight')
        plt.show()


model = DiffusionLimitedAggregation(200)
model.render(2000)
model.show()
