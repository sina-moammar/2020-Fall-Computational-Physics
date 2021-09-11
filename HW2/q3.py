import numpy as np
import matplotlib.pyplot as plt


class RandomSierpinskiTriangle:
    def __init__(self, num_points, time=0):
        n = max(num_points, np.power(3, time))
        self.num_points = num_points
        self.width = np.sqrt(n * 4 / np.sqrt(3))
        self.height = np.sqrt(3) / 2 * self.width
        self.points = []
        self.time = 0

    def _random_points(self, num_points):
        random_points = np.random.rand(num_points, 2) * [self.width / 2, self.height]
        indexes = random_points[:, 1] > (random_points[:, 0] * np.sqrt(3))
        random_points[indexes, 1] = self.height - random_points[indexes, 1]
        random_points[indexes, 0] = random_points[indexes, 0] + self.width / 2
        return random_points

    def _random_transform(self, old_points):
        shift_options = np.array([[0, 0], [self.width / 2, 0], [self.width / 4, self.height / 2]])
        shifts_indexes = np.random.randint(0, len(shift_options), self.num_points)
        new_points = old_points / 2 + shift_options[shifts_indexes]
        return new_points

    def render(self, time):
        self.time = time
        self.points = self._random_points(self.num_points)

        for t in range(time):
            self.points = self._random_transform(self.points)

    def show(self):
        image_width = np.int(np.ceil(self.width))
        image_height = np.int(np.ceil(self.height))
        image = np.ones((image_height, image_width))
        indexes = self.points.astype(int)
        image[indexes[:, 1], indexes[:, 0]] = 0

        plt.gray()
        fig, ax = plt.subplots(dpi=300)
        ax.tick_params(axis='both', which='both', bottom=False, left=False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.axis('equal')
        ax.axis('off')
        plt.pcolormesh(image)
        plt.savefig('images/q3_' + str(self.num_points) + '_' + str(self.time) + '.png', pad_inches=0, bbox_inches='tight')
        plt.show()


sample = RandomSierpinskiTriangle(10**6, 9)
sample.render(9)
sample.show()
