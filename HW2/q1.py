import numpy as np
import matplotlib.pyplot as plt


class KochSnowflake:
    def __init__(self, length):
        self.length = length
        self.stages_points = [np.array([0, length])]
        self.time = 0

    def _transform(self, old_points):
        new_points = np.empty(len(old_points) * 4 - 3, dtype=np.complex128)
        length = len(old_points) - 1
        shift = self.length / 3
        scaled_points = (old_points / 3)[:-1]
        new_points[:length] = scaled_points
        new_points[length:2 * length] = (scaled_points * np.exp(np.pi / 3 * 1j) + shift)
        new_points[2 * length:3 * length] = (scaled_points * np.exp(np.pi / 3 * -1j) + shift * (1.5 + np.sqrt(3) / 2 * 1j))
        new_points[3 * length:-1] = (scaled_points + 2 * shift)
        new_points[-1] = old_points[-1]
        return new_points

    def render(self, time):
        self.time = time
        self.stages_points = [self.stages_points[0]]

        for t in range(time):
            new_points = self._transform(self.stages_points[t])
            self.stages_points.append(new_points)

    def show(self):
        for t in range(self.time + 1):
            fig, ax = plt.subplots(figsize=(self.length, self.length * np.sqrt(3) / 6), dpi=300)
            ax.tick_params(axis='both', which='both', bottom=False, left=False)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.plot(self.stages_points[t].real, self.stages_points[t].imag, linewidth=.5, color='black')
            ax.axis('equal')
            ax.axis('off')
            plt.savefig('images/q1-stage-' + str(t) + '.png', pad_inches=0, bbox_inches='tight')
            plt.show()


sample = KochSnowflake(6)
sample.render(5)
sample.show()
