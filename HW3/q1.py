import numpy as np
import matplotlib.pyplot as plt


class Percolation:
    def __init__(self, length):
        self.length = length
        self.grid = np.zeros((length, length), dtype=bool)
        self.p = 0

    def render(self, p):
        self.p = p
        probability = np.random.rand(self.length, self.length)
        self.grid = probability < p

    def show(self):
        image = np.bitwise_not(self.grid)
        plt.gray()
        fig, ax = plt.subplots()
        plt.tick_params(axis='both', which='both', bottom=False, left=False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_aspect('equal', 'box')
        plt.pcolormesh(image)
        plt.savefig('images/q1_' + str(self.length) + '_' + str(self.p) + '.png', pad_inches=0, bbox_inches='tight')
        plt.show()


model = Percolation(100)
model.render(.4)
model.show()

model.render(.9)
model.show()
