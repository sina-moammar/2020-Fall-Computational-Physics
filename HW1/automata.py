from bitarray import bitarray
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math
from typing import Union


class CA1D:
    grids = None
    length = 0
    time = 0
    rule = 0

    def __init__(self, length: int, initial: Union[list, str]):
        self.length = length
        self.grids = []
        if initial == 'rand':
            self.grids.append(bitarray([random.choice([True, False]) for i in range(length)]))
        elif isinstance(initial, list):
            self.grids = [bitarray('0' * length)]
            for index in initial:
                self.grids[0][index - 1] = 1

    def _LUT(self, rule: int, size: int):
        lut_dict = {}
        rule_bin = bin(rule)[2:]
        rule_bin = '0' * (2 ** size - len(rule_bin)) + rule_bin
        rule_length = len(rule_bin)
        for i in range(rule_length):
            key = bin(rule_length - 1 - i)[2:]
            key = ('0' * (size - len(key))) + key
            lut_dict.update({key: int(rule_bin[i])})

        return lut_dict

    def render(self, rule: int, time: int):
        self.time = time
        self.rule = rule
        self.grids = [self.grids[0]]
        lut = self._LUT(rule, 3)
        for t in range(time):
            row = bitarray(self.length)
            for i in range(self.length):
                key = bitarray(0)
                key.append(self.grids[t][i - 1])
                key.append(self.grids[t][i])
                key.append(self.grids[t][(i + 1) % self.length])
                row[i] = lut.get(key.to01())

            self.grids.append(row)

    def show(self, name=None):
        image_data = [[not state for state in self.grids[t]] for t in range(self.time)]
        plt.gray()
        dpi = math.ceil(1000 / max(self.length, self.time + 1))
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(self.length, self.time + 1), dpi=dpi)
        plt.tick_params(axis='both', which='both', bottom=False, left=False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.imshow(image_data, interpolation='nearest', origin='lower', vmin=False)
        plt.tight_layout()
        if name is not None:
            plt.savefig('images/' + name + '.png')
        plt.show()


class GameOfLife:
    grids = None
    width = 0
    height = 0
    time = 0

    def __init__(self, width: int, height: int, initial: Union[list, str]):
        self.width, self.height = width, height
        self.grids = []
        if isinstance(initial, str):
            if initial == 'rand':
                self.grids.append([[random.choice([True, False]) for w in range(width)] for h in range(height)])
            else:
                rows = initial.split('\n')
                self.grids.append([list(map(lambda state: False if state == '*' else True, list(row))) for row in rows])

        elif isinstance(initial, list):
            self.grids.append([[True for w in range(width)] for h in range(height)])
            for index in initial:
                self.grids[0][index[0] - 1][index[1] - 1] = False

    def _live_neighbors(self, grid: list, h: int, w: int) -> int:
        count = 0
        for i in range(h - 1, h + 2):
            for j in range(w - 1, w + 2):
                if not grid[(i + self.height) % self.height][(j + self.width) % self.width]:
                    count += 1

        return count

    def _new_state(self, prev_state, live_neighbors):
        if prev_state:
            if live_neighbors == 3:
                return False
            else:
                return True
        else:
            if 2 < live_neighbors < 5:
                return False
            else:
                return True

    def render(self, time: int):
        self.time = time
        self.grids = [self.grids[0]]
        for t in range(time):
            grid = []
            for h in range(self.height):
                row = []
                for w in range(self.width):
                    live_neighbors = self._live_neighbors(self.grids[t], h, w)
                    row.append(self._new_state(self.grids[t][h][w], live_neighbors))

                grid.append(row)

            self.grids.append(grid)

    def animate(self, interval=500, name='game_of_life'):
        plt.gray()
        dpi = math.ceil(1000 / max(self.width, self.height))
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(self.width, self.height), dpi=dpi)
        ax.set_xticks([i - .5 for i in range(self.width)])
        ax.set_yticks([i - .5 for i in range(self.height)])
        plt.tick_params(axis='both', which='both', bottom=False, left=False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.grid(True)
        images = []
        for grid in self.grids:
            images.append([ax.imshow(grid, interpolation='nearest', animated=True, vmin=False)])

        ani = animation.ArtistAnimation(fig, images, interval=interval, blit=True, repeat_delay=1000)
        ani.save('game_of_life_animations/' + name + '.mp4')
