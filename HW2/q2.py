import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


class SierpinskiTriangle:
    def __init__(self, length):
        self.length = length
        self.stages_triangles = [np.array([[[0, 0], [self.length, 0], [self.length / 2, self.length * np.sqrt(3) / 2]]])]
        self.time = 0

    def _transform(self, old_triangles):
        new_triangles = np.empty((len(old_triangles) * 3, 3, 2))
        length = len(old_triangles)
        shift = self.length / 2
        scaled_triangles = (old_triangles / 2)
        new_triangles[:length] = scaled_triangles
        new_triangles[length:2 * length] = scaled_triangles + np.array([shift, 0])
        new_triangles[2 * length:] = scaled_triangles + np.array([.5, np.sqrt(3) / 2]) * shift
        return new_triangles

    def render(self, time):
        self.time = time
        self.stages_triangles = [self.stages_triangles[0]]

        for t in range(time):
            new_triangles = self._transform(self.stages_triangles[t])
            self.stages_triangles.append(new_triangles)

    def show(self):
        for t in range(self.time + 1):
            fig, ax = plt.subplots(figsize=(self.length, self.length * np.sqrt(3) / 2))
            ax.tick_params(axis='both', which='both', bottom=False, left=False)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            shapes = []
            for triangle in self.stages_triangles[t]:
                shape = Polygon(triangle, fill=True)
                shapes.append(shape)
            patch = PatchCollection(shapes)
            ax.add_collection(patch)
            ax.axis('equal')
            ax.axis('off')
            plt.savefig('images/q2-stage-' + str(t) + '.png', pad_inches=0, bbox_inches='tight')
            plt.show()


sample = SierpinskiTriangle(6)
sample.render(5)
sample.show()
