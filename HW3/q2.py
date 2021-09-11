import numpy as np
import matplotlib.pyplot as plt


class Percolation:
    def __init__(self, length):
        self.length = length
        self.grid = np.zeros((length + 2, length + 2), dtype=int)
        self.p = 0
        self.labels = [0]
        self.sizes = [self.length * self.length]

    def __find_label(self, label):
        if label == self.labels[label]:
            return label
        else:
            self.labels[label] = self.__find_label(self.labels[label])
            return self.labels[label]

    def render(self, p):
        self.p = p
        probability = np.random.rand(self.length, self.length)
        active_cells = probability < p
        self.grid[1:-1, 1:-1] = active_cells
        active_count = np.count_nonzero(active_cells)
        self.labels = np.zeros(active_count + 1, dtype=int)
        self.sizes = np.zeros(active_count + 1, dtype=int)
        self.sizes[0] = (self.length * self.length) - active_count

        prev_label = 0
        for col in range(1, self.length + 1):
            for row in range(1, self.length + 1):
                if self.grid[row][col] != 0:
                    top_label = self.grid[row - 1][col]
                    left_label = self.grid[row][col - 1]

                    if top_label != 0:
                        if left_label != 0:
                            left_final_label = self.__find_label(left_label)
                            top_final_label = self.__find_label(top_label)
                            self.grid[row - 1][col] = top_final_label
                            self.grid[row][col - 1] = left_final_label
                            self.grid[row][col] = left_final_label
                            if left_final_label != top_final_label:
                                self.labels[top_final_label] = left_final_label
                                self.sizes[left_final_label] += (self.sizes[top_final_label] + 1)
                                self.sizes[top_final_label] = 0
                            else:
                                self.sizes[left_final_label] += 1
                        else:
                            top_final_label = self.__find_label(top_label)
                            self.grid[row - 1][col] = top_final_label
                            self.grid[row][col] = top_final_label
                            self.sizes[top_final_label] += 1
                    elif left_label != 0:
                        left_final_label = self.__find_label(left_label)
                        self.grid[row][col - 1] = left_final_label
                        self.grid[row][col] = left_final_label
                        self.sizes[left_final_label] += 1
                    else:
                        prev_label += 1
                        self.labels[prev_label] = prev_label
                        self.sizes[prev_label] = 1
                        self.grid[row][col] = prev_label

        for row in range(1, self.length + 1):
            for col in range(1, self.length + 1):
                self.grid[row][col] = self.__find_label(self.grid[row][col])

    def infinity_clusters(self):
        start_unique_hor_labels = np.unique(self.grid[1:-1, 1])
        end_unique_hor_labels = np.unique(self.grid[1:-1, -2])
        hor_clusters = np.intersect1d(start_unique_hor_labels, end_unique_hor_labels, assume_unique=True)
        infinity_hor_clusters_indexes = np.nonzero(hor_clusters)[0]
        infinity_hor_clusters = hor_clusters[infinity_hor_clusters_indexes]

        start_unique_ver_labels = np.unique(self.grid[1, 1:-1])
        end_unique_ver_labels = np.unique(self.grid[-2, 1:-1])
        ver_clusters = np.intersect1d(start_unique_ver_labels, end_unique_ver_labels, assume_unique=True)
        infinity_ver_clusters_indexes = np.nonzero(ver_clusters)[0]
        infinity_ver_clusters = ver_clusters[infinity_ver_clusters_indexes]

        infinity_clusters = np.unique(np.append(infinity_hor_clusters, infinity_ver_clusters))
        return infinity_clusters, self.sizes[infinity_clusters]

    def is_percolated(self):
        clusters, sizes = self.infinity_clusters()
        return clusters[0] if len(clusters) > 0 else 0

    def correlation_length(self):
        clusters_coordination = [[] for i in range(len(self.labels))]
        for row in range(1, self.length + 1):
            for col in range(1, self.length + 1):
                label = self.grid[row][col]
                if label != 0:
                    clusters_coordination[label].append(col + row * 1j)

        inf_label = self.is_percolated()
        clusters_sizes = self.sizes.copy()
        clusters_sizes[0] = 0
        clusters_sizes[inf_label] = 0
        largest_cluster_index = np.argmax(clusters_sizes)
        if len(clusters_coordination[largest_cluster_index]) > 0:
            return np.std(clusters_coordination[largest_cluster_index])
        else:
            return 0

    def show(self):
        image = self.grid[1:-1, 1:-1]
        fig, ax = plt.subplots()
        plt.tick_params(axis='both', which='both', bottom=False, left=False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_aspect('equal', 'box')
        plt.pcolormesh(image, cmap='gist_stern_r')
        plt.savefig('images/q2_' + str(self.length) + '_' + str(self.p) + '.png', pad_inches=0, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    model = Percolation(100)
    model.render(.4)
    model.show()

    model.render(.6)
    model.show()
